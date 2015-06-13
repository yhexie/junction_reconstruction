#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <boost/filesystem.hpp>
#include "color_map.h"
#include "trajectories.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <queue>
#include "gps_trajectory.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <pcl/common/centroid.h>
#include <pcl/search/impl/flann_search.hpp>

#include "latlon_converter.h"

using namespace Eigen;

static const float search_distance = 250.0f; // in meters
static const float arrow_size = 50.0f;   // in meters
static const int arrow_angle = 15; // in degrees

static const float SPEED_FILTER = 2.5; // meters per second

struct QueueValue{
    int index;
    double dist;
};

class MyComparison{
public:
    MyComparison(){}
    bool operator()(const QueueValue &lhs, const QueueValue& rhs) const{
        return (lhs.dist > rhs.dist);
    }
};

bool myFunction(pair<int, pair<int, int>> p1, pair<int,pair<int, int>> p2) {
    if (p1.second.first > p2.second.first) {
        return true;
    }
    else if (p1.second.first == p2.second.first){
        if (p1.second.second > p2.second.second) {
            return true;
        }
        else{
            return false;
        }
    }
    else{
        return false;
    }
}

Trajectories::Trajectories(QObject *parent) : Renderable(parent), 
    data_(new PclPointCloud), 
    tree_(new pcl::search::FlannSearch<PclPoint>(false)), 
    samples_(new PclPointCloud), 
    sample_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    // Data
    trajectories_.clear();
    min_speed_ = POSITIVE_INFINITY;
    max_speed_ = 0.0f;
    avg_speed_ = 0.0f;
    
    // Sample
    sample_locs_.clear();
    
    // Rendering
        // points
    scale_factor_ = 1.0f;
    point_size_ = 5.0f;
    line_width_ = 3.0f;
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    vertex_speed_.clear();
    vertex_speed_colors_.clear();
    vertex_speed_indices_.clear();
    selected_trajectories_.clear();
    
        // Samples
    show_samples_ = true;
    sample_point_size_ = 10.0f;
    sample_color_ = Color(0.0f, 0.3f, 0.6f, 1.0f);
    samples_->clear();
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    picked_sample_idxs_.clear();
    
    render_mode_ = false;
    selection_mode_ = false;
    show_direction_ = false;
}

Trajectories::~Trajectories(){
}

//////////////////////////////////////////////////
//                      IO
//////////////////////////////////////////////////
bool Trajectories::load(const string& filename)
{
    bool success = false;
    cout << "Loading Trajectories from " << filename << "..."<<endl;
    string extension = boost::filesystem::path(filename).extension().string();
    if (extension == ".txt")
        success = loadTXT(filename);
    else if (extension == ".pcd"){
        //success = loadPCD(filename);
    }
    else if (extension == ".pbf"){
        success = loadPBF(filename);
    }
    
    if (success)
    {
        tree_->setInputCloud(data_);
    }
    
    return true;
}

bool Trajectories::save(const string& filename){
    if (savePBF(filename)){
        printf("%s saved.\n", filename.c_str());
    }
    else{
        printf("File cannot be saved.");
    }
    return true;
}

bool Trajectories::extractFromFiles(const QStringList &filenames, QVector4D bound_box, PclSearchTree::Ptr &map_search_tree, int nmin_inside_pts){
    data_->clear();
    trajectories_.clear();
    
    // Check bound box
    if (bound_box[1] < bound_box[0] || bound_box[3] < bound_box[2]) {
        fprintf(stderr, "Bound Box ERROR when extractng trajectories.\n");
        return false;
    }
    
    for (size_t i = 0; i < filenames.size(); ++i) {
        std::string filename(filenames.at(i).toLocal8Bit().constData());
        printf("\tExtracting from %s\n", filename.c_str());
        Trajectories *new_trajectories = new Trajectories(NULL);
        new_trajectories->load(filename);
        new_trajectories->extractTrajectoriesFromRegion(bound_box, this, map_search_tree, nmin_inside_pts);
        delete new_trajectories;
    }
    
    tree_->setInputCloud(data_);
    
    printf("Totally %lu trajectories, %lu points.\n", trajectories_.size(), data_->size());
    
    return true;
}

bool Trajectories::extractTrajectoriesFromRegion(QVector4D bound_box, Trajectories *container,  PclSearchTree::Ptr &map_search_tree, int min_inside_pts){
    // Find all potential trajectories indices inside using kdtree search
    PclPoint center_point;
    float x_length = bound_box[1] - bound_box[0];
    float y_length = bound_box[3] - bound_box[2];
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    center_point.setCoordinate(center_x, center_y, 0.0f);
    float radius = 0.5*sqrt(x_length*x_length + y_length*y_length);
    
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(center_point, radius, k_indices, k_distances);
    
    set<int> inside_trajectory_idxs;
    for (size_t i = 0; i < k_indices.size(); ++i)
    {
        const PclPoint &point = data_->at(k_indices[i]);
        inside_trajectory_idxs.insert(point.id_trajectory);
    }
    
    for (set<int>::iterator it = inside_trajectory_idxs.begin(); it != inside_trajectory_idxs.end(); ++it) {
        vector<int> &trajectory = trajectories_[*it];
        vector<PclPoint> extracted_trajectory;
        bool recording = false;
        
        // Traverse trajectory to chop segments.
        for (size_t pt_idx = 0; pt_idx < trajectory.size(); ++pt_idx) {
            const PclPoint &point = data_->at(trajectory[pt_idx]);
            if (point.x < bound_box[1] && point.x > bound_box[0] && point.y < bound_box[3] && point.y > bound_box[2]) {
                // Point is inside query bound_box
                // Check if point is near the map
                vector<int> k_idxs;
                vector<float> k_dist_sqrs;
                map_search_tree->nearestKSearch(point, 1, k_idxs, k_dist_sqrs);
                bool to_skip = false;
                if (k_idxs.size() != 0) {
                    if (k_dist_sqrs[0] > 625.0f) {
                        to_skip = true;
                    }
                }
                if (to_skip) {
                    // If it is recording, then this trajectory is not qualified
                    recording = false;
                    extracted_trajectory.clear();
                    continue;
                }
                
                if (!recording) {
                    recording = true;
                    extracted_trajectory.clear();
//                    if (pt_idx > 0) {
//                        // Insert previous point
//                        PclPoint &prev_point = data_->at(trajectory[pt_idx-1]);
//                        extracted_trajectory.push_back(prev_point);
//                    }
                    // Insert this point
                    extracted_trajectory.push_back(point);
                }
                else{
                    extracted_trajectory.push_back(point);
                }
            }
            else{
                // Point is outside query bound_box
                if (recording) {
                    recording = false;
                    // Insert this point
                    //extracted_trajectory.push_back(point);
                    if (extracted_trajectory.size() > min_inside_pts + 2) {
                        if(isQualifiedTrajectory(extracted_trajectory))
                            container->insertNewTrajectory(extracted_trajectory);
                    }
                }
            }
        }
        if (recording){
            if (extracted_trajectory.size() > min_inside_pts + 2) {
                if (isQualifiedTrajectory(extracted_trajectory))
                    container->insertNewTrajectory(extracted_trajectory);
            }
        }
    }
    return true;
}

bool Trajectories::isQualifiedTrajectory(vector<PclPoint> &trajectory){
    // Filter bad trajectories by average speed
    if (trajectory.size() < 2) {
        return false;
    }
    float length = 0.0f;
    for (size_t pt_idx = 0; pt_idx < trajectory.size() - 1; ++pt_idx){
        float delta_x = trajectory[pt_idx+1].x - trajectory[pt_idx].x;
        float delta_y = trajectory[pt_idx+1].y - trajectory[pt_idx].y;
        length += sqrt(delta_x*delta_x + delta_y*delta_y);
    }
    float delta_t = trajectory[trajectory.size()-1].t - trajectory[0].t;
    
    float avg_velocity = length / delta_t;
    if (avg_velocity < SPEED_FILTER) {
        return false;
    }
    return true;
}

bool Trajectories::insertNewTrajectory(vector<PclPoint> &pt_container){
    if (pt_container.empty()) {
        return true;
    }
    
    vector<int> trajectory;
    trajectory.resize(pt_container.size());
    
    for (size_t i = 0; i < pt_container.size(); ++i) {
        PclPoint &point = pt_container[i];
        
        point.id_trajectory = trajectories_.size();
        point.id_sample = i;
        trajectory[i] = data_->size();
        data_->push_back(point);
    }
    
    trajectories_.push_back(trajectory);
    return true;
}

bool Trajectories::loadTXT(const string& filename)
{
    ifstream fin(filename);
    if (!fin.good())
        return false;
    
    PclPoint point;
    point.setNormal(0, 0, 1);
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    
    int num_trajectory = 0;
    fin >> num_trajectory;
    trajectories_.resize(num_trajectory, vector<int>());
    
    QVector4D bound_box(SceneConst::getInstance().getBoundBox());
    
    for (int id_trajectory = 0; id_trajectory < num_trajectory; ++ id_trajectory)
    {
        int num_samples;
        fin >> num_samples;
        vector<int>& trajectory = trajectories_[id_trajectory];
        trajectory.resize(num_samples);
        for (int id_sample = 0; id_sample < num_samples; ++ id_sample)
        {
            double x, y, t;
            int heavy;
            fin >> x >> y >> t >> heavy;
            x -= BJ_X_OFFSET;
            y -= BJ_Y_OFFSET;
            t -= UTC_OFFSET;
            
            if (x < bound_box[0]) {
                bound_box[0] = x;
            }
            if (x > bound_box[1]){
                bound_box[1] = x;
            }
            if (y < bound_box[2]) {
                bound_box[2] = y;
            }
            if (y > bound_box[3]){
                bound_box[3] = y;
            }
            
            point.setCoordinate(x, y, 0.0);
            point.t = static_cast<float>(t);
            point.id_trajectory = id_trajectory;
            point.id_sample = id_sample;
            
            trajectory[id_sample] = data_->size();
            data_->push_back(point);
        }
    }
    
    SceneConst::getInstance().updateBoundBox(bound_box);
    
    return true;
}

bool Trajectories::loadPBF(const string &filename){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // Prepare the coordinate projector
    Projector &converter = Projector::getInstance();
    bool has_correct_projection = false;
    
    // Read the trajectory collection from file.
    int fid = open(filename.c_str(), O_RDONLY);
    if (fid == -1) {
        fprintf(stderr, "ERROR! Cannot open protobuf trajectory file!\n");
        return false;
    }
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    min_speed_ = POSITIVE_INFINITY;
    max_speed_ = 0.0f;
    avg_speed_ = 0.0f;
    
    google::protobuf::io::ZeroCopyInputStream *raw_input = new google::protobuf::io::FileInputStream(fid);
    google::protobuf::io::CodedInputStream *coded_input = new google::protobuf::io::CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(1e9, 9e8);
    
    PclPoint point;
    point.setNormal(0, 0, 1);
    
    uint32_t num_trajectory;

    QVector4D bound_box = QVector4D(SceneConst::getInstance().getBoundBox());
    
    if (!coded_input->ReadLittleEndian32(&num_trajectory)) {
        return false; // possibly end of file
    }

    if(num_trajectory > 1e9){
        cout << "ERROR: More than 1e9 trajectories detected from loadPBF!" << endl;
        cout << "\tnum_traj: " << num_trajectory << endl;
        return false;
    }

    trajectories_.clear();
    for (size_t id_traj = 0; id_traj < num_trajectory; ++id_traj) {
        uint32_t length;
        if (!coded_input->ReadLittleEndian32(&length)) {
            break; // possibly end of file
        }
        auto limit = coded_input->PushLimit(length);
        GpsTraj new_traj;
        if(!new_traj.MergePartialFromCodedStream(coded_input)){
            fprintf(stderr, "ERROR! Protobuf trajectory file contaminated!\n");
            return false;
        }
        coded_input->PopLimit(limit);

        if (id_traj == 0) {
            // Check if the x, y field of new_traj point is correct according to current projector. If not, we will reporject all x, y field from lat lon coordinate.
            float x;
            float y;
            converter.convertLatlonToXY(new_traj.point(0).lat()/1.0e5, new_traj.point(0).lon()/1.0e5, x, y);
            
            if ( fabs(x - new_traj.point(0).x()) < 1e-4 &&
                fabs(y - new_traj.point(0).y()) < 1e-4) {
                has_correct_projection = true;
            }
        }

        if(id_traj % 10 >= 5){
            continue;
        }

        // Add the points to point cloud and trajectory
        vector<int> trajectory;
        trajectory.resize(new_traj.point_size());
        if (has_correct_projection) {
            for (int pt_idx=0; pt_idx < new_traj.point_size(); ++pt_idx) {
                if (new_traj.point(pt_idx).x() < bound_box[0]) {
                    bound_box[0] = new_traj.point(pt_idx).x();
                }
                if (new_traj.point(pt_idx).x() > bound_box[1]) {
                    bound_box[1] = new_traj.point(pt_idx).x();
                }
                if (new_traj.point(pt_idx).y() < bound_box[2]) {
                    bound_box[2] = new_traj.point(pt_idx).y();
                }
                if (new_traj.point(pt_idx).y() > bound_box[3]) {
                    bound_box[3] = new_traj.point(pt_idx).y();
                }
                
                point.setCoordinate(new_traj.point(pt_idx).x(), new_traj.point(pt_idx).y(), 0.0);
                point.t = new_traj.point(pt_idx).timestamp();
                point.id_trajectory = id_traj;
                point.id_sample = pt_idx;
                point.car_id = new_traj.point(pt_idx).car_id();
                point.lon = new_traj.point(pt_idx).lon();
                point.lat = new_traj.point(pt_idx).lat();
                point.speed = new_traj.point(pt_idx).speed();

                if(point.speed > max_speed_)
                    max_speed_ = point.speed;
                if(point.speed < min_speed_)
                    min_speed_ = point.speed;
                avg_speed_ += point.speed;
                
                int new_traj_point_head = 450 - new_traj.point(pt_idx).head();
                if (new_traj_point_head > 360) {
                    new_traj_point_head -= 360;
                }
                
                point.head = new_traj_point_head;
               
                trajectory[pt_idx] = data_->size();
                data_->push_back(point);
            }
        }
        else{
            for (int pt_idx=0; pt_idx < new_traj.point_size(); ++pt_idx) {
                // Check if the x, y field is correct. If not, we will redo the projection from lat lon field.
                float x;
                float y;
                converter.convertLatlonToXY(new_traj.point(pt_idx).lat()/1.0e5, new_traj.point(pt_idx).lon()/1.0e5, x, y);
                
                double t = new_traj.point(pt_idx).timestamp();
                
                if (x < bound_box[0]) {
                    bound_box[0] = x;
                }
                if (x > bound_box[1]){
                    bound_box[1] = x;
                }
                if (y < bound_box[2]) {
                    bound_box[2] = y;
                }
                if (y > bound_box[3]){
                    bound_box[3] = y;
                }
                
                point.setCoordinate(x, y, 0.0);
                point.t = t - UTC_OFFSET;
                point.id_trajectory = id_traj;
                point.id_sample = pt_idx;
                point.car_id = new_traj.point(pt_idx).car_id();
                point.lon = new_traj.point(pt_idx).lon();
                point.lat = new_traj.point(pt_idx).lat();
                point.speed = new_traj.point(pt_idx).speed();

                if(point.speed > max_speed_)
                    max_speed_ = point.speed;
                if(point.speed < min_speed_)
                    min_speed_ = point.speed;
                avg_speed_ += point.speed;
                
                int new_traj_point_head = 450 - new_traj.point(pt_idx).head();
                if (new_traj_point_head > 360) {
                    new_traj_point_head -= 360;
                }
               
                point.head = new_traj_point_head;
                
                trajectory[pt_idx] = data_->size();
                data_->push_back(point);
            }
        }
        trajectories_.emplace_back(trajectory);
    }
    
    close(fid);

    if(data_->size() > 0)
        avg_speed_ /= data_->size();
    
    SceneConst::getInstance().updateBoundBox(bound_box);
    
    delete raw_input;
    delete coded_input;
    printf("Loading completed. %zu trajectories loaded. Totally %zu points. Average speed is: %.2fm/s\n", trajectories_.size(), data_->size(), avg_speed_ / 100.0f);
    return true;
}

bool Trajectories::savePBF(const string &filename){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // Read the trajectory collection from file.
    int fid = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
    
    if (fid == -1) {
        fprintf(stderr, "ERROR! Cannot create protobuf trajectory file!\n");
        return false;
    }
    
    google::protobuf::io::ZeroCopyOutputStream *raw_output = new google::protobuf::io::FileOutputStream(fid);
    google::protobuf::io::CodedOutputStream *coded_output = new google::protobuf::io::CodedOutputStream(raw_output);
    
    uint32_t num_trajectory = trajectories_.size();
    
    coded_output->WriteLittleEndian32(num_trajectory);
    
    for (size_t id_traj = 0; id_traj < num_trajectory; ++id_traj) {
        GpsTraj new_traj;
        for (size_t pt_idx = 0; pt_idx < trajectories_[id_traj].size(); ++pt_idx) {
            TrajPoint* new_pt = new_traj.add_point();
            int pt_idx_in_data = trajectories_[id_traj][pt_idx];
            new_pt->set_car_id(data_->at(pt_idx_in_data).car_id);
            new_pt->set_speed(data_->at(pt_idx_in_data).speed);
            
            int old_head = 450 - data_->at(pt_idx_in_data).head;
            if (old_head > 360) {
                old_head -= 360;
            }
            
            new_pt->set_head(old_head);
            new_pt->set_lon(data_->at(pt_idx_in_data).lon);
            new_pt->set_lat(data_->at(pt_idx_in_data).lat);
            new_pt->set_x(data_->at(pt_idx_in_data).x);
            new_pt->set_y(data_->at(pt_idx_in_data).y);
            new_pt->set_timestamp(data_->at(pt_idx_in_data).t);
        }
        string s;
        new_traj.SerializeToString(&s);
        coded_output->WriteLittleEndian32(s.size());
        coded_output->WriteString(s);
    }
    delete coded_output;
    delete raw_output;
    close(fid);
    
    return true;
}

//////////////////////////////////////////////////
//                 Rendering
//////////////////////////////////////////////////
void Trajectories::prepareForVisualization(){
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    normalized_sample_locs_.clear();
    vertex_speed_.clear();
    vertex_speed_indices_.clear();
    vertex_speed_colors_.clear();
    
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (delta_x < 0 || delta_y < 0) {
        fprintf(stderr, "Trajectory bounding box error! Min greater than Max!\n");
    }
    
    scale_factor_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
    
    for (size_t i=0; i<data_->size(); ++i) {
        const PclPoint &point = data_->at(i);
        Vertex v = SceneConst::getInstance().normalize(point.x, point.y, Z_TRAJECTORIES);
        normalized_vertices_.push_back(v);
        
        // Speed and heading
        float speed = static_cast<float>(data_->at(i).speed) / 100.0f;
        
        Color tmp_color(0.8, 0.8, 0.8, 1.0);
        if (data_->at(i).head < 45 || data_->at(i).head >= 315 ) {
            tmp_color.r = 1.0f;
            tmp_color.g = 0.0f;
            tmp_color.b = 0.0f;
            tmp_color.alpha = 1.0f;
        }
        else if(data_->at(i).head >= 45 && data_->at(i).head < 135){
            tmp_color.r = 0.0f;
            tmp_color.g = 1.0f;
            tmp_color.b = 0.0f;
            tmp_color.alpha = 1.0f;
        }
        else if(data_->at(i).head >= 135 && data_->at(i).head < 225){
            tmp_color.r = 0.0f;
            tmp_color.g = 0.0f;
            tmp_color.b = 1.0f;
            tmp_color.alpha = 1.0f;
        }
        else if(data_->at(i).head >= 225 && data_->at(i).head < 315){
            tmp_color.r = 1.0f;
            tmp_color.g = 1.0f;
            tmp_color.b = 0.0f;
            tmp_color.alpha = 1.0f;
        }
        
        float speed_line_length = (speed + 1.0f) / scale_factor_;
        Eigen::Vector2d speed_dir = speed_line_length * headingTo2dVector(data_->at(i).head);
        vertex_speed_indices_.push_back(vertex_speed_.size());
        vertex_speed_.push_back(Vertex(v.x, v.y, Z_TRAJECTORY_SPEED));
        vertex_speed_indices_.push_back(vertex_speed_.size());
        vertex_speed_.push_back(Vertex(v.x+speed_dir.x(), v.y+speed_dir.y(), Z_TRAJECTORY_SPEED));
        
        vertex_speed_colors_.push_back(tmp_color);
        vertex_speed_colors_.push_back(tmp_color);
    }
    
    Color dark_gray = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
    vertex_colors_const_.resize(normalized_vertices_.size(), dark_gray);
    
    //Color orange = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
    //vertex_colors_const_.resize(normalized_vertices_.size(), orange);
    
    vertex_colors_individual_.resize(data_->size(), Color(0,0,0,1));
    for (int i=0; i < trajectories_.size(); ++i) {
        Color traj_color = ColorMap::getInstance().getDiscreteColor(i);
        for (int j = 0; j < trajectories_[i].size(); ++j) {
            int vertex_idx = trajectories_[i][j];
            vertex_colors_individual_[vertex_idx].r = traj_color.r;
            vertex_colors_individual_[vertex_idx].g = traj_color.g;
            vertex_colors_individual_[vertex_idx].b = traj_color.b;
        }
    }
    
    // Prepare for sample visualization
    if (samples_->size() == 0)
        return;
    
    normalized_sample_locs_.resize(samples_->size());
    
    for (size_t i = 0; i < samples_->size(); ++i){
        PclPoint &loc = samples_->at(i);
        Vertex v = SceneConst::getInstance().normalize(loc.x, loc.y, Z_SAMPLES);
        normalized_sample_locs_[i].x = v.x;
        normalized_sample_locs_[i].y = v.y;
        normalized_sample_locs_[i].z = v.z;
    }
}

void Trajectories::draw(){
    if(!render_mode_){
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
        shader_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&vertex_colors_const_[0], 4*vertex_colors_const_.size()*sizeof(float));
        shader_program_->setupColorAttributes();
        
        glPointSize(point_size_);
        glDrawArrays(GL_POINTS, 0, normalized_vertices_.size());
        
        // Draw speed and heading
        if (show_direction_){
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&vertex_speed_[0], 3*vertex_speed_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&vertex_speed_colors_[0], 4*vertex_speed_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(vertex_speed_indices_[0]), vertex_speed_indices_.size()*sizeof(int));
            glLineWidth(line_width_);
            glDrawElements(GL_LINES, vertex_speed_indices_.size(), GL_UNSIGNED_INT, 0);
        }
    }
    else{
        if(trajectories_.size() == 0)
            return;
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
        shader_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&vertex_colors_individual_[0], 4*vertex_colors_individual_.size()*sizeof(float));
        shader_program_->setupColorAttributes();
        
        for (int i=0; i < trajectories_.size(); ++i) {
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(trajectories_[i][0]), trajectories_[i].size()*sizeof(size_t));
            glLineWidth(line_width_);
            glDrawElements(GL_LINE_STRIP, trajectories_[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_);
            glDrawElements(GL_POINTS, trajectories_[i].size(), GL_UNSIGNED_INT, 0);
        }
    }
    
    // Draw selected trajectories
    if (selection_mode_){
        drawSelectedTrajectories(selected_trajectories_);
    }else{
        selected_trajectories_.clear();
    }
    
    // Draw samples
    if (show_samples_){
        if (samples_->size() > 0) {
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&normalized_sample_locs_[0], 3*normalized_sample_locs_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&sample_vertex_colors_[0], 4*sample_vertex_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            
            glPointSize(sample_point_size_);
            glDrawArrays(GL_POINTS, 0, normalized_sample_locs_.size());
            
            // Draw sample heading
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&normalized_sample_headings_[0], 3*normalized_sample_headings_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&sample_heading_colors_[0], 4*sample_heading_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            
            glDrawArrays(GL_LINES, 0, normalized_sample_headings_.size());
            
            // Draw picked samples
            if (picked_sample_idxs_.size() > 0) {
                vector<Vertex> picked_sample_locs;
                vector<Color> picked_sample_colors;
                for (set<int>::iterator it = picked_sample_idxs_.begin(); it != picked_sample_idxs_.end(); ++it) {
                    int sample_id = *it;
                    Vertex &n_sample_loc = normalized_sample_locs_[sample_id];
                    picked_sample_locs.push_back(Vertex(n_sample_loc.x, n_sample_loc.y, Z_SELECTED_SAMPLES));
                    picked_sample_colors.push_back(Color(1.0, 0.0, 0.0, 1.0));
                }
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&picked_sample_locs[0], 3*picked_sample_locs.size()*sizeof(float));
                shader_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&picked_sample_colors[0], 4*picked_sample_colors.size()*sizeof(float));
                shader_program_->setupColorAttributes();
                
                glPointSize(1.5*sample_point_size_);
                glDrawArrays(GL_POINTS, 0, picked_sample_locs.size());
            }
        }
    }
}

///////////////////////////////////////////////////
//              Point Cloud
///////////////////////////////////////////////////
bool Trajectories::isEmpty(){
    if (data_->size() > 0){
        return false;
    }
    else{
        return true;
    }
}

///////////////////////////////////////////////////
//              Trajectories
///////////////////////////////////////////////////


///////////////////////////////////////////////////
//               Sampling
///////////////////////////////////////////////////
void Trajectories::samplePointCloud(float neighborhood_size, PclPointCloud::Ptr &samples, PclSearchTree::Ptr &sample_tree){
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    printf("bound box: %.2f, %.2f, %.2f, %.2f\n", bound_box[0], bound_box[1], bound_box[2], bound_box[3]);
    
    // Clear samples
    PclPoint point;
    samples->clear();
    
    // Compute samples
    // Initialize the samples using a unified grid
    vector<Vertex> seed_samples;
    set<long> grid_idxs;
    int n_x = static_cast<int>(delta_x / neighborhood_size) + 1;
    int n_y = static_cast<int>(delta_y / neighborhood_size) + 1;
    float step_x = delta_x / n_x;
    float step_y = delta_y / n_y;
    for (size_t pt_idx = 0; pt_idx < data_->size(); ++pt_idx) {
        PclPoint &point = data_->at(pt_idx);
        long pt_i = floor((point.x - bound_box[0]) / step_x);
        long pt_j = floor((point.y - bound_box[2]) / step_y);
        if (pt_i == n_x){
            pt_i -= 1;
        }
        if (pt_j == n_y) {
            pt_j -= 1;
        }
        long idx = pt_i + pt_j*n_x;
        grid_idxs.insert(idx);
    }
    for (set<long>::iterator it = grid_idxs.begin(); it != grid_idxs.end(); ++it) {
        long idx = *it;
        int pt_i = idx % n_x;
        int pt_j = idx / n_x;
        float pt_x = bound_box[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box[2] + (pt_j+0.5)*step_y;
        seed_samples.push_back(Vertex(pt_x, pt_y, 0.0));
    }
    
    int n_iter = 1;
    float search_radius = (step_x > step_y) ? step_x : step_y;
    search_radius /= sqrt(2.0);
    search_radius += 1.0;
    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter) {
        // For each seed, search its neighbors and do an average
        for (vector<Vertex>::iterator it = seed_samples.begin(); it != seed_samples.end(); ++it) {
            PclPoint search_point;
            search_point.setCoordinate((*it).x, (*it).y, 0.0f);
            // Search nearby and mark nearby
            vector<int> k_indices;
            vector<float> k_distances;
            tree_->radiusSearch(search_point, search_radius, k_indices, k_distances);
            
            if (k_indices.size() == 0) {
                printf("WARNING!!! Empty sampling cell! %.1f, %.1f\n", search_point.x, search_point.y);
                continue;
            }
            float sum_x = 0.0;
            float sum_y = 0.0;
            for (vector<int>::iterator neighbor_it = k_indices.begin(); neighbor_it != k_indices.end(); ++neighbor_it) {
                PclPoint &nb_pt = data_->at(*neighbor_it);
                sum_x += nb_pt.x;
                sum_y += nb_pt.y;
            }
            
            sum_x /= k_indices.size();
            sum_y /= k_indices.size();
            
            point.setCoordinate(sum_x, sum_y, 0.0f);
            point.t = 0;
            point.id_trajectory = 0;
            point.id_sample = 0;
            
            samples->push_back(point);
        }
    }
    
    sample_tree->setInputCloud(samples);
    printf("Sampling Done. Totally %zu samples.\n", samples->size());
}

void Trajectories::samplePointCloud(float neighborhood_size){
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    
    // Clear samples
    PclPoint point;
    samples_->clear();
    picked_sample_idxs_.clear();
    
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    
    // Compute samples
    // Initialize the samples using a unified grid
    vector<Vertex> seed_samples;
    set<int> grid_idxs;
    int n_x = static_cast<int>(delta_x / neighborhood_size) + 1;
    int n_y = static_cast<int>(delta_y / neighborhood_size) + 1;
    float step_x = delta_x / n_x;
    float step_y = delta_y / n_y;
    for (size_t pt_idx = 0; pt_idx < data_->size(); ++pt_idx) {
        PclPoint &point = data_->at(pt_idx);
        int pt_i = floor((point.x - bound_box[0]) / step_x);
        int pt_j = floor((point.y - bound_box[2]) / step_y);
        if (pt_i == n_x){
            pt_i -= 1;
        }
        if (pt_j == n_y) {
            pt_j -= 1;
        }
        int idx = pt_i + pt_j*n_x;
        grid_idxs.insert(idx);
    }
    for (set<int>::iterator it = grid_idxs.begin(); it != grid_idxs.end(); ++it) {
        int idx = *it;
        int pt_i = idx % n_x;
        int pt_j = idx / n_x;
        float pt_x = bound_box[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box[2] + (pt_j+0.5)*step_y;
        seed_samples.push_back(Vertex(pt_x, pt_y, 0.0));
    }
    
    int n_iter = 1;
    float search_radius = (step_x > step_y) ? step_x : step_y;
    search_radius /= sqrt(2.0);
    search_radius += 1.0;
    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter) {
        // For each seed, search its neighbors and do an average
        for (vector<Vertex>::iterator it = seed_samples.begin(); it != seed_samples.end(); ++it) {
            PclPoint search_point;
            search_point.setCoordinate((*it).x, (*it).y, 0.0f);
            // Search nearby and mark nearby
            vector<int> k_indices;
            vector<float> k_distances;
            tree_->radiusSearch(search_point, search_radius, k_indices, k_distances);
            
            if (k_indices.size() == 0) {
                printf("WARNING!!! Empty sampling cell!\n");
                continue;
            }
            float sum_x = 0.0;
            float sum_y = 0.0;
            double sum_lat = 0;
            double sum_lon = 0;
            for (vector<int>::iterator neighbor_it = k_indices.begin(); neighbor_it != k_indices.end(); ++neighbor_it) {
                PclPoint &nb_pt = data_->at(*neighbor_it);
                sum_x += nb_pt.x;
                sum_y += nb_pt.y;
                sum_lat += nb_pt.lat;
                sum_lon += nb_pt.lon;
            }
            
            sum_x /= k_indices.size();
            sum_y /= k_indices.size();
            sum_lat /= k_indices.size();
            sum_lon /= k_indices.size();
            
            point.setCoordinate(sum_x, sum_y, 0.0f);
            point.lat = floor(sum_lat);
            point.lon = floor(sum_lon);
            point.t = 0;
            point.id_trajectory = 0;
            point.id_sample = 0;
            
            samples_->push_back(point);
            sample_vertex_colors_.push_back(sample_color_);
        }
    }
    
    sample_tree_->setInputCloud(samples_);
    printf("Sampling Done. Totally %zu samples.\n", samples_->size());
}

void Trajectories::pickASampleNear(float x, float y, float max_distance){
    if (samples_->size() == 0)
        return;
    picked_sample_idxs_.clear();
    
    PclPoint center_point;
    center_point.setCoordinate(x, y, 0.0f);
    vector<int> k_indices;
    vector<float> k_sqr_distances;
    
    sample_tree_->nearestKSearch(center_point, 1, k_indices, k_sqr_distances);
    
    if (k_indices.empty())
        return;
    
    picked_sample_idxs_.insert(k_indices[0]);
}

void Trajectories::pickAnotherSampleNear(float x, float y, float max_distance){
    if (samples_->size() == 0)
        return;
    
    PclPoint center_point;
    center_point.setCoordinate(x, y, 0.0f);
    vector<int> k_indices;
    vector<float> k_sqr_distances;
    
    sample_tree_->nearestKSearch(center_point, 1, k_indices, k_sqr_distances);
    
    if (k_indices.empty())
        return;
}

void Trajectories::clearPickedSamples(void){
    picked_sample_idxs_.clear();
}

void Trajectories::singleDirectionSamplePointCloud(float neighborhood_size){
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    
    // Clear samples
    PclPoint point;
    samples_->clear();
    
    normalized_sample_locs_.clear();
    normalized_sample_headings_.clear();
    sample_heading_colors_.clear();
    sample_vertex_colors_.clear();
    
    // Compute samples
    // Initialize the samples using a unified grid
    vector<Vertex> seed_samples;
    set<int> grid_idxs;
    int n_x = static_cast<int>(delta_x / neighborhood_size) + 1;
    int n_y = static_cast<int>(delta_y / neighborhood_size) + 1;
    float step_x = delta_x / n_x;
    float step_y = delta_y / n_y;
    for (size_t pt_idx = 0; pt_idx < data_->size(); ++pt_idx) {
        PclPoint &point = data_->at(pt_idx);
        int pt_i = floor((point.x - bound_box[0]) / step_x);
        int pt_j = floor((point.y - bound_box[2]) / step_y);
        if (pt_i == n_x){
            pt_i -= 1;
        }
        if (pt_j == n_y) {
            pt_j -= 1;
        }
        int idx = pt_i + pt_j*n_x;
        grid_idxs.insert(idx);
    }
    for (set<int>::iterator it = grid_idxs.begin(); it != grid_idxs.end(); ++it) {
        int idx = *it;
        int pt_i = idx % n_x;
        int pt_j = idx / n_x;
        float pt_x = bound_box[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box[2] + (pt_j+0.5)*step_y;
        seed_samples.push_back(Vertex(pt_x, pt_y, 0.0));
    }
    
    float search_radius = (step_x > step_y) ? step_x : step_y;
    search_radius /= sqrt(2.0);
    search_radius  = ceil(search_radius);
    
    // For each seed, search its neighbors
    int N_BINS = 36;
    float DELTA_BIN = 360.0f / N_BINS;
    float ANGLE_THETA = DELTA_BIN;
    for (vector<Vertex>::iterator it = seed_samples.begin(); it != seed_samples.end(); ++it) {
        PclPoint search_point;
        search_point.setCoordinate((*it).x, (*it).y, 0.0f);
        // Search nearby and mark nearby
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->radiusSearch(search_point, search_radius, k_indices, k_distances);
        
        if (k_indices.size() == 0) {
            printf("WARNING!!! Empty sampling cell!\n");
            continue;
        }
        
        // Collect nearby point headings
        vector<float> heading_hist(N_BINS, 0.0f);
        map<int, int> traj_heading_vote;
        for (size_t i = 0; i < k_indices.size(); ++i) {
            // Each trajectory will only vote one direction for one sample
            PclPoint& pt = data_->at(k_indices[i]);
            if(traj_heading_vote.find(pt.id_trajectory) == traj_heading_vote.end()){
                traj_heading_vote[pt.id_trajectory] = pt.head;
            }
        }
        
        for(map<int, int>::iterator mit = traj_heading_vote.begin(); mit != traj_heading_vote.end(); ++mit){
            float heading = mit->second;
            //float speed = data_->at(k_indices[i]).speed * 1.0f / 100.0f;
            
            int angle_bin_idx = floor(heading / DELTA_BIN);
            for (int s = -1; s <= 1; ++s){
                // Assume Gaussian angle distribution
                int nearby_bin_idx = angle_bin_idx + s;
                if (nearby_bin_idx < 0) {
                    nearby_bin_idx += N_BINS;
                }
                if (nearby_bin_idx >= N_BINS) {
                    nearby_bin_idx %= N_BINS;
                }
                float nearby_bin_center = (nearby_bin_idx + 0.5)*DELTA_BIN;
                float delta_angle = abs(deltaHeading1MinusHeading2(nearby_bin_center, heading));
                
                heading_hist[nearby_bin_idx] += exp(-1 * delta_angle * delta_angle / 2.0 / ANGLE_THETA / ANGLE_THETA);
            }
        }
        
        // Find the maximum
        int max_peak_idx = -1;
        float max_value = 0.1f;
        for(int i = 0; i < N_BINS; ++i){
            if(max_value < heading_hist[i]){
                max_value = heading_hist[i];
                max_peak_idx = i;
            }
        }
        
        if(max_peak_idx == -1){
            continue;
        }
        
        float angle1 = (max_peak_idx + 0.5) * DELTA_BIN;
        
        // Compute sample center and scale (essentially road width)
        float ANGLE_THRESHOLD = 15;
        float cum_x = 0.0f;
        float cum_y = 0.0f;
        int n_consist_point = 0;
        for (size_t i = 0; i < k_indices.size(); ++i) {
            PclPoint &nb_pt = data_->at(k_indices[i]);
            float delta_angle = abs(deltaHeading1MinusHeading2(nb_pt.head, angle1));
            if (delta_angle < ANGLE_THRESHOLD) {
                cum_x += nb_pt.x;
                cum_y += nb_pt.y;
                n_consist_point++;
            }
        }
        
        // New sample location
        if (n_consist_point > 0) {
            float new_x = cum_x / n_consist_point;
            float new_y = cum_y / n_consist_point;
            
            point.setCoordinate(new_x, new_y, 0.0f);
            point.t = 0;
            point.id_trajectory = 0;
            point.id_sample = 0;
            point.head = floor(angle1);
            
            samples_->push_back(point);
        }
    }
    
    if(samples_->size() > 0){
        sample_tree_->setInputCloud(samples_);
    }
    else{
        return;
    }
    
    float ARROW_LENGTH = 10.0f;
    for(size_t i = 0; i < samples_->size(); ++i){
        PclPoint& point = samples_->at(i);
        float head_in_radius = point.head / 180.0 * PI;
        normalized_sample_locs_.push_back(SceneConst::getInstance().normalize(point.x, point.y, Z_SAMPLES));
        normalized_sample_headings_.push_back(SceneConst::getInstance().normalize(point.x, point.y, Z_SAMPLES));
        
        normalized_sample_headings_.push_back(SceneConst::getInstance().normalize(point.x+ARROW_LENGTH*cos(head_in_radius), point.y+ARROW_LENGTH*sin(head_in_radius), Z_SAMPLES));
        
        sample_vertex_colors_.push_back(sample_color_);
        sample_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::ORANGE));
        sample_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::ORANGE));
    }
    
    printf("Sampling Done. Totally %zu samples.\n", samples_->size());
}

///////////////////////////////////////////////////
//                  Selection
///////////////////////////////////////////////////
void Trajectories::selectNearLocation(float x, float y, float max_distance){
    selected_trajectories_.clear();
    PclPoint center_point;
    center_point.setCoordinate(x, y, 0.0f);
    vector<int> k_indices;
    vector<float> k_sqr_distances;
    
    tree_->nearestKSearch(center_point, 1, k_indices, k_sqr_distances);
    
    if (k_indices.empty())
        return;
    
    PclPoint &selected_point = data_->at(k_indices[0]);
    
    if (sqrt(k_sqr_distances[0]) < search_distance){
        int traj_id = selected_point.id_trajectory;
        selected_trajectories_.push_back(selected_point.id_trajectory);
        printf("Trajectory %d selected.\n", traj_id);
        //for (size_t pt_idx = 0; pt_idx < trajectories_[traj_id].size(); ++pt_idx) {
        //int pt_idx_in_data = trajectories_[traj_id][pt_idx];
        //printf("t = %.2f, x=%.2f, y=%.2f, head=%d, speed=%d\n", data_->at(pt_idx_in_data).t, data_->at(pt_idx_in_data).x, data_->at(pt_idx_in_data).y, data_->at(pt_idx_in_data).head, data_->at(pt_idx_in_data).speed);
        //}
    }
}

void Trajectories::drawSelectedTrajectories(vector<int> &traj_indices){
    if (traj_indices.empty()){
        return;
    }
    
    vector<Vertex>          selected_vertices;
    vector<Color>           selected_vertex_colors;
    vector<vector<int> >    selected_trajectories;
    vector<Vertex>              direction_vertices;
    vector<Color>               direction_colors;
    vector<vector<unsigned> >   direction_idxs;
    
    float arrow_length = arrow_size / scale_factor_;
    
    selected_trajectories.resize(traj_indices.size());
    
    for(size_t idx = 0; idx < traj_indices.size(); ++idx){
        int selected_traj_idx = traj_indices[idx];
        vector<int> &orig_trajectory = trajectories_[selected_traj_idx];
        vector<int> &new_trajectory_indices = selected_trajectories[idx];
        for(size_t pt_idx = 0; pt_idx < orig_trajectory.size(); ++pt_idx){
            new_trajectory_indices.push_back(selected_vertices.size());
            
            Vertex &origVertex = normalized_vertices_[orig_trajectory[pt_idx]];
            Color &origColor = vertex_colors_individual_[orig_trajectory[pt_idx]];
            selected_vertices.push_back(Vertex(origVertex.x, origVertex.y, Z_SELECTED_TRAJ));
            selected_vertex_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
            
            // Prepare for drawing arrow for trajectory
            if (pt_idx != 0) {
                int last_pt_idx = selected_vertices.size() - 1;
                float dx = selected_vertices[last_pt_idx].x - selected_vertices[last_pt_idx-1].x;
                float dy = selected_vertices[last_pt_idx].y - selected_vertices[last_pt_idx-1].y;
                float length = sqrt(dx*dx + dy*dy);
                if (length*scale_factor_ < 3*arrow_size) {
                    continue;
                }
                dx = dx / length * arrow_length;
                dy = dy / length * arrow_length;
                QVector3D vec(dx, dy, 0.0);
                QMatrix4x4 m1;
                m1.setToIdentity();
                m1.rotate(arrow_angle, 0.0f, 0.0f, 1.0f);
                QMatrix4x4 m2;
                m2.setToIdentity();
                m2.rotate(-arrow_angle, 0.0f, 0.0f, 1.0f);
                QVector3D vec1 = m1.map(vec);
                QVector3D vec2 = m2.map(vec);
                float center_x = 0.5*selected_vertices[last_pt_idx].x + 0.5*selected_vertices[last_pt_idx-1].x;
                float center_y = 0.5*selected_vertices[last_pt_idx].y + 0.5*selected_vertices[last_pt_idx-1].y;
                vector<unsigned> direction_idx(3, 0);
                direction_idx[0] = direction_vertices.size();
                direction_vertices.push_back(Vertex(center_x - vec1.x(), center_y - vec1.y(), Z_SELECTED_TRAJ));
                direction_idx[1] = direction_vertices.size();
                direction_vertices.push_back(Vertex(center_x, center_y, Z_SELECTED_TRAJ));
                direction_idx[2] = direction_vertices.size();
                direction_idxs.push_back(direction_idx);
                direction_vertices.push_back(Vertex(center_x - vec2.x(), center_y - vec2.y(), Z_SELECTED_TRAJ));
                direction_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
                direction_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
                direction_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
            }
        }
    }
    
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&selected_vertices[0], 3*selected_vertices.size()*sizeof(float));
    shader_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&selected_vertex_colors[0], 4*selected_vertex_colors.size()*sizeof(float));
    shader_program_->setupColorAttributes();
    
    for (int i=0; i < selected_trajectories.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(selected_trajectories[i][0]), selected_trajectories[i].size()*sizeof(size_t));
        glLineWidth(line_width_*1.5);
        glDrawElements(GL_LINE_STRIP, selected_trajectories[i].size(), GL_UNSIGNED_INT, 0);
        glPointSize(point_size_*1.5);
        glDrawElements(GL_POINTS, selected_trajectories[i].size(), GL_UNSIGNED_INT, 0);
    }
    
    // Draw arrows
    QOpenGLBuffer vertex_buffer;
    vertex_buffer.create();
    vertex_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertex_buffer.bind();
    vertex_buffer.allocate(&direction_vertices[0], 3*direction_vertices.size()*sizeof(float));
    shader_program_->setupPositionAttributes();
    
    QOpenGLBuffer color_buffer;
    color_buffer.create();
    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    color_buffer.bind();
    color_buffer.allocate(&direction_colors[0], 4*direction_colors.size()*sizeof(float));
    shader_program_->setupColorAttributes();
    
    for (size_t i=0; i < direction_idxs.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(direction_idxs[i][0]), direction_idxs[i].size()*sizeof(unsigned));
        glLineWidth(line_width_*2.5);
        glDrawElements(GL_TRIANGLES, direction_idxs[i].size(), GL_UNSIGNED_INT, 0);
    }
    
    // Draw speed and heading
    vector<int> speed_heading_idxs;
    for (size_t i = 0; i < traj_indices.size(); ++i) {
        vector<int> &traj = trajectories_[traj_indices[i]];
        for (size_t j = 0; j < traj.size(); ++j) {
            int pt_idx = traj[j];
            speed_heading_idxs.push_back(2*pt_idx);
            speed_heading_idxs.push_back(2*pt_idx+1);
        }
    }
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&vertex_speed_[0], 3*vertex_speed_.size()*sizeof(float));
    shader_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&vertex_speed_colors_[0], 4*vertex_speed_colors_.size()*sizeof(float));
    shader_program_->setupColorAttributes();
    QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
    element_buffer.create();
    element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    element_buffer.bind();
    element_buffer.allocate(&(speed_heading_idxs[0]), speed_heading_idxs.size()*sizeof(int));
    glLineWidth(line_width_);
    glDrawElements(GL_LINES, speed_heading_idxs.size(), GL_UNSIGNED_INT, 0);
}

void Trajectories::setSelectedTrajectories(vector<int> &selectedIdxs){
    selected_trajectories_.clear();
    selected_trajectories_.resize(selectedIdxs.size());
    for (size_t i = 0; i < selectedIdxs.size(); ++i) {
        selected_trajectories_[i] = selectedIdxs[i];
    }
}

////////////////////////////////////////
//          Clear Data
////////////////////////////////////////
void Trajectories::clearData(void){
    point_size_ = 5.0f;
    line_width_ = 3.0f;
    render_mode_ = false;
    selection_mode_ = false;
    
    data_->clear();
    min_speed_ = POSITIVE_INFINITY;
    max_speed_ = 0.0f;
    avg_speed_ = 0.0f;

    trajectories_.clear();
    selected_trajectories_.clear();
    
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    vertex_speed_.clear();
    vertex_speed_colors_.clear();
    vertex_speed_indices_.clear();
    selected_trajectories_.clear();
    
    // Clear samples
    samples_->clear();
    picked_sample_idxs_.clear();
    normalized_sample_locs_.clear();
    normalized_sample_headings_.clear();
    sample_vertex_colors_.clear();
    sample_heading_colors_.clear();
    picked_sample_idxs_.clear();
}

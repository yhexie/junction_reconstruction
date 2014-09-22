#include <ctime>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <boost/filesystem.hpp>
#include "color_map.h"
#include "trajectories.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "graph.h"
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

Trajectories::Trajectories(QObject *parent) : Renderable(parent), data_(new PclPointCloud), tree_(new pcl::search::FlannSearch<PclPoint>(false)), distance_graph_vertices_(new PclPointCloud), distance_graph_search_tree_(new pcl::search::FlannSearch<PclPoint>(false)), sample_point_size_(10.0f), sample_color_(Color(0.0f, 0.3f, 0.6f, 1.0f)),samples_(new PclPointCloud), sample_tree_(new pcl::search::FlannSearch<PclPoint>(false)), cluster_centers_(new PclPointCloud), cluster_center_search_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    point_size_ = 5.0f;
    line_width_ = 3.0f;
    render_mode_ = false;
    selection_mode_ = false;
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    show_interpolated_trajectories_ = true;
    
    graph_ = NULL;
    
    // Distance graph
    distance_graph_ = NULL;
    show_distance_graph_ = true;
    
    // Samples
    show_samples_ = true;
    samples_->clear();
    picked_sample_idxs_.clear();
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    cluster_descriptors_ = NULL;
    normalized_cluster_center_locs_.clear();
    cluster_center_colors_.clear();
    
    // Segments
    show_segments_ = true;
    segments_.clear();
    graph_interpolated_segments_.clear();
    segments_to_draw_.clear();
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    search_range_ = 10.0f;
}

Trajectories::~Trajectories(){
    if (distance_graph_ != NULL){
        delete distance_graph_;
    }
    if (cluster_descriptors_ != NULL){
        delete cluster_descriptors_;
    }
}

void Trajectories::draw(){
    if(!render_mode_){
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&vertex_colors_const_[0], 4*vertex_colors_const_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        glPointSize(point_size_);
        glDrawArrays(GL_POINTS, 0, normalized_vertices_.size());
    }
    else{
        if(trajectories_.size() == 0)
            return;
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&vertex_colors_individual_[0], 4*vertex_colors_individual_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
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
    
    // Draw Distance Graph
    if (show_distance_graph_ && distance_graph_ != NULL) {
        // Draw Edges
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_distance_graph_vertices_[0], 3*normalized_distance_graph_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&distance_graph_vertex_colors_[0], 4*distance_graph_vertex_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        vector<int> &distance_graph_edge_idxs = distance_graph_->edge_idxs();
        element_buffer.allocate(&(distance_graph_edge_idxs[0]), distance_graph_edge_idxs.size()*sizeof(int));
        glLineWidth(line_width_);
        glDrawElements(GL_LINES, distance_graph_edge_idxs.size(), GL_UNSIGNED_INT, 0);
        
        // Draw Vertices
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_distance_graph_vertices_[0], 3*normalized_distance_graph_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&distance_graph_vertex_colors_[0], 4*distance_graph_vertex_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        glPointSize(point_size_);
        glDrawArrays(GL_POINTS, 0, normalized_distance_graph_vertices_.size());
    }
    
    // Draw samples
    if (show_samples_){
        if (samples_->size() == 0) {
            return;
        }
      
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_sample_locs_[0], 3*normalized_sample_locs_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&sample_vertex_colors_[0], 4*sample_vertex_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        glPointSize(sample_point_size_);
        glDrawArrays(GL_POINTS, 0, normalized_sample_locs_.size());
        
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
            shadder_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&picked_sample_colors[0], 4*picked_sample_colors.size()*sizeof(float));
            shadder_program_->setupColorAttributes();
            
            glPointSize(1.5*sample_point_size_);
            glDrawArrays(GL_POINTS, 0, picked_sample_locs.size());
        }
    }
    
    // Draw segments_to_draw_
    if (show_segments_){
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&segment_vertices_[0], 3*segment_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&segment_colors_[0], 4*segment_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        for (int i=0; i < segment_idxs_.size(); ++i) {
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(segment_idxs_[i][0]), segment_idxs_[i].size()*sizeof(size_t));
            glLineWidth(line_width_);
            glDrawElements(GL_LINE_STRIP, segment_idxs_[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_);
            glDrawElements(GL_POINTS, segment_idxs_[i].size(), GL_UNSIGNED_INT, 0);
        }
    }
}

void Trajectories::prepareForVisualization(QVector4D bound_box){
    display_bound_box_ = bound_box;
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    normalized_cluster_center_locs_.clear();
    cluster_center_colors_.clear();
    vertex_speed_.clear();
    vertex_speed_indices_.clear();
    vertex_speed_colors_.clear();
    
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (delta_x < 0 || delta_y < 0) {
        fprintf(stderr, "Trajectory bounding box error! Min greater than Max!\n");
    }
    
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    scale_factor_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
   
    for (size_t i=0; i<data_->size(); ++i) {
        const PclPoint &point = data_->at(i);
        float n_x = (point.x - center_x) / scale_factor_;
        float n_y = (point.y - center_y) / scale_factor_;
        normalized_vertices_.push_back(Vertex(n_x, n_y, Z_TRAJECTORIES));
        
        // Speed and heading
        float speed = static_cast<float>(data_->at(i).speed) / 100.0f;
        float heading = static_cast<float>(data_->at(i).head) *PI / 180.0f;
        float speed_line_length = speed / scale_factor_;
        float speed_delta_x = speed_line_length * sin(heading);
        float speed_delta_y = speed_line_length * cos(heading);
        vertex_speed_indices_.push_back(vertex_speed_.size());
        vertex_speed_.push_back(Vertex(n_x, n_y, Z_TRAJECTORY_SPEED));
        vertex_speed_indices_.push_back(vertex_speed_.size());
        vertex_speed_.push_back(Vertex(n_x+speed_delta_x, n_y+speed_delta_y, Z_TRAJECTORY_SPEED));
    }
    
    Color light_gray = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_GRAY);
    vertex_colors_const_.resize(normalized_vertices_.size(), light_gray);
    
    vertex_colors_individual_.resize(data_->size(), Color(0,0,0,1));
    vertex_speed_colors_.resize(data_->size()*2, Color(0,0,0,1));
    for (int i=0; i < trajectories_.size(); ++i) {
        Color traj_color = ColorMap::getInstance().getDiscreteColor(i);
        for (int j = 0; j < trajectories_[i].size(); ++j) {
            int vertex_idx = trajectories_[i][j];
            vertex_colors_individual_[vertex_idx].r = traj_color.r;
            vertex_colors_individual_[vertex_idx].g = traj_color.g;
            vertex_colors_individual_[vertex_idx].b = traj_color.b;
            vertex_speed_colors_[2*vertex_idx].r = 1.0 - traj_color.r;
            vertex_speed_colors_[2*vertex_idx].g = 1.0 - traj_color.g;
            vertex_speed_colors_[2*vertex_idx].b = 1.0 - traj_color.b;
            vertex_speed_colors_[2*vertex_idx+1].r = 1.0 - traj_color.r;
            vertex_speed_colors_[2*vertex_idx+1].g = 1.0 - traj_color.g;
            vertex_speed_colors_[2*vertex_idx+1].b = 1.0 - traj_color.b;
        }
    }
    
    // Prepare for distance graph
    if (!show_distance_graph_) {
        return;
    }
    else{
        if (distance_graph_ != NULL) {
            normalized_distance_graph_vertices_.clear();
            distance_graph_vertex_colors_.clear();
            normalized_distance_graph_vertices_.resize(distance_graph_vertices_->size());
            Color dark_gray = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
            distance_graph_vertex_colors_.resize(distance_graph_vertices_->size(), dark_gray);
            
            for (size_t i = 0; i < distance_graph_vertices_->size(); ++i){
                PclPoint &loc = distance_graph_vertices_->at(i);
                normalized_distance_graph_vertices_[i].x = (loc.x - center_x) / scale_factor_;
                normalized_distance_graph_vertices_[i].y = (loc.y - center_y) / scale_factor_;
                normalized_distance_graph_vertices_[i].z = Z_DISTANCE_GRAPH;
            }
        }
    }
    
    // Prepare for sample visualization
    if (samples_->size() == 0)
        return;
    
    normalized_sample_locs_.resize(samples_->size());
    sample_vertex_colors_.resize(samples_->size());
    
    for (size_t i = 0; i < samples_->size(); ++i){
        PclPoint &loc = samples_->at(i);
        normalized_sample_locs_[i].x = (loc.x - center_x) / scale_factor_;
        normalized_sample_locs_[i].y = (loc.y - center_y) / scale_factor_;
        normalized_sample_locs_[i].z = Z_SAMPLES;
        sample_vertex_colors_[i].r = sample_color_.r;
        sample_vertex_colors_[i].g = sample_color_.g;
        sample_vertex_colors_[i].b = sample_color_.b;
        sample_vertex_colors_[i].alpha = sample_color_.alpha;
    }
    
    // Prepare for graph visualization
    if (cluster_centers_->size() == 0)
        return;
    
    normalized_cluster_center_locs_.resize(cluster_centers_->size());
    cluster_center_colors_.resize(cluster_centers_->size());
    
    for (size_t i = 0; i < cluster_centers_->size(); ++i){
        PclPoint &loc = cluster_centers_->at(i);
        normalized_cluster_center_locs_[i].x = (loc.x - center_x) / scale_factor_;
        normalized_cluster_center_locs_[i].y = (loc.y - center_y) / scale_factor_;
        normalized_cluster_center_locs_[i].z = Z_DISTANCE_GRAPH;
        cluster_center_colors_[i].r = sample_color_.r;
        cluster_center_colors_[i].g = sample_color_.g;
        cluster_center_colors_[i].b = sample_color_.b;
        cluster_center_colors_[i].alpha = sample_color_.alpha;
    }
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
            
            if (x < bound_box_[0]) {
                bound_box_[0] = x;
            }
            if (x > bound_box_[1]){
                bound_box_[1] = x;
            }
            if (y < bound_box_[2]) {
                bound_box_[2] = y;
            }
            if (y > bound_box_[3]){
                bound_box_[3] = y;
            }
            
			point.setCoordinate(x, y, 0.0);
			point.t = static_cast<float>(t);
			point.id_trajectory = id_trajectory;
			point.id_sample = id_sample;
            
			trajectory[id_sample] = data_->size();
			data_->push_back(point);
		}
	}
    
	return true;
}

bool Trajectories::loadPBF(const string &filename){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // Prepare the coordinate projector
    Projector converter;
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
    
    google::protobuf::io::ZeroCopyInputStream *raw_input = new google::protobuf::io::FileInputStream(fid);
    google::protobuf::io::CodedInputStream *coded_input = new google::protobuf::io::CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(1e9, 9e8);
    
    PclPoint point;
	point.setNormal(0, 0, 1);
    
    uint32_t num_trajectory;
    if (!coded_input->ReadLittleEndian32(&num_trajectory)) {
        return false; // possibly end of file
    }
    
    trajectories_.resize(num_trajectory, vector<int>());
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
        
        // Add the points to point cloud and trajectory
        vector<int>& trajectory=trajectories_[id_traj];
		trajectory.resize(new_traj.point_size());
        
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
        
        if (has_correct_projection) {
            for (int pt_idx=0; pt_idx < new_traj.point_size(); ++pt_idx) {
                if (new_traj.point(pt_idx).x() < bound_box_[0]) {
                    bound_box_[0] = new_traj.point(pt_idx).x();
                }
                if (new_traj.point(pt_idx).x() > bound_box_[1]) {
                    bound_box_[1] = new_traj.point(pt_idx).x();
                }
                if (new_traj.point(pt_idx).y() < bound_box_[2]) {
                    bound_box_[2] = new_traj.point(pt_idx).y();
                }
                if (new_traj.point(pt_idx).y() > bound_box_[3]) {
                    bound_box_[3] = new_traj.point(pt_idx).y();
                }
                
                point.setCoordinate(new_traj.point(pt_idx).x(), new_traj.point(pt_idx).y(), 0.0);
    			point.t = new_traj.point(pt_idx).timestamp();
    			point.id_trajectory = id_traj;
    			point.id_sample = pt_idx;
                point.car_id = new_traj.point(pt_idx).car_id();
                point.lon = new_traj.point(pt_idx).lon();
                point.lat = new_traj.point(pt_idx).lat();
                point.speed = new_traj.point(pt_idx).speed();
                point.head = new_traj.point(pt_idx).head();
                
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
                
                if (x < bound_box_[0]) {
                    bound_box_[0] = x;
                }
                if (x > bound_box_[1]){
                    bound_box_[1] = x;
                }
                if (y < bound_box_[2]) {
                    bound_box_[2] = y;
                }
                if (y > bound_box_[3]){
                    bound_box_[3] = y;
                }
                
    			point.setCoordinate(x, y, 0.0);
    			point.t = t - UTC_OFFSET;
    			point.id_trajectory = id_traj;
    			point.id_sample = pt_idx;
                point.car_id = new_traj.point(pt_idx).car_id();
                point.lon = new_traj.point(pt_idx).lon();
                point.lat = new_traj.point(pt_idx).lat();
                point.speed = new_traj.point(pt_idx).speed();
                point.head = new_traj.point(pt_idx).head();
                
    			trajectory[pt_idx] = data_->size();
    			data_->push_back(point);
            }
        }
    }
    
    close(fid);
    delete raw_input;
    delete coded_input;
    printf("Loading completed. %d trajectories loaded. Totally %zu points.\n", num_trajectory, data_->size());
    return true;
}

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
    savePBF(filename);
    return true;
}

bool Trajectories::isEmpty(){
    if (data_->size() > 0){
        return false;
    }
    else{
        return true;
    }
}

bool Trajectories::savePBF(const string &filename){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // Read the trajectory collection from file.
    int fid = open(filename.c_str(), O_WRONLY | O_CREAT);
    
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
            new_pt->set_head(data_->at(pt_idx_in_data).head);
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

bool Trajectories::extractFromFiles(const QStringList &filenames, QVector4D bound_box, int nmin_inside_pts){
    data_->clear();
    trajectories_.clear();
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
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
        new_trajectories->extractTrajectoriesFromRegion(bound_box, this, nmin_inside_pts);
        delete new_trajectories;
    }
    
    tree_->setInputCloud(data_);
    
    printf("Totally %lu trajectories, %lu points.\n", trajectories_.size(), data_->size());
    
    return true;
}

bool Trajectories::extractTrajectoriesFromRegion(QVector4D bound_box, Trajectories *container, int min_inside_pts){
    // Find all potential trajectories indices inside using kdtree search
    PclPoint center_point;
    float x_length = bound_box[1] - bound_box[0];
    float y_length = bound_box[3] - bound_box[2];
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    center_point.setCoordinate(center_x, center_y, 0.0f);
    double radius = 0.5*sqrt(x_length*x_length + y_length*y_length);
    
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
                if (!recording) {
                    recording = true;
                    extracted_trajectory.clear();
                    if (pt_idx > 0) {
                        // Insert previous point
                        PclPoint &prev_point = data_->at(trajectory[pt_idx-1]);
                        extracted_trajectory.push_back(prev_point);
                    }
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
                    extracted_trajectory.push_back(point);
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
        if (point.x < bound_box_[0]) {
            bound_box_[0] = point.x;
        }
        if (point.x > bound_box_[1]){
            bound_box_[1] = point.x;
        }
        if (point.y < bound_box_[2]) {
            bound_box_[2] = point.y;
        }
        if (point.y > bound_box_[3]){
            bound_box_[3] = point.y;
        }
        
        point.id_trajectory = trajectories_.size();
        point.id_sample = i;
        trajectory[i] = data_->size();
        data_->push_back(point);
    }
    trajectories_.push_back(trajectory);
    return true;
}

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
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&selected_vertex_colors[0], 4*selected_vertex_colors.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
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
    shadder_program_->setupPositionAttributes();
    
    QOpenGLBuffer color_buffer;
    color_buffer.create();
    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    color_buffer.bind();
    color_buffer.allocate(&direction_colors[0], 4*direction_colors.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
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
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&vertex_speed_colors_[0], 4*vertex_speed_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
    element_buffer.create();
    element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    element_buffer.bind();
    element_buffer.allocate(&(speed_heading_idxs[0]), speed_heading_idxs.size()*sizeof(int));
    glLineWidth(line_width_);
    glDrawElements(GL_LINES, speed_heading_idxs.size(), GL_UNSIGNED_INT, 0);
    
    // Draw Interpolated trajectories
    if (show_interpolated_trajectories_) {
        if (graph_ == NULL) {
            return;
        }
        
        // Compute interpolation using graph_.
        vector<vector<int> >    interpolated_trajectories(traj_indices.size());
        for (size_t i = 0; i < traj_indices.size(); ++i) {
            int traj_id = traj_indices[i];
            vector<int> &orig_trajectory = trajectories_[traj_id];
            vector<int> &result_traj = interpolated_trajectories[i];
            graph_->shortestPathInterpolation(orig_trajectory, result_traj);
        }
        vector<Vertex>          interpolated_trajectory_vertices;
        vector<Color>           interpolated_trajectory_vertex_colors;
        vector<vector<int>>             interpolated_trajectory_indices;
        
        for (size_t i = 0; i < interpolated_trajectories.size(); ++i) {
            vector<int> &traj = interpolated_trajectories[i];
            vector<int> new_trajectory_indices;
            //Color &origColor = vertex_colors_individual_[orig_trajectory[0]];
            Color origColor = Color(1.0, 0.0, 0, 1);
            for(size_t pt_idx = 0; pt_idx < traj.size(); ++pt_idx){
                float alpha = 1.0 - 0.5 * static_cast<float>(pt_idx) / traj.size();
                new_trajectory_indices.push_back(interpolated_trajectory_vertices.size());
                Vertex &sampleVertex = normalized_cluster_center_locs_[traj[pt_idx]];
                interpolated_trajectory_vertices.push_back(Vertex(sampleVertex.x, sampleVertex.y, 0));
                interpolated_trajectory_vertex_colors.push_back(Color(alpha*origColor.r, alpha*origColor.g, alpha*origColor.b, 1.0));
            }
            interpolated_trajectory_indices.push_back(new_trajectory_indices);
        }
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&interpolated_trajectory_vertices[0], 3*interpolated_trajectory_vertices.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&interpolated_trajectory_vertex_colors[0], 4*interpolated_trajectory_vertex_colors.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        for (int i=0; i < interpolated_trajectory_indices.size(); ++i) {
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(interpolated_trajectory_indices[i][0]), interpolated_trajectory_indices[i].size()*sizeof(size_t));
            glLineWidth(line_width_*2);
            glDrawElements(GL_LINE_STRIP, interpolated_trajectory_indices[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_*2);
            glDrawElements(GL_POINTS, interpolated_trajectory_indices[i].size(), GL_UNSIGNED_INT, 0);
        }
    }
}

void Trajectories::setSelectedTrajectories(vector<int> &selectedIdxs){
    selected_trajectories_.clear();
    selected_trajectories_.resize(selectedIdxs.size());
    for (size_t i = 0; i < selectedIdxs.size(); ++i) {
        selected_trajectories_[i] = selectedIdxs[i];
    }
}

void Trajectories::clearData(void){
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    point_size_ = 5.0f;
    line_width_ = 3.0f;
    render_mode_ = false;
    selection_mode_ = false;
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    
    // Clear samples
    samples_->clear();
    picked_sample_idxs_.clear();
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    
    // Clear segments
    segments_.clear();
    segments_to_draw_.clear();
    search_range_ = 0.0f;
    segment_idxs_.clear();
    segment_vertices_.clear();
    segment_colors_.clear();
}

void Trajectories::computeWeightAtScale(int neighborhood_size){
    if (samples_->size() == 0){
        return;
    }
    
    sample_weight_.clear();
    sample_weight_.resize(samples_->size(), 0.0);
    
    clock_t begin = clock();
    
    int start = 10;
    int step = 10;
    int stop = 100;
    
    double max = 0.0;
    
    double sigma_max = 0.2;
    
    for(size_t i = 0; i < samples_->size(); ++i){
        vector<int> k_indices;
        vector<float> k_distances;
        sample_tree_->nearestKSearch(samples_->at(i), stop, k_indices, k_distances);
        
        MatrixXf m(stop, 2);
        for(size_t j = 0; j < stop; ++j){
            m(j, 0) = samples_->at(k_indices[j]).x;
            m(j, 1) = samples_->at(k_indices[j]).y;
        }
        MatrixXf centered = m.rowwise() - m.colwise().mean();
       
        double prev_weight = 0.0f;
        double prev_slop = 1.0f;
        int count = 0;
        for (int neighborhood_size = start; neighborhood_size <= stop; neighborhood_size += step) {
            MatrixXf block = centered.block(0, 0, neighborhood_size, 2);
            MatrixXf cov = (block.adjoint() * block) / double(m.rows());
            
            SelfAdjointEigenSolver<MatrixXf> es(cov);
            double sum_values = es.eigenvalues().sum();
            double tmp_weight;
            if (sum_values < 1e-6) {
                tmp_weight = 0.0f;
            }
            else{
                tmp_weight = es.eigenvalues()(0) / sum_values;
            }
            
            double cur_slop = (tmp_weight - prev_weight) / step;
            if (cur_slop > 4*prev_slop) {
                //printf("i=%lu, neighborhood size: %d\n", i, neighborhood_size);
                break;
            }
            
            if (tmp_weight > sigma_max) {
                sample_weight_[i] += 1.0f;
            }
            
            count += 1;
            prev_weight = tmp_weight;
            prev_slop = cur_slop;
        }
        
        if (sample_weight_[i] > max) {
                max = sample_weight_[i];
        }
    }
    
    
    double w_max = 0.6;
    double w_min = 0.5;
    for (size_t i = 0; i < samples_->size(); ++i) {
        double w = sample_weight_[i] / 9;
        Color pt_color;
        if (w > w_max) {
            pt_color = ColorMap::getInstance().getContinusColor(0.9);
        }
        else if (w < w_min){
            pt_color = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_GRAY);
        }
        else{
            pt_color = ColorMap::getInstance().getContinusColor(0.5);
        }
        
        sample_vertex_colors_[i].r = pt_color.r;
        sample_vertex_colors_[i].g = pt_color.g;
        sample_vertex_colors_[i].b = pt_color.b;
    }
    
    clock_t end = clock();
    
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    printf("time elapsed = %.1f sec.", elapsed_time);
}

void Trajectories::computeSegments(float extension){
   /*  Compute segment for each GPS point by extension to both direction.
        Args:
            - extension: in meters. For example, extension = 50 will result in extending the point in both direction by at least 50 meters, and the resulting segment will be around 100 meters.
    */
    
    // For each GPS point, compute its segment
   
    clock_t begin = clock();
    segments_.resize(data_->size());
    for (vector<vector<int>>::iterator traj_itr = trajectories_.begin(); traj_itr != trajectories().end(); ++traj_itr) {
        vector<int> &traj = *traj_itr;
        int prev_idx = 0;
        int nxt_idx = 0;
        int prev_pt_idx_in_data = -1;
        float cum_prev_to_current = 0.0f;
        float cum_current_to_nxt = 0.0f;
        for (size_t pt_idx = 0; pt_idx < traj.size(); ++pt_idx) {
            int pt_idx_in_data = traj[pt_idx];
            if (pt_idx > 0) {
                float delta_x = data_->at(pt_idx_in_data).x - data_->at(prev_pt_idx_in_data).x;
                float delta_y = data_->at(pt_idx_in_data).y - data_->at(prev_pt_idx_in_data).y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                cum_prev_to_current += delta_dist;
                cum_current_to_nxt -= delta_dist;
            }
           
            // Check if prev_idx needs to move
            while (cum_prev_to_current > extension && prev_idx < traj.size() - 1) {
                float delta_x = data_->at(traj[prev_idx+1]).x - data_->at(traj[prev_idx]).x;
                float delta_y = data_->at(traj[prev_idx+1]).y - data_->at(traj[prev_idx]).y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                if (cum_prev_to_current - delta_dist > extension) {
                    ++prev_idx;
                    cum_prev_to_current -= delta_dist;
                }
                else{
                    break;
                }
            }
            
            // Check if nxt_idx needs to move
            while (cum_current_to_nxt < extension && nxt_idx < traj.size() - 1) {
                float delta_x = data_->at(traj[nxt_idx + 1]).x - data_->at(traj[nxt_idx]).x;
                float delta_y = data_->at(traj[nxt_idx + 1]).y - data_->at(traj[nxt_idx]).y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                cum_current_to_nxt += delta_dist;
                ++nxt_idx;
                if (cum_current_to_nxt > extension) {
                    break;
                }
            }
            
            // Add points from prev_idx to nxt_idx as a new segment
            Segment &this_segment = segments_[pt_idx_in_data];
            this_segment.segLength() = cum_prev_to_current + cum_current_to_nxt;
            //printf("Extracted one segment: from %d to %d, prev_to_cur = %.1f m, cur_to_nxt = %.1f\n", prev_idx, nxt_idx, cum_prev_to_current, cum_current_to_nxt);
            float current_time = data_->at(traj[pt_idx]).t;
            for (size_t croped_idx = prev_idx; croped_idx <= nxt_idx; ++croped_idx) {
                PclPoint &this_point = data_->at(traj[croped_idx]);
                this_segment.points().push_back(SegPoint(this_point.x, this_point.y, this_point.t - current_time, traj[croped_idx]));
            }
            
            if (cum_prev_to_current > 3 * extension || cum_current_to_nxt > 3 * extension){
                this_segment.quality() = false;
            }
          
            prev_pt_idx_in_data = pt_idx_in_data;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Computing segments Done. Time elapsed: %.1f sec\n", elapsed_secs);
  
//    auto highres_begin = chrono::high_resolution_clock::now();
//    int from = 0;
//    int to = 1;
//    float distance;
//    for (int tmp = 0; tmp < 10000; ++tmp) {
//        distance = segments_[from].distanceTo(segments_[to]);
//    }
//    auto highres_end = chrono::high_resolution_clock::now();
//    auto duration = chrono::duration_cast<chrono::nanoseconds>(highres_end - highres_begin).count();
//    printf("Distance from %d to %d is %.2f\n", from, to, distance);
//    cout<<"ns total: "<<duration << " ns."<<endl;
    
//    ofstream test;
//    test.open("test.txt");
//    int selected_traj_idx = 0;
//    test<<trajectories_[selected_traj_idx].size()<<endl;
//    for (size_t i = 0; i < trajectories_[selected_traj_idx].size(); ++i) {
//        vector<int> &traj = trajectories_[selected_traj_idx];
//        test<<data_->at(traj[i]).x<<" "<<data_->at(traj[i]).y<<endl;
//    }
//    
//    for (size_t i = 0; i < trajectories_[selected_traj_idx].size(); ++i) {
//        vector<int> &traj = trajectories_[selected_traj_idx];
//        int idx = traj[i];
//        test<<segments_[idx].points().size()<<endl;
//        for (size_t j=0; j < segments_[idx].points().size(); ++j ) {
//            test<<segments_[idx].points()[j].x<<" "<<segments_[idx].points()[j].y<<endl;
//        }
//    }
//    
//    test.close();
}

void Trajectories::selectSegmentsWithSearchRange(float range){
    search_range_ = range;
    if (!show_segments_)
        return;
    
    if (picked_sample_idxs_.size() == 0){
        return;
    }
    
    recomputeSegmentsForSamples();
    updateSelectedSegments();
}

void Trajectories::recomputeSegmentsForSamples(){
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    for (set<int>::iterator it = picked_sample_idxs_.begin(); it != picked_sample_idxs_.end(); ++it) {
        selectSegmentsNearSample(*it);
    }
}

void Trajectories::updateSelectedSegments(void){
    segment_vertices_.clear();
    segment_colors_.clear();
    segment_idxs_.clear();
    
    // Draw general picked segment in segments_to_draw_
    for (size_t idx = 0; idx < segments_to_draw_.size(); ++idx) {
        Segment &this_seg = segments_[segments_to_draw_[idx]];
        vector<int> idxs;
        Color seg_color = ColorMap::getInstance().getDiscreteColor(idx);
        for (int j = 0; j < this_seg.points().size(); ++j) {
            int orig_idx = this_seg.points()[j].orig_idx;
            idxs.push_back(segment_vertices_.size());
            Vertex &orig_vertex = normalized_vertices_[orig_idx];
            segment_vertices_.push_back(Vertex(orig_vertex.x, orig_vertex.y, Z_SEGMENTS));
            float alpha = 1.0 - 0.7 * static_cast<float>(j) / this_seg.points().size();
            Color pt_color(alpha*seg_color.r, alpha*seg_color.g, alpha*seg_color.b, 1.0);
            segment_colors_.push_back(pt_color);
        }
        segment_idxs_.push_back(idxs);
    }
    
    // Draw segments due to picked samples
    for (size_t idx = 0; idx < segments_to_draw_for_samples_.size(); ++idx) {
        Segment &this_seg = segments_[segments_to_draw_for_samples_[idx]];
        Color &this_color = segments_to_draw_for_samples_colors_[idx];
        vector<int> idxs;
        for (int j = 0; j < this_seg.points().size(); ++j) {
            int orig_idx = this_seg.points()[j].orig_idx;
            idxs.push_back(segment_vertices_.size());
            Vertex &orig_vertex = normalized_vertices_[orig_idx];
            segment_vertices_.push_back(Vertex(orig_vertex.x, orig_vertex.y, Z_SEGMENTS));
            float alpha = 1.0 - 0.7 * static_cast<float>(j) / this_seg.points().size();
            Color pt_color(alpha*this_color.r, alpha*this_color.g, alpha*this_color.b, 1.0);
            segment_colors_.push_back(pt_color);
        }
        segment_idxs_.push_back(idxs);
    }
}

void Trajectories::selectSegmentsNearSample(int sample_id){
    if (segments_.size() == 0)
        return;
    
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    printf("Search range is %2f\n", search_range_);
    
    Color seg_color = ColorMap::getInstance().getDiscreteColor(sample_id);
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        segments_to_draw_for_samples_.push_back(k_indices[i]);
        segments_to_draw_for_samples_colors_.push_back(seg_color);
    }
    
    updateSelectedSegments();
}

void Trajectories::drawSegmentAt(int seg_idx){
    if (segments_.size() == 0){
        return;
    }
    //segments_to_draw_.clear();
    segments_to_draw_.push_back(seg_idx);
    updateSelectedSegments();
}

void Trajectories::clearDrawnSegments(){
    segments_to_draw_.clear();
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    updateSelectedSegments();
}

Segment &Trajectories::segmentAt(int seg_idx){
    return segments_[seg_idx];
}

void Trajectories::interpolateSegmentWithGraph(Graph *g){
    if (g == NULL){
        printf("Empty graph!\n");
        return;
    }
    if (segments_.empty()){
        printf("No segment to interpolate.\n");
        return;
    }
    
    graph_interpolated_segments_.clear();
    graph_interpolated_segments_.resize(segments_.size());
    
    for (size_t seg_id = 0; seg_id < segments_.size(); ++seg_id){
        Segment &seg = segmentAt(seg_id);
        vector<int> seg_pt_idxs;
        for(size_t i = 0; i < seg.points().size(); ++i){
            SegPoint &pt = seg.points()[i];
            seg_pt_idxs.push_back(pt.orig_idx);
        }
        
        vector<int> &result_path = graph_interpolated_segments_[seg_id];
        g->shortestPathInterpolation(seg_pt_idxs, result_path);
    }
}

void Trajectories::interpolateSegmentWithGraphAtSample(int sample_id, Graph *g, vector<int> &nearby_segments, vector<vector<int>> &results){
    if (g == NULL){
        printf("Empty graph!\n");
        return;
    }
    if (segments_.empty()){
        printf("No segment to interpolate.\n");
        return;
    }
    
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    
    //Color seg_color = ColorMap::getInstance().getDiscreteColor(picked_sample_idxs_.size());
    nearby_segments.clear();
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        nearby_segments.push_back(k_indices[i]);
    }
    
    results.clear();
    results.resize(nearby_segments.size());
    for (size_t seg_id = 0; seg_id < nearby_segments.size(); ++seg_id){
        Segment &seg = segmentAt(nearby_segments[seg_id]);
        vector<int> seg_pt_idxs;
        for(size_t i = 0; i < seg.points().size(); ++i){
            SegPoint &pt = seg.points()[i];
            seg_pt_idxs.push_back(pt.orig_idx);
        }
        
        vector<int> &result_path = results[seg_id];
        g->shortestPathInterpolation(seg_pt_idxs, result_path);
    }
}

void Trajectories::clusterSegmentsWithGraphAtSample(int sample_id, Graph *g){
    vector<int> nearby_segment_idxs;
    vector<vector<int>> interpolated_segments;
    interpolateSegmentWithGraphAtSample(sample_id, g, nearby_segment_idxs, interpolated_segments);
    
    printf("%lu nearby segments.\n", nearby_segment_idxs.size());
    
    set<int> remaining_segments;
    for (size_t i = 0;  i < nearby_segment_idxs.size(); ++i) {
        remaining_segments.insert(i);
    }
    
    vector<vector<int>> clusters;
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;

    while (remaining_segments.size() != 0) {
        // Randomly pick a segment
        int idx = rand() % remaining_segments.size();
        set<int>::iterator it = remaining_segments.begin();
        advance(it, idx);
        int selected_seg_idx = *it;
        
        vector<int> &selected_segment = interpolated_segments[*it];
        
        // Check if this segment can be merged to existing segments; Otherwise it will become a new cluster.
        bool absorbed = false;
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            vector<int> &cluster = clusters[cluster_id];
            float max_distance = 0.0f;
            for (size_t s_id = 0; s_id < cluster.size(); ++s_id) {
                key_type this_key(*it, cluster[s_id]);
                key_type reverse_key(cluster[s_id], *it);
                
                // Check distance between selected segment and this cluster segment.
                map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                float dist;
                if (m_it != computed_dist.end()) {
                    dist = m_it->second;
                }
                else{
                    // Compute distance
                    dist = g->DTWDistance(selected_segment, interpolated_segments[cluster[s_id]]);
                    computed_dist[this_key] = dist;
                    computed_dist[reverse_key] = dist;
                }
                if (dist > max_distance) {
                    max_distance = dist;
                }
            }
            
            if (max_distance < 50.0f) {
                cluster.push_back(selected_seg_idx);
                absorbed = true;
            }
        }
        
        if (!absorbed){
            vector<int> new_cluster;
            new_cluster.push_back(selected_seg_idx);
            clusters.push_back(new_cluster);
        }
        
        remaining_segments.erase(it);
    }
    
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    int n_valid_cluster = 0;
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        if (cluster.size() < 5) {
            continue;
        }
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(n_valid_cluster);
        ++n_valid_cluster;
        for (size_t seg_id = 0; seg_id < cluster.size(); ++seg_id) {
            segments_to_draw_for_samples_.push_back(nearby_segment_idxs[cluster[seg_id]]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    printf("%d cluster extracted.\n", n_valid_cluster);
    updateSelectedSegments();
}

int Trajectories::clusterSegmentsUsingDistanceAtSample(int sample_id, double sigma, double threshold, int minClusterSize){
    vector<int> nearby_segment_idxs;
    
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    
    nearby_segment_idxs.clear();
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        nearby_segment_idxs.push_back(k_indices[i]);
    }
    
    printf("%lu nearby segments.\n", nearby_segment_idxs.size());
    
    //clock_t begin = clock();
    int first_idx = -1;
    float max_dist = 0.0f;
    for (size_t i = 0; i < nearby_segment_idxs.size(); ++i){
        Segment &seg = segments_[nearby_segment_idxs[i]];
        if (seg.points().size() < 2) {
            continue;
        }
        float delta_x = seg.points()[0].x - samples_->at(sample_id).x;
        float delta_y = seg.points()[0].y - samples_->at(sample_id).y;
        float dist1 = sqrt(delta_x*delta_x + delta_y*delta_y);
       
        int last_pt_idx = seg.points().size() - 1;
        delta_x = seg.points()[last_pt_idx].x - samples_->at(sample_id).x;
        delta_y = seg.points()[last_pt_idx].y - samples_->at(sample_id).y;
        float dist2 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        float dist = (dist1 > dist2) ? dist1 : dist2;
        if (max_dist < dist) {
            max_dist = dist;
            first_idx = i;
        }
    }
    
    
    // Randomly pick the starting segment
    set<int> remaining_segments;
    for (size_t i = 0;  i < nearby_segment_idxs.size(); ++i) {
        if (i == first_idx) {
            continue;
        }
        remaining_segments.insert(nearby_segment_idxs[i]);
    }
    
    // Start furthest point clustering
    vector<vector<int>> clusters;
    vector<int> first_cluster;
    first_cluster.push_back(nearby_segment_idxs[first_idx]);
    clusters.push_back(first_cluster);
    
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    while (remaining_segments.size() != 0) {
        // Look over remaining segments, find its distance to each existing clusters.
        float max_distance = -1e6;
        int max_segment_idx = -1;
        
        vector<int> segments_to_merge;
        vector<int> merging_to;
        for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
            int seg_id = *it;
            
            // Check the minimum distance of this segment to each of the existing clusters.
            float min_distance = 1e6;
            int min_cluster_id = -1;
            for (size_t i_cluster = 0; i_cluster < clusters.size(); ++i_cluster) {
                vector<int> &current_cluster = clusters[i_cluster];
                // Look through each segment in the cluster
                float tmp_min_distance = 1e6;
                for (size_t j_seg = 0; j_seg < current_cluster.size(); ++j_seg) {
                    int other_seg_id = current_cluster[j_seg];
                    // Reuse distance if it has been computed before
                    key_type this_key(seg_id, other_seg_id);
                    key_type reverse_key(other_seg_id, seg_id);
                    map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                    float tmp_dist = 0.0f;
                    if (m_it != computed_dist.end()) {
                        tmp_dist = m_it->second;
                    }
                    else{
                        // Compute tmp_dist.
                        tmp_dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                        // Insert the computed dist for reuse.
                        //printf("dist = %.2f\n", tmp_dist);
                        computed_dist[this_key] = tmp_dist;
                        computed_dist[reverse_key] = tmp_dist;
                    }
                    if(tmp_dist < tmp_min_distance){
                        tmp_min_distance = tmp_dist;
                    }
                }
                
                if (tmp_min_distance < min_distance) {
                    min_distance = tmp_min_distance;
                    min_cluster_id = i_cluster;
                }
            }
            
            // If min_distance is small, we will merge this segment into that corresponding cluster
            if (min_distance < threshold) {
                // Add this segment id to merging list
                segments_to_merge.push_back(seg_id);
                merging_to.push_back(min_cluster_id);
            }
            
            if (max_distance < min_distance) {
                max_distance = min_distance;
                max_segment_idx = seg_id;
            }
        }
        
        // Remove merging segments
        for (size_t i = 0; i < segments_to_merge.size(); ++i) {
            clusters[merging_to[i]].push_back(segments_to_merge[i]);
            remaining_segments.erase(segments_to_merge[i]);
        }
        
        if (max_distance < threshold) {
            break;
        }
        
        // reate new cluster
        vector<int> new_cluster;
        new_cluster.push_back(max_segment_idx);
        clusters.push_back(new_cluster);
        remaining_segments.erase(max_segment_idx);
    }
    
    //clock_t end = clock();
    //double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    //printf("Clustering %lu segments took %.1f sec.\n", nearby_segment_idxs.size(), elapsed_time);
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    int n_valid_cluster = 0;
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        if (cluster.size() < minClusterSize) {
            continue;
        }
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(n_valid_cluster);
        ++n_valid_cluster;
        for (size_t i = 0; i < cluster.size(); ++i) {
            segments_to_draw_for_samples_.push_back(cluster[i]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    updateSelectedSegments();
    return n_valid_cluster;
}

void Trajectories::clusterSegmentsAtAllSamples(double sigma, int minClusterSize){
    if (samples_->size() == 0)
        return;
    
    int total_n_clusters = 0;
    
    PclPoint point;
    vector<vector<vector<int>>> all_sample_clusters(samples_->size());
    for (size_t sample_id = 0; sample_id < samples_->size(); ++sample_id){
        vector<vector<int>> &clusters = all_sample_clusters[sample_id];
        int n_clusters = fpsSegmentSampling(sample_id, sigma, minClusterSize, clusters);
        if (n_clusters == 0) {
            total_n_clusters += 1;
        }
        else{
            total_n_clusters += n_clusters;
        }
    }
    
    cluster_centers_->clear();
    cluster_popularity_.clear();
    cluster_descriptors_ = new cv::Mat(total_n_clusters, 16, cv::DataType<float>::type, 0.0f);
    int current_cluster_id = 0;
    for (size_t sample_id = 0; sample_id < samples_->size(); ++sample_id){
        vector<vector<int>> &clusters = all_sample_clusters[sample_id];
        // Deal with empty clusterd sample
        if (clusters.size() == 0) {
            point.setCoordinate(samples_->at(sample_id).x, samples_->at(sample_id).y, 0.0f);
            cluster_centers_->push_back(point);
            cluster_popularity_.push_back(0);
            current_cluster_id += 1;
            continue;
        }
        // Compute descriptors for each clusters and store them.
        for (size_t i_cluster = 0; i_cluster < clusters.size(); ++i_cluster){
            vector<int> &cluster = clusters[i_cluster];
            point.setCoordinate(samples_->at(sample_id).x, samples_->at(sample_id).y, 0.0f);
            cluster_centers_->push_back(point);
            cluster_popularity_.push_back(cluster.size());
            float cummulated_count = 0.0f;
            for (size_t j = 0; j < cluster.size(); ++j){
                int seg_id = cluster[j];
                Segment &seg = segments_[seg_id];
                for (size_t k = 0; k < seg.points().size(); ++k) {
                    float delta_x = seg.points()[k].x - samples_->at(sample_id).x;
                    float delta_y = seg.points()[k].y - samples_->at(sample_id).y;
                    float r = sqrt(delta_x*delta_x + delta_y*delta_y);
                    int r_bin_id = 0;
                    if (r > 50.0f) {
                        r_bin_id = 1;
                    }
//                    else if (r > 100.0f && r <= 250.0f){
//                        r_bin_id = 2;
//                    }
//                    else if (r > 250.0f){
//                        r_bin_id = 3;
//                    }
                    
                    float cos_value = delta_x / r;
                    float theta = acos(cos_value) * 180.0f / PI;
                    if (delta_y < 0) {
                        theta += 180;
                    }
                    int theta_bin_id;
                    if (theta > 45.0f && theta <= 135.0f) {
                        theta_bin_id = 1;
                    }
                    else if (theta > 135.0f && theta <= 225.0f){
                        theta_bin_id = 2;
                    }
                    else if (theta > 225.0f && theta <= 315.0f){
                        theta_bin_id = 3;
                    }
                    else{
                        theta_bin_id = 0;
                    }
                    
                    int bin_idx = r_bin_id * theta_bin_id;
                    
                    if (seg.points()[k].t >= 0) {
                        bin_idx += 8;
                    }
                    cluster_descriptors_->at<float>(current_cluster_id, bin_idx) += 1.0;
                    cummulated_count += 1.0f;
                }
            }
           
            // Normalize this distribution
            if (cummulated_count > 0.0f) {
                for (int s=0; s < 16; ++s) {
                    cluster_descriptors_->at<float>(current_cluster_id, s) /= cummulated_count;
                }
            }
            current_cluster_id += 1;
        }
    }
    
    //cout << *cluster_descriptors_ << endl;
    
    cluster_center_search_tree_->setInputCloud(cluster_centers_);
    
    //cv::flann::Index flann_index(obj_32f, cv::flann::KDTreeIndexParams());
    
    //cv::flann::Index flann_index(, cv::flann::Kd);
    
//    segments_to_draw_for_samples_.clear();
//    segments_to_draw_for_samples_colors_.clear();
//    for (size_t sample_id = 0; sample_id < samples_->size(); ++sample_id) {
//        vector<int> &cluster_center = sample_segment_clusters_[sample_id];
//        Color cluster_color = ColorMap::getInstance().getDiscreteColor(sample_id);
//        for (size_t i = 0; i < cluster_center.size(); ++i) {
//            segments_to_draw_for_samples_.push_back(cluster_center[i]);
//            segments_to_draw_for_samples_colors_.push_back(cluster_color);
//        }
//    }
//    
//    updateSelectedSegments();
}

int Trajectories::fpsSegmentSampling(int sample_id, double sigma, int minClusterSize, vector<vector<int>> &clusters){
    vector<int> nearby_segment_idxs;
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    
    nearby_segment_idxs.clear();
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        nearby_segment_idxs.push_back(k_indices[i]);
    }
    
    if (nearby_segment_idxs.size() == 0)
        return 0;
    
    int first_idx = -1;
    float max_dist = 0.0f;
    for (size_t i = 0; i < nearby_segment_idxs.size(); ++i){
        Segment &seg = segments_[nearby_segment_idxs[i]];
        if (seg.points().size() < 2) {
            continue;
        }
        float delta_x = seg.points()[0].x - samples_->at(sample_id).x;
        float delta_y = seg.points()[0].y - samples_->at(sample_id).y;
        float dist1 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        int last_pt_idx = seg.points().size() - 1;
        delta_x = seg.points()[last_pt_idx].x - samples_->at(sample_id).x;
        delta_y = seg.points()[last_pt_idx].y - samples_->at(sample_id).y;
        float dist2 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        float dist = (dist1 > dist2) ? dist1 : dist2;
        if (max_dist < dist) {
            max_dist = dist;
            first_idx = i;
        }
    }
    
    //clock_t begin = clock();
    set<int> remaining_segments;
    for (size_t i = 0;  i < nearby_segment_idxs.size(); ++i) {
        if (i == first_idx) {
            continue;
        }
        remaining_segments.insert(nearby_segment_idxs[i]);
    }
    
    // Start furthest point clustering
    vector<vector<int>> tmp_clusters;
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    int MAX_CLUSTERS = 20;
    vector<float> max_distances(MAX_CLUSTERS, 0.0f);
    vector<int> selected_centers;
    selected_centers.push_back(nearby_segment_idxs[first_idx]);
    for (int i=0; i < MAX_CLUSTERS-1; ++i){
        float max_distance = -1;
        int max_segment_idx = -1;
        for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
            int seg_id = *it;
            float min_distance = POSITIVE_INFINITY;
            // Compute distance between current segment and the already selected cluster center segments.
            for (size_t ith_selected = 0; ith_selected < selected_centers.size(); ++ith_selected) {
                int other_seg_id = selected_centers[ith_selected];
                // Reuse distance if it has been computed before
                key_type this_key(seg_id, other_seg_id);
                key_type reverse_key(other_seg_id, seg_id);
                map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                float dist = 0.0f;
                if (m_it != computed_dist.end()) {
                    dist = m_it->second;
                }
                else{
                    // Compute dist.
                    dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                    // Insert the computed dist for reuse.
                    computed_dist[this_key] = dist;
                    computed_dist[reverse_key] = dist;
                }
                
                if (dist < min_distance) {
                    min_distance = dist;
                }
            }
            // Record the current max distance segment and its id
            if (max_distance < min_distance) {
                max_distance = min_distance;
                max_segment_idx = seg_id;
            }
        }
        
        // Insert max_segment_idx into selected_centers
        selected_centers.push_back(max_segment_idx);
        max_distances[i] = max_distance;
    }
    
    // Determine the number of clusters we need to use
    int n_clusters = 0;
    float threshold = POSITIVE_INFINITY;
    bool threshold_selected = false;
    for (int i = MAX_CLUSTERS-1; i >= 0; --i) {
        if (max_distances[i] > 1e-3 && !threshold_selected) {
            threshold = 1.5*max_distances[i];
            threshold_selected = true;
            //printf("Min value is: %.2f\n", threshold);
        }
        if (max_distances[i] > threshold) {
            n_clusters = i + 2;
            break;
        }
    }
    
    // Initialiaze tmp_clusters
    for (int i = 0; i < n_clusters; ++i) {
        vector<int> new_cluster;
        new_cluster.push_back(selected_centers[i]);
        tmp_clusters.push_back(new_cluster);
        remaining_segments.erase(selected_centers[i]);
    }
    
    // Put remaining segments to clusters
    for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
        int seg_id = *it;
        float min_distance = POSITIVE_INFINITY;
        int min_cluster_id = -1;
        // Compute distance between current segment and the already selected cluster center segments.
        for (size_t ith_cluster = 0; ith_cluster < tmp_clusters.size(); ++ith_cluster) {
            int other_seg_id = tmp_clusters[ith_cluster][0];
            // Reuse distance if it has been computed before
            key_type this_key(seg_id, other_seg_id);
            key_type reverse_key(other_seg_id, seg_id);
            map<key_type, float>::iterator m_it = computed_dist.find(this_key);
            float dist = 0.0f;
            if (m_it != computed_dist.end()) {
                dist = m_it->second;
            }
            else{
                // Compute dist.
                dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                // Insert the computed dist for reuse.
                computed_dist[this_key] = dist;
                computed_dist[reverse_key] = dist;
            }
            
            if (dist < min_distance) {
                min_distance = dist;
                min_cluster_id = ith_cluster;
            }
        }
        if (min_cluster_id != -1)
            tmp_clusters[min_cluster_id].push_back(seg_id);
    }
    
    // Record GOOD clusters into clusters.
    clusters.clear();
    clusters_.clear();
    for (int i = 0; i < tmp_clusters.size(); ++i) {
        vector<int> &cluster = tmp_clusters[i];
        if (cluster.size() >= minClusterSize) {
            clusters.push_back(cluster);
            clusters_.push_back(cluster);
        }
    }
    
    //clock_t end = clock();
    //double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    //printf("Clustering %lu segments took %.1f sec.\n", nearby_segment_idxs.size(), elapsed_time);
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(cluster_id);
        for (size_t i = 0; i < cluster.size(); ++i) {
            segments_to_draw_for_samples_.push_back(cluster[i]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    updateSelectedSegments();
    
    // Write clustering results to .txt file
    ofstream test;
    test.open("clustering_results.txt");
    test<<clusters.size()<<endl;
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<int> &cluster = clusters[i];
        test << cluster.size() << endl;
        for (size_t j = 0; j < cluster.size(); ++j) {
            Segment s = segments_[cluster[j]];
            test << s.points().size() <<endl;
            for (size_t k = 0; k < s.points().size(); ++k) {
                test << s.points()[k].x << " " << s.points()[k].y << " " << s.points()[k].t << endl;
            }
        }
    }
    
    test.close();
    
    return clusters.size();
}

int Trajectories::fpsMultiSiteSegmentClustering(vector<int> selected_samples, double sigma, int minClusterSize, vector<vector<int>> &clusters){
    set<int> nearby_segment_idxs;
    // Search GPS points around the each sample point
    for(size_t i = 0; i < selected_samples.size(); ++i){
        int sample_id = selected_samples[i];
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
        
        for (size_t i = 0; i < k_indices.size(); ++i) {
            if (!segments_[k_indices[i]].quality()) {
                continue;
            }
            nearby_segment_idxs.insert(k_indices[i]);
        }
    }
    
    if (nearby_segment_idxs.size() == 0)
        return 0;
    
    int first_idx = rand() % nearby_segment_idxs.size();
    int first_segment_id = -1;
    
    clock_t begin = clock();
    set<int> remaining_segments;
    int i = 0;
    for(set<int>::iterator it = nearby_segment_idxs.begin(); it != nearby_segment_idxs.end(); ++it){
        if (i == first_idx) {
            first_segment_id = *it;
            continue;
        }
        remaining_segments.insert(*it);
        ++i;
    }
    
    // Start furthest point clustering
    vector<vector<int>> tmp_clusters;
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    int MAX_CLUSTERS = 20;
    vector<float> max_distances(MAX_CLUSTERS, 0.0f);
    vector<int> selected_centers;
    selected_centers.push_back(first_segment_id);
    for (int i=0; i < MAX_CLUSTERS-1; ++i){
        float max_distance = -1;
        int max_segment_idx = -1;
        for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
            int seg_id = *it;
            float min_distance = POSITIVE_INFINITY;
            // Compute distance between current segment and the already selected cluster center segments.
            for (size_t ith_selected = 0; ith_selected < selected_centers.size(); ++ith_selected) {
                int other_seg_id = selected_centers[ith_selected];
                // Reuse distance if it has been computed before
                key_type this_key(seg_id, other_seg_id);
                key_type reverse_key(other_seg_id, seg_id);
                map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                float dist = 0.0f;
                if (m_it != computed_dist.end()) {
                    dist = m_it->second;
                }
                else{
                    // Compute dist.
                    dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                    // Insert the computed dist for reuse.
                    computed_dist[this_key] = dist;
                    computed_dist[reverse_key] = dist;
                }
                
                if (dist < min_distance) {
                    min_distance = dist;
                }
            }
            // Record the current max distance segment and its id
            if (max_distance < min_distance) {
                max_distance = min_distance;
                max_segment_idx = seg_id;
            }
        }
        
        // Insert max_segment_idx into selected_centers
        selected_centers.push_back(max_segment_idx);
        max_distances[i] = max_distance;
    }
    
    // Determine the number of clusters we need to use
    int n_clusters = 0;
    float threshold = POSITIVE_INFINITY;
    bool threshold_selected = false;
    for (int i = MAX_CLUSTERS-1; i >= 0; --i) {
        if (max_distances[i] > 1e-3 && !threshold_selected) {
            threshold = 1.5*max_distances[i];
            threshold_selected = true;
            //printf("Min value is: %.2f\n", threshold);
        }
        if (max_distances[i] > threshold) {
            n_clusters = i + 2;
            break;
        }
    }
    
    // Initialiaze tmp_clusters
    for (int i = 0; i < n_clusters; ++i) {
        vector<int> new_cluster;
        new_cluster.push_back(selected_centers[i]);
        tmp_clusters.push_back(new_cluster);
        remaining_segments.erase(selected_centers[i]);
    }
    
    // Put remaining segments to clusters
    for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
        int seg_id = *it;
        float min_distance = POSITIVE_INFINITY;
        int min_cluster_id = -1;
        // Compute distance between current segment and the already selected cluster center segments.
        for (size_t ith_cluster = 0; ith_cluster < tmp_clusters.size(); ++ith_cluster) {
            int other_seg_id = tmp_clusters[ith_cluster][0];
            // Reuse distance if it has been computed before
            key_type this_key(seg_id, other_seg_id);
            key_type reverse_key(other_seg_id, seg_id);
            map<key_type, float>::iterator m_it = computed_dist.find(this_key);
            float dist = 0.0f;
            if (m_it != computed_dist.end()) {
                dist = m_it->second;
            }
            else{
                // Compute dist.
                dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                // Insert the computed dist for reuse.
                computed_dist[this_key] = dist;
                computed_dist[reverse_key] = dist;
            }
            
            if (dist < min_distance) {
                min_distance = dist;
                min_cluster_id = ith_cluster;
            }
        }
        if (min_cluster_id != -1)
            tmp_clusters[min_cluster_id].push_back(seg_id);
    }
    
    // Record GOOD clusters into clusters.
    clusters.clear();
    clusters_.clear();
    for (int i = 0; i < tmp_clusters.size(); ++i) {
        vector<int> &cluster = tmp_clusters[i];
        if (cluster.size() >= minClusterSize) {
            clusters.push_back(cluster);
            clusters_.push_back(cluster);
        }
    }
    
    clock_t end = clock();
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    printf("Clustering %lu segments took %.1f sec.\n", nearby_segment_idxs.size(), elapsed_time);
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(cluster_id);
        for (size_t i = 0; i < cluster.size(); ++i) {
            segments_to_draw_for_samples_.push_back(cluster[i]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    updateSelectedSegments();
    
    // Write clustering results to .txt file
//    ofstream test;
//    test.open("clustering_results.txt");
//    test<<clusters.size()<<endl;
//    for (size_t i = 0; i < clusters.size(); ++i) {
//        vector<int> &cluster = clusters[i];
//        test << cluster.size() << endl;
//        for (size_t j = 0; j < cluster.size(); ++j) {
//            Segment s = segments_[cluster[j]];
//            test << s.points().size() <<endl;
//            for (size_t k = 0; k < s.points().size(); ++k) {
//                test << s.points()[k].x << " " << s.points()[k].y << " " << s.points()[k].t << endl;
//            }
//        }
//    }
//    test.close();
    
    return clusters.size();
}

void Trajectories::showClusterAtIdx(int idx){
    if (idx < 0 || idx >= clusters_.size())
        return;
    
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    vector<int> &cluster = clusters_[idx];
    
    Color cluster_color = ColorMap::getInstance().getDiscreteColor(idx);
    for (size_t i = 0; i < cluster.size(); ++i) {
        segments_to_draw_for_samples_.push_back(cluster[i]);
        segments_to_draw_for_samples_colors_.push_back(cluster_color);
    }
    updateSelectedSegments();
}

float Trajectories::computeSegmentDistance(int seg_idx1, int seg_idx2, double sigma){
    // Dynamic Time Wraping distance using different search range with t.
    Segment &seg1 = segments_[seg_idx1];
    Segment &seg2 = segments_[seg_idx2];
    
    if(seg1.points().size() < 2 || seg2.points().size() < 2)
        return POSITIVE_INFINITY;
    
    // Add head and tail nodes for DTWD
    SegPoint &seg1_start = seg1.points()[0];
    int seg1_size = seg1.points().size();
    SegPoint &seg1_end = seg1.points()[seg1_size-1];
    
    SegPoint &seg2_start = seg2.points()[0];
    int seg2_size = seg2.points().size();
    SegPoint &seg2_end = seg2.points()[seg2_size-1];
    
    float seg1_radius1 = sigma * abs(seg1_start.t);
    float seg1_radius2 = sigma * abs(seg1_end.t);
    float seg2_radius1 = sigma * abs(seg2_start.t);
    float seg2_radius2 = sigma * abs(seg2_end.t);
    
    float delta1_x = seg1_start.x - seg2_start.x;
    float delta1_y = seg1_start.y - seg2_start.y;
    float dist1 = sqrt(delta1_x*delta1_x+delta1_y*delta1_y);
    
    float delta2_x = seg1_end.x - seg2_end.x;
    float delta2_y = seg1_end.y - seg2_end.y;
    float dist2 = sqrt(delta2_x*delta2_x+delta2_y*delta2_y);
    
    float delta3_x = seg1_start.x - seg2_end.x;
    float delta3_y = seg1_start.y - seg2_end.y;
    float dist3 = sqrt(delta3_x*delta3_x+delta3_y*delta3_y);
    
    float delta4_x = seg1_end.x - seg2_start.x;
    float delta4_y = seg1_end.y - seg2_start.y;
    float dist4 = sqrt(delta4_x*delta4_x+delta4_y*delta4_y);
    
    float tmp_dist1 = dist1 - seg1_radius1 - seg2_radius1;
    float tmp_dist2 = dist2 - seg1_radius2 - seg2_radius2;
    
    if (tmp_dist1 < 0)
        tmp_dist1 = 0.0f;
    if (tmp_dist2 < 0)
        tmp_dist2 = 0.0f;
    
    float penalty = 10.0f;
    
    float dist = tmp_dist1 + tmp_dist2;
    
    if (dist1 + dist2 > dist3 + dist4)
        dist *= penalty;
    
    return dist;
}

float Trajectories::segPointDistance(int idx1, int idx2, Segment &seg1, Segment &seg2, double sigma){
    SegPoint &pt1 = seg1.points()[idx1];
    SegPoint &pt2 = seg2.points()[idx2];
    
    float dist = 0.0f;
   
    float radius1 = sigma * abs(pt1.t);
    float radius2 = sigma * abs(pt2.t);
    float sign1 = 1.0f;
    float sign2 = 1.0f;
    if (pt1.t < 0)
        sign1 = -1.0f;
    if (pt2.t < 0)
        sign2 = -1.0f;
   
    float delta_x = sign1*pt1.x - sign2*pt2.x;
    float delta_y = sign1*pt1.y - sign2*pt2.y;
    dist += sqrt(delta_x*delta_x + delta_y*delta_y);
    dist -= radius1;
    dist -= radius2;
    if (dist < 0.0f) {
        dist = 0.0f;
    }
    
    return dist;
}

void Trajectories::computeDistanceGraph(){
    if (distance_graph_ != NULL) {
        delete distance_graph_;
    }
    
    distance_graph_ = new DistanceGraph();
    float grid_size = 10; //in meters
    samplePointCloud(grid_size, distance_graph_vertices_, distance_graph_search_tree_);
    
    distance_graph_->computeGraphUsingGpsPointCloud(data_, tree_, distance_graph_vertices_, distance_graph_search_tree_, grid_size);
    prepareForVisualization(display_bound_box_);
   
    distance_graph_->Distance(data_->at(10), data_->at(10000));
}

void Trajectories::samplePointCloud(float neighborhood_size, PclPointCloud::Ptr &samples, PclSearchTree::Ptr &sample_tree){
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    printf("bound box: %.2f, %.2f, %.2f, %.2f\n", bound_box_[0], bound_box_[1], bound_box_[2], bound_box_[3]);
    
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
        long pt_i = floor((point.x - bound_box_[0]) / step_x);
        long pt_j = floor((point.y - bound_box_[2]) / step_y);
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
        float pt_x = bound_box_[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box_[2] + (pt_j+0.5)*step_y;
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
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
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
        int pt_i = floor((point.x - bound_box_[0]) / step_x);
        int pt_j = floor((point.y - bound_box_[2]) / step_y);
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
        float pt_x = bound_box_[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box_[2] + (pt_j+0.5)*step_y;
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
            
			samples_->push_back(point);
        }
    }
    
    sample_tree_->setInputCloud(samples_);
    printf("Sampling Done. Totally %zu samples.\n", samples_->size());
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
    
    if (sqrt(k_sqr_distances[0]) < search_distance){
        auto ret = picked_sample_idxs_.emplace(k_indices[0]);
        if (ret.second) {
            // Update segment_to_draw_for_samples_ and corresponding segment vertices and colors
            selectSegmentsNearSample(k_indices[0]);
        }
    }
}

void Trajectories::clearPickedSamples(void){
    picked_sample_idxs_.clear();
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    updateSelectedSegments();
}
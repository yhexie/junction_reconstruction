//
//  features.cpp
//  junction_reconstruction
//
//  Created by Chen Chen on 1/8/15.
//
//

#include "features.h"
#include <fstream>

void computeQueryInitFeatureAt(float radius, PclPoint& point, Trajectories* trajectories, query_init_sample_type& feature, float canonical_heading){
    // Initialize feature to zero.
    feature = 0.0f;
    
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    // Heading histogram parameters
    int N_HEADING_BINS = 8;
    float DELTA_HEADING_BIN = 360.0f / N_HEADING_BINS;
    
    // Speed histogram
    int N_SPEED_BINS = 4;
    float MAX_LOW_SPEED = 5.0f; // meter per second
    float MAX_MID_LOW_SPEED = 10.0f; // meter per second
    float MAX_MID_HIGH_SPEED = 20.0f; // meter per second
    
    // Point density histogram parameters
    float SEARCH_RADIUS = radius; // in meters
    
    vector<int> k_indices;
    vector<float> k_dist_sqrs;
    tree->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dist_sqrs);
    
    // Histograms
    int N_LATERAL_DIST = 8;
    float LAT_BIN_RES = 5.0f; // in meters
    vector<double> speed_heading_hist(N_HEADING_BINS*N_SPEED_BINS, 0.0f);
    vector<double> fwd_lateral_dist(N_LATERAL_DIST, 0.0f);
    vector<double> bwd_lateral_dist(N_LATERAL_DIST, 0.0f);
    float canonical_heading_in_radius = canonical_heading * PI / 180.0f;
    Eigen::Vector3d canonical_dir(cos(canonical_heading_in_radius),
                           sin(canonical_heading_in_radius),
                           0.0f);
    
    for(size_t i = 0; i < k_indices.size(); ++i){
        PclPoint& pt = data->at(k_indices[i]);
        float speed = pt.speed * 1.0f / 100.0f;
        float delta_heading = deltaHeading1MinusHeading2(pt.head, canonical_heading) + 0.5f * DELTA_HEADING_BIN;
        if (delta_heading < 0.0f) {
            delta_heading += 360.0f;
        }
        
        int heading_bin_idx = floor(delta_heading / DELTA_HEADING_BIN);
        
        if (heading_bin_idx == 0) {
            Eigen::Vector3d vec(pt.x - point.x,
                         pt.y - point.y,
                         0.0f);
            
            float parallel_distance = vec.cross(canonical_dir)[2];
            int lat_bin_idx = floor(abs(parallel_distance / LAT_BIN_RES));
            if (lat_bin_idx > 3) {
                lat_bin_idx = 3;
            }
            if (parallel_distance > 0.0f) {
                lat_bin_idx += 4;
            }
            fwd_lateral_dist[lat_bin_idx] += 1.0f;
        }
        
        if (heading_bin_idx == 4) {
            Eigen::Vector3d vec(pt.x - point.x,
                                pt.y - point.y,
                                0.0f);
            
            float parallel_distance = canonical_dir.cross(vec)[2];
            int lat_bin_idx = floor(abs(parallel_distance / LAT_BIN_RES));
            if (lat_bin_idx > 3) {
                lat_bin_idx = 3;
            }
            if (parallel_distance > 0.0f) {
                lat_bin_idx += 4;
            }
            bwd_lateral_dist[lat_bin_idx] += 1.0f;
        }
        
        int speed_bin_idx = 0;
        if (speed < MAX_LOW_SPEED) {
            speed_bin_idx = 0;
        }
        else if(speed < MAX_MID_LOW_SPEED){
            speed_bin_idx = 1;
        }
        else if (speed < MAX_MID_HIGH_SPEED){
            speed_bin_idx = 2;
        }
        else{
            speed_bin_idx = 3;
        }
        
        int bin_idx = speed_bin_idx + heading_bin_idx * N_SPEED_BINS;
        speed_heading_hist[bin_idx] += 1.0f;
    }
    
    // Normalize speed_heading_hist
    double cum_sum = 0.0f;
    for (size_t i = 0; i < speed_heading_hist.size(); ++i) {
        cum_sum += speed_heading_hist[i];
    }
    if (cum_sum > 1.0f) {
        for (size_t i = 0; i < speed_heading_hist.size(); ++i) {
            speed_heading_hist[i] /= cum_sum;
        }
    }
    
    // Normalize fwd_lateral_dist
    cum_sum = 0.0f;
    for (size_t i = 0; i < fwd_lateral_dist.size(); ++i) {
        cum_sum += fwd_lateral_dist[i];
    }
    if (cum_sum > 1.0f) {
        for (size_t i = 0; i < fwd_lateral_dist.size(); ++i) {
            fwd_lateral_dist[i] /= cum_sum;
        }
    }
    
    // Normalize bwd_lateral_dist
    cum_sum = 0.0f;
    for (size_t i = 0; i < bwd_lateral_dist.size(); ++i) {
        cum_sum += bwd_lateral_dist[i];
    }
    if (cum_sum > 1.0f) {
        for (size_t i = 0; i < bwd_lateral_dist.size(); ++i) {
            bwd_lateral_dist[i] /= cum_sum;
        }
    }
    
    // Store the distributions into feature
    for (int i = 0; i < speed_heading_hist.size(); ++i) {
        feature(i) = speed_heading_hist[i];
    }
    
    int n1 = speed_heading_hist.size();
    for (int i = 0; i < fwd_lateral_dist.size(); ++i) {
        feature(i + n1) = fwd_lateral_dist[i];
    }
    
    int n2 = fwd_lateral_dist.size() + n1;
    for (int i = 0; i < bwd_lateral_dist.size(); ++i) {
        feature(i + n2) = bwd_lateral_dist[i];
    }
}

void trainQueryInitClassifier(vector<query_init_sample_type>& samples,
                              vector<int>& orig_labels,
                              query_init_decision_function& df){
    if (samples.size() != orig_labels.size()) {
        cout << "WARNING: sample and label do not have same size." << endl;
        return;
    }
    
    cout << "Start query init classifier training..." << endl;
    cout << "\tSample size: " << samples.size() << endl;
    
    vector<double> labels(orig_labels.size()); // convert int label to double (for dlib)
    for(int i = 0; i < orig_labels.size(); ++i){
        labels[i] = static_cast<double>(orig_labels[i]);
    }
    
    query_init_ovo_trainer trainer;
    dlib::krr_trainer<rbf_kernel> rbf_trainer;
    rbf_trainer.set_kernel(rbf_kernel(0.1f));
    
    trainer.set_trainer(rbf_trainer);
    dlib::randomize_samples(samples, labels);
    
    cout << "\tCross validation: \n" << dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
    
    df = trainer.train(samples, labels);
}

void computeQueryQFeatureAt(float radius,
                            PclPoint& point,
                            Trajectories* trajectories,
                            query_q_sample_type& feature,
                            float heading,
                            bool is_oneway,
                            bool is_reverse_dir){
    // Feature parameters
    float SEARCH_RADIUS = radius; // in radius
    float ANGLE_THRESHOLD = 10.0f;
    float MAX_DIST_TO_ROAD_CENTER = 10.0f; // in meters
    
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    vector<int> k_indices;
    vector<float> k_dists;
    PclPoint search_point;
    search_point.setCoordinate(point.x, point.y, 0.0f);
    tree->radiusSearch(search_point, SEARCH_RADIUS, k_indices, k_dists);
    
//    float growing_heading = heading;
//    if (is_reverse_dir) {
//        growing_heading += 180.0f;
//        if (growing_heading > 360.0f) {
//            growing_heading -= 360.0f;
//        }
//    }
//    
//    float growing_heading_in_radius = growing_heading / 180.0f * PI;
//    Eigen::Vector2d dir(cos(growing_heading_in_radius), sin(growing_heading_in_radius)); // growing direction
//    
//    //    feature_vertices_.push_back(SceneConst::getInstance().normalize(search_point.x, search_point.y, Z_FEATURES));
//    //    feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
//    
//    // Filter nearby point, and only leave the compatible points
//    vector<int> compatible_points;
//    set<int>    allowable_trajs;
//    map<int, float> allowable_traj_ts;
//    float OVERSHOOT = 1.0f;
//    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//        PclPoint& pt = trajectories->data()->at(*it);
//        Eigen::Vector2d vec(pt.x - point.x, pt.y - point.y);
//        float dot_product = vec.dot(dir);
//        
//        if (dot_product > OVERSHOOT) {
//            continue;
//        }
//        
//        float length = vec.norm();
//        float dist_to_road = sqrt(length*length - dot_product*dot_product);
//        
//        if (dist_to_road > MAX_DIST_TO_ROAD_CENTER) {
//            continue;
//        }
//        
//        // Check angle
//        float pt_heading = pt.head;
//        
//        float delta_angle = abs(pt_heading - heading);
//        if(delta_angle > 180.0f){
//            delta_angle = 360.0f - delta_angle;
//        }
//        
//        bool angle_is_acceptable = true;
//        if (is_oneway) {
//            if (delta_angle > ANGLE_THRESHOLD) {
//                angle_is_acceptable = false;
//            }
//        }
//        else{
//            if (delta_angle > 90.0f) {
//                delta_angle = 180.0f - delta_angle;
//            }
//            if (delta_angle > ANGLE_THRESHOLD) {
//                angle_is_acceptable = false;
//            }
//        }
//        
//        if (angle_is_acceptable) {
//            compatible_points.push_back(*it);
//            //            feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
//            //            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
//            allowable_trajs.insert(pt.id_trajectory);
//            allowable_traj_ts[pt.id_trajectory] = pt.t;
//        }
//    }
//    
//    //    printf("compatible points: %lu\n", compatible_points.size());
//    
//    // Compute features
//    vector<int> k_indices1;
//    vector<float> k_dists1;
//    vector<int> points_to_consider;
//    float MAX_DELTA_T = 100.0f;
//    tree->radiusSearch(search_point, 6 * SEARCH_RADIUS, k_indices1, k_dists1);
//    for (vector<int>::iterator it=k_indices1.begin(); it != k_indices1.end(); ++it) {
//        int id_traj = data->at(*it).id_trajectory;
//        const bool is_allowable = allowable_trajs.find(id_traj) != allowable_trajs.end();
//        if (is_allowable) {
//            float traj_t = allowable_traj_ts.find(id_traj)->second;
//            
//            float delta_t = abs(traj_t - data->at(*it).t);
//            if (delta_t > MAX_DELTA_T) {
//                continue;
//            }
//            
//            PclPoint& pt = trajectories->data()->at(*it);
//            Eigen::Vector2d vec(pt.x - point.x, pt.y - point.y);
//            float dot_product = vec.dot(dir);
//            
//            if (dot_product < OVERSHOOT) {
//                continue;
//            }
//            points_to_consider.push_back(*it);
//            //            feature_vertices_.push_back(SceneConst::getInstance().normalize(data->at(*it).x, data->at(*it).y, Z_FEATURES));
//            //            Color orange = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
//            //            Color this_color(orange.r*weight, orange.g*weight, orange.b*weight, 1.0f);
//            //            feature_colors_.push_back(this_color);
//        }
//    }
//    
//    // Compute features
//    float DELTA_HEADING = 15.0f; // in degrees
//    int N_HEADING_BIN = ceil(360.0f / DELTA_HEADING);
//    DELTA_HEADING = 360.0f / N_HEADING_BIN;
//    float DELTA_R = 25.0f;
//    int N_R_BIN = 4;
//    int N_TOTAL_BIN = N_HEADING_BIN * N_R_BIN + 1;
//    
//    feature.clear();
//    feature.resize(N_TOTAL_BIN, 0.0f);
//    float heading_in_radius = heading / 180.0f * PI;
//    Eigen::Vector3d heading_dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
//    for (vector<int>::iterator it = points_to_consider.begin(); it != points_to_consider.end(); ++it) {
//        PclPoint& pt = data->at(*it);
//        
//        float traj_t = allowable_traj_ts.find(pt.id_trajectory)->second;
//        float delta_t = abs(traj_t - data->at(*it).t);
//        float weight = exp(-2.0f * delta_t * delta_t / MAX_DELTA_T / MAX_DELTA_T);
//        
//        Eigen::Vector3d vec(pt.x - point.x, pt.y - point.y, 0.0f);
//        float r_dist = vec.norm();
//        vec.normalize();
//        
//        float dot_value = vec.dot(heading_dir);
//        if(dot_value > 1.0f){
//            dot_value = 1.0f;
//        }
//        if (dot_value < -1.0f) {
//            dot_value = -1.0f;
//        }
//        
//        float cross_value = heading_dir.cross(heading_dir)[2];
//        float angle = acos(dot_value) * 180.0f / PI;
//        if (cross_value < 0) {
//            angle = 360.0f - angle;
//        }
//        
//        int r_bin_idx = floor(r_dist / DELTA_R);
//        if(r_bin_idx >= N_R_BIN){
//            r_bin_idx = N_R_BIN - 1;
//        }
//        
//        int heading_bin_idx = floor(angle / DELTA_HEADING);
//        
//        int feature_bin_idx = heading_bin_idx + r_bin_idx * N_HEADING_BIN;
//        feature[feature_bin_idx] += weight;
//    }
//    
//    //Normalize heading feature
//    float sum = 0.0f;
//    for (int i = 0; i < N_TOTAL_BIN - 1; ++i) {
//        sum += feature[i];
//    }
//    if (sum > 0.1f) {
//        for (int i = 0; i < N_TOTAL_BIN - 1; ++i) {
//            feature[i] /= sum;
//        }
//    }
//    if (is_oneway) {
//        feature[N_TOTAL_BIN-1] = 0.0f; // oneway
//    }
//    else{
//        feature[N_TOTAL_BIN-1] = 1.0f; // twoway
//    }
}

QueryInitFeatureSelector::QueryInitFeatureSelector(QObject* parent, Trajectories* trajectories) : Renderable(parent) {
    trajectories_ = trajectories;
    osmMap_ = NULL;
    features_.clear();
    labels_.clear();
    
    decision_function_valid_ = false;
    
    feature_vertices_.clear();
    feature_colors_.clear();
}

QueryInitFeatureSelector::~QueryInitFeatureSelector(){
}

void QueryInitFeatureSelector::setTrajectories(Trajectories* new_trajectories){
    trajectories_ = new_trajectories;
}

void QueryInitFeatureSelector::computeLabelAt(float radius, PclPoint& pt, int& label){
    float SEARCH_RADIUS = 15.0f; // in meter
    float HEADING_THRESHOLD = 15.0f; // in degrees
    
    // Check if there is nearby points
    PclSearchTree::Ptr& tree = trajectories_->tree();
    vector<int> k_indices;
    vector<float> k_dist_sqrs;
    tree->radiusSearch(pt, radius, k_indices, k_dist_sqrs);
    if(k_indices.size() < 2){
        label = NON_OBVIOUS_ROAD;
        return;
    }
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
    PclSearchTree::Ptr& map_search_tree = osmMap_->map_search_tree();
    
    vector<int> map_k_indices;
    vector<float> map_k_dists;
    map_search_tree->radiusSearch(pt, SEARCH_RADIUS, map_k_indices, map_k_dists);
    bool is_obvious_road = true;
    int current_way_id = pt.id_trajectory;
    set<int> nearby_ways;
    nearby_ways.insert(current_way_id);
    
    bool is_current_way_highway = false;
    const WAYTYPE& current_way_type = osmMap_->ways()[current_way_id].wayType();
    if (current_way_type == MOTORWAY ||
        current_way_type == PRIMARY) {
        is_current_way_highway = true;
    }
    
    float maximum_delta_heading = 0;
    for (size_t i = 0; i < map_k_indices.size(); ++i){
        PclPoint& nearby_pt = map_sample_points->at(map_k_indices[i]);
        int nearby_way_id = nearby_pt.id_trajectory;
        if (nearby_way_id == current_way_id) {
            continue;
        }
        
        if(osmMap_->twoWaysEquivalent(current_way_id, nearby_way_id)){
            continue;
        }
        
        float delta_heading = abs(deltaHeading1MinusHeading2(pt.head, nearby_pt.head));
        // If current way is higyway, rule out the other direction highway
        if (is_current_way_highway) {
            // Check heading
            if (delta_heading > 180.0f - HEADING_THRESHOLD) {
                // Check highway
                const WAYTYPE& nearby_way_type = osmMap_->ways()[nearby_way_id].wayType();
                if (nearby_way_type == MOTORWAY ||
                    nearby_way_type == PRIMARY) {
                    continue;
                }
            }
        }
        
        if(!nearby_ways.emplace(nearby_way_id).second){
            continue;
        }
        
        // Check heading
        if(!osmMap_->ways()[current_way_id].isOneway() || !osmMap_->ways()[nearby_pt.id_trajectory].isOneway()){
            if(delta_heading > 90.0f){
                delta_heading = 180.0f - delta_heading;
            }
        }
        
        if(delta_heading > maximum_delta_heading){
            maximum_delta_heading = delta_heading;
        }
       
    }
    
    if (nearby_ways.size() > 3) {
        is_obvious_road = false;
    }
    
    if(maximum_delta_heading > HEADING_THRESHOLD){
        is_obvious_road = false;
    }
    
    if(is_obvious_road){
        if(osmMap_->ways()[current_way_id].isOneway()){
            label = ONEWAY_ROAD;
        }
        else{
            label = TWOWAY_ROAD;
        }
    }
    else{
        label = NON_OBVIOUS_ROAD;
    }
}

void QueryInitFeatureSelector::extractTrainingSamplesFromMap(float radius, OpenStreetMap *osmMap){
    if (osmMap->isEmpty()) {
        return;
    }
    
    osmMap_ = osmMap;
    
    if (trajectories_ == NULL) {
        return;
    }
    
    features_.clear();
    labels_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
   
    // Resample map with a point cloud using a grid
    osmMap_->updateMapSearchTree(0.5f * radius);
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
    
    // Compute Query Init features for each map sample point
    for (size_t pt_idx = 0; pt_idx < map_sample_points->size(); ++pt_idx) {
        PclPoint& pt = map_sample_points->at(pt_idx);
        
        query_init_sample_type new_feature;
        computeQueryInitFeatureAt(radius, pt, trajectories_, new_feature, pt.head);
        
        int label = 0; // 0: oneway; 1: twoway; 2: non-road
        computeLabelAt(radius, pt, label);
        
        // Add to features
        feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
        
        if (label == ONEWAY_ROAD) {
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::ONEWAY_COLOR));
        }
        else if(label == TWOWAY_ROAD){
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::TWOWAY_COLOR));
        }
        else{
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::NON_ROAD_COLOR));
        }
        
        features_.push_back(new_feature);
        labels_.push_back(label);
    }
}

bool QueryInitFeatureSelector::loadTrainingSamples(const string& filename){
    features_.clear();
    labels_.clear();
    feature_vertices_.clear();
    feature_colors_.clear();
    
    ifstream input;
    input.open(filename);
    if (input.fail()) {
        return false;
    }
   
    string str;
    getline(input, str);
    int n_features = stoi(str);
    
    for(int i = 0; i < n_features; ++i){
        getline(input, str);
        int label = stoi(str);
       
        getline(input, str);
        QString str1(str.c_str());
        QStringList list = str1.split(", ");
        
        float loc_x = stof(list.at(0).toStdString());
        float loc_y = stof(list.at(1).toStdString());
        
        getline(input, str);
        QString str2(str.c_str());
        list = str2.split(", ");
        
        query_init_sample_type new_feature;
        for (int j = 0; j < list.size(); ++j) {
            new_feature(j) = stof(list.at(j).toStdString());
        }
        
        // Add to features
        feature_vertices_.push_back(SceneConst::getInstance().normalize(loc_x, loc_y, Z_FEATURES));
        
        if (label == ONEWAY_ROAD) {
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::ONEWAY_COLOR));
        }
        else if(label == TWOWAY_ROAD){
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::TWOWAY_COLOR));
        }
        else{
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::NON_ROAD_COLOR));
        }
        
        features_.push_back(new_feature);
        labels_.push_back(label);
    }

    input.close();
    return true;
}

bool QueryInitFeatureSelector::saveTrainingSamples(const string& filename){
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
    output << map_sample_points->size() << endl;
    int n_rows = features_[0].nr();
    for (size_t i = 0; i < features_.size(); ++i) {
        output << labels_[i] << endl;
        PclPoint& pt = map_sample_points->at(i);
        output << pt.x << ", " << pt.y << ", " << pt.head << endl;
        for (int j = 0; j < n_rows-1; ++j) {
            output << features_[i](j) << ", ";
        }
        output << features_[i](n_rows-1) << endl;
        
    }
    output.close();
    return true;
}

bool QueryInitFeatureSelector::trainClassifier(){
    if(features_.size() == 0){
        cout << "Error! Empty training samples." << endl;
        return false;
    }
    if(features_.size() != labels_.size()){
        cout << "Error! Feature and label sizes do not match." << endl;
        return false;
    }
    
    trainQueryInitClassifier(features_,
                             labels_,
                             df_);
    
    decision_function_valid_ = true;
    
    return true;
}

bool QueryInitFeatureSelector::saveClassifier(const string& filename){
    if(!decision_function_valid_){
        cout << "Invalid decision function." << endl;
        return false;
    }
   
    dlib::serialize(filename.c_str()) << df_;
    
    return true;
}

void QueryInitFeatureSelector::draw(){
    QOpenGLBuffer position_buffer;
    position_buffer.create();
    position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    position_buffer.bind();
    position_buffer.allocate(&feature_vertices_[0], 3*feature_vertices_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    QOpenGLBuffer color_buffer;
    color_buffer.create();
    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    color_buffer.bind();
    color_buffer.allocate(&feature_colors_[0], 4*feature_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    glPointSize(30);
    glDrawArrays(GL_POINTS, 0, feature_vertices_.size());
}

void QueryInitFeatureSelector::clearVisibleFeatures(){
    feature_vertices_.clear();
    feature_colors_.clear();
}

void QueryInitFeatureSelector::clear(){
    trajectories_ = NULL;
    osmMap_ = NULL;
    
    features_.clear();
    labels_.clear();
}

QueryQFeatureSelector::QueryQFeatureSelector(QObject* parent, Trajectories* trajectories) : Renderable(parent) {
    trajectories_ = trajectories;
    osmMap_ = NULL;
    
    features_.clear();
    labels_.clear();
    
    decision_function_valid_ = false;
    
    feature_vertices_.clear();
    feature_colors_.clear();
}

QueryQFeatureSelector::~QueryQFeatureSelector(){
}

void QueryQFeatureSelector::setTrajectories(Trajectories* new_trajectories){
    trajectories_ = new_trajectories;
    feature_vertices_.clear();
    feature_colors_.clear();
}

void QueryQFeatureSelector::addFeatureAt(vector<float> &loc, QueryQLabel type){
    if(loc.size() != 4){
        return;
    }
    
    float SEARCH_RADIUS = 25.0f; // in meters
//    int FEATURE_DIMENSION = 40;
//    float ANGLE_BIN_RESOLUTION = 10.0f; // in degrees
//    float SPEED_BIN_RESOLUTION = 5.0f; // in m/s
    
    Eigen::Vector3d canonical_direction = Eigen::Vector3d(loc[2]-loc[0], loc[3]-loc[1], 0.0f);
    canonical_direction.normalize();
    
    // Compute feature
//    PclPointCloud::Ptr& gps_point_cloud = trajectories_->data();
    PclSearchTree::Ptr& gps_point_kdtree = trajectories_->tree();
    PclPoint search_point;
    search_point.setCoordinate(loc[2], loc[3], 0.0f);
    vector<int> k_indices;
    vector<float> k_distances;
    gps_point_kdtree->radiusSearch(search_point, SEARCH_RADIUS, k_indices, k_distances);
    
    query_q_sample_type new_feature;
    
//    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//        float heading = static_cast<float>(gps_point_cloud->at(*it).head) * PI / 180.0f;
//        float speed = gps_point_cloud->at(*it).speed / 100.0f;
//        Eigen::Vector3d pt_dir = Eigen::Vector3d(sin(heading), cos(heading), 0.0f);
//        float dot_value = canonical_direction.dot(pt_dir);
//        float delta_angle = acos(dot_value) * 180.0f / PI;
//        Eigen::Vector3d cross_direction = canonical_direction.cross(pt_dir);
//        if (cross_direction[2] < 0) {
//            delta_angle = 360 - delta_angle;
//        }
//        
//        int angle_bin = floor(delta_angle / ANGLE_BIN_RESOLUTION);
//        new_feature[angle_bin] += 1.0f;
//        int speed_bin = floor(speed / SPEED_BIN_RESOLUTION);
//        if(speed_bin > 3){
//            speed_bin = 3;
//        }
//        speed_bin += 36;
//        new_feature[speed_bin] += 1.0f;
//    }
//    
//    // Normalize the histogram
//    for(size_t i = 0; i < new_feature.size(); ++i){
//        new_feature[i] /= k_indices.size();
//    }
//    
//    printf("dist square: %.2f\n", k_distances[0]);
//    
//    features_.push_back(new_feature);
//    labels_.push_back(type);
//    
//    feature_vertices_.push_back(SceneConst::getInstance().normalize(loc[0], loc[1], Z_SELECTION));
//    feature_colors_.push_back(Color(0.0f, 0.0f, 1.0f, 1.0f));
//    feature_vertices_.push_back(SceneConst::getInstance().normalize(loc[2], loc[3], Z_SELECTION));
//    feature_colors_.push_back(Color(0.0f, 1.0f, 1.0f, 1.0f));
}

bool QueryQFeatureSelector::save(const string& filename){
//    ofstream output;
//    output.open(filename);
//    if (output.fail()) {
//        return false;
//    }
//    
//    for (size_t i = 0; i < features_.size(); ++i) {
//        output << labels_[i] << endl;
//        for (size_t j = 0; j < features_[i].size()-1; ++j) {
//            output << features_[i][j] << ", ";
//        }
//        output << features_[i].back() << endl;
//    }
//    output.close();
    return true;
}

void QueryQFeatureSelector::draw(){
//    QOpenGLBuffer position_buffer;
//    position_buffer.create();
//    position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
//    position_buffer.bind();
//    position_buffer.allocate(&feature_vertices_[0], 3*feature_vertices_.size()*sizeof(float));
//    shadder_program_->setupPositionAttributes();
//    
//    QOpenGLBuffer color_buffer;
//    color_buffer.create();
//    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
//    color_buffer.bind();
//    color_buffer.allocate(&feature_colors_[0], 4*feature_colors_.size()*sizeof(float));
//    shadder_program_->setupColorAttributes();
//    glPointSize(20);
////    glLineWidth(5);
////    glDrawArrays(GL_LINES, 0, feature_vertices_.size());
//    glDrawArrays(GL_POINTS, 0, feature_vertices_.size());
}

void QueryQFeatureSelector::extractTrainingSamplesFromMap(float radius, OpenStreetMap *osmMap){
    if (osmMap->isEmpty()) {
        return;
    }
    
    osmMap_ = osmMap;
    
    if (trajectories_ == NULL) {
        return;
    }
    
    features_.clear();
    labels_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
    
    // Resample map with a point cloud using a grid
    osmMap_->updateMapSearchTree(0.5f * radius);
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
    
    // Compute Query Init features for each map sample point
    for (size_t pt_idx = 0; pt_idx < map_sample_points->size(); ++pt_idx) {
        PclPoint& pt = map_sample_points->at(pt_idx);
        int way_id = pt.id_trajectory;
        
        query_q_sample_type new_feature;
        computeQueryQFeatureAt(radius,
                               pt,
                               trajectories_,
                               new_feature, pt.head,
                               osmMap_->ways()[way_id].isOneway());
        
        int label = 0; // 0: oneway; 1: twoway; 2: non-road
//        computeLabelAt(radius, pt, label);
        
        // Add to features
        feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
        
        if (label == ONEWAY_ROAD) {
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::ONEWAY_COLOR));
        }
        else if(label == TWOWAY_ROAD){
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::TWOWAY_COLOR));
        }
        else{
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::NON_ROAD_COLOR));
        }
        
        features_.push_back(new_feature);
        labels_.push_back(label);
    }
}

void QueryQFeatureSelector::computeQueryQLabelAt(PclPoint& point, QueryQLabel& label, bool is_reverse_dir){
//    if (osmMap_ == NULL) {
//        return;
//    }
//    
//    float SEARCH_RADIUS = 25.0f;
//    
//    int way_id = point.id_trajectory;
//    
//    OsmWay& the_way = osmMap_->ways()[way_id];
//    set<int> nearby_junctions;
//    set<int> related_way_ids;
//    for(int i = 0; i < the_way.node_ids().size(); ++i){
//        int node_id = the_way.node_ids()[i];
//        OsmNode& a_node = osmMap_->nodes()[node_id];
//        if(a_node.degree() <= 2){
//            continue;
//        }
//        
//        for (int j = 0; j < a_node.way_ids().size(); ++j) {
//            related_way_ids.insert(a_node.way_ids()[j]);
//        }
//    }
//    
//    float heading_in_radius = point.head / 180.0f * PI;
//    Eigen::Vector2d dir(cos(heading_in_radius), sin(heading_in_radius));
//    
//    vector<int> k_indices;
//    vector<float> k_dists;
//    osmMap_->map_search_tree()->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dists);
//    
//    label = R_GROW;
//    
//    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//        PclPoint& map_pt = osmMap_->map_point_cloud()->at(*it);
//        
//        Eigen::Vector2d vec(map_pt.x - point.x, map_pt.y - point.y);
//        float dot_value = vec.dot(dir);
//        bool to_be_considered = false;
//        if (is_reverse_dir) {
//            if (dot_value <= -1.0) {
//                to_be_considered = true;
//            }
//        }
//        else{
//            if (dot_value >= 1.0) {
//                to_be_considered = true;
//            }
//        }
//        if (to_be_considered) {
//            if(map_pt.id_trajectory != way_id){
//                if(related_way_ids.find(map_pt.id_trajectory) != related_way_ids.end()){
//                    label = R_BRANCH;
//                }
//            }
//        }
//    }
}

void QueryQFeatureSelector::clearVisibleFeatures(){
    feature_vertices_.clear();
    feature_colors_.clear();
}

void QueryQFeatureSelector::clear(){
    trajectories_ = NULL;
    osmMap_ = NULL;
    
    features_.clear();
    labels_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
}
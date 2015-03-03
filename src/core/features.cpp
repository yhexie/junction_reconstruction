//
//  features.cpp
//  junction_reconstruction
//
//  Created by Chen Chen on 1/8/15.
//
//

#include "features.h"
#include <fstream>

void computeQueryInitFeatureAt(float radius, PclPoint& point, Trajectories* trajectories, vector<float>& feature, float canonical_heading){
    feature.clear();
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    // Heading histogram parameters
    int N_HEADING_BINS = 36;
    float DELTA_HEADING_BIN = 360.0f / N_HEADING_BINS;
    float ANGLE_THETA = 7.5f; // in degrees, this is the heading standard derivation
    int window_size = ceil(ANGLE_THETA / DELTA_HEADING_BIN);
    
    // Speed histogram
    int N_SPEED_BINS = 3;
    float MAX_LOW_SPEED = 5.0f; // meter per second
    float MAX_MID_SPEED = 15.0f; // meter per second
    
    // Point density histogram parameters
    float SEARCH_RADIUS = radius; // in meters
    
    int N_TOTAL_BINS = N_HEADING_BINS * N_SPEED_BINS;
    
    vector<int> k_indices;
    vector<float> k_dists;
    tree->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dists);
    
    feature.clear();
    feature.resize(N_TOTAL_BINS, 0.0f);
    
    // Compute feature
    for (size_t i = 0; i < k_indices.size(); ++i) {
        float pt_heading = data->at(k_indices[i]).head;
        float speed = data->at(k_indices[i]).speed;
        speed /= 100.0f;
        
        float relative_heading = deltaHeading1MinusHeading2(pt_heading, canonical_heading);
        if (relative_heading < 0) {
            relative_heading += 360.0f;
        }
        
        int angle_bin_idx = floor(relative_heading / DELTA_HEADING_BIN);
        int speed_bin_idx = 0;
        if (speed > MAX_LOW_SPEED) {
            speed_bin_idx = 1;
        }
        if (speed > MAX_MID_SPEED) {
            speed_bin_idx = 2;
        }
        
        for (int s = -1 * window_size; s < window_size; ++s) {
            int bin_idx = angle_bin_idx + s;
            if (bin_idx < 0) {
                bin_idx += N_HEADING_BINS;
            }
            if (bin_idx >= N_HEADING_BINS) {
                bin_idx %= N_HEADING_BINS;
            }
            float bin_center = (bin_idx + 0.5) * DELTA_HEADING_BIN;
            float delta_angle = abs(bin_center - relative_heading);
            if (delta_angle > 180) {
                delta_angle = 360 - delta_angle;
            }
            
            int final_bin_idx = bin_idx * speed_bin_idx;
            feature[final_bin_idx] += exp(-1.0f * delta_angle * delta_angle / 2.0f / ANGLE_THETA / ANGLE_THETA) / sqrt(2 * PI);
        }
    }
    
    // Normalization
    float sum = 0.0f;
    for (size_t i = 0; i < feature.size(); ++i) {
        sum += feature[i];
    }
    if (sum > 1e-3) {
        for (size_t i = 0; i < feature.size(); ++i) {
            feature[i] /= sum;
        }
    }
}

void computeQueryQFeatureAt(PclPoint& point, Trajectories* trajectories, vector<float>& feature, float heading, bool is_oneway, bool is_reverse_dir){
    // Feature parameters
    float SEARCH_RADIUS = 15.0f; // in radius
    float ANGLE_THRESHOLD = 10.0f;
    float MAX_DIST_TO_ROAD_CENTER = 10.0f; // in meters
    
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    vector<int> k_indices;
    vector<float> k_dists;
    PclPoint search_point;
    search_point.setCoordinate(point.x, point.y, 0.0f);
    tree->radiusSearch(search_point, SEARCH_RADIUS, k_indices, k_dists);
    
    float growing_heading = heading;
    if (is_reverse_dir) {
        growing_heading += 180.0f;
        if (growing_heading > 360.0f) {
            growing_heading -= 360.0f;
        }
    }
    
    float growing_heading_in_radius = growing_heading / 180.0f * PI;
    Eigen::Vector2d dir(cos(growing_heading_in_radius), sin(growing_heading_in_radius)); // growing direction
    
    //    feature_vertices_.push_back(SceneConst::getInstance().normalize(search_point.x, search_point.y, Z_FEATURES));
    //    feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
    
    // Filter nearby point, and only leave the compatible points
    vector<int> compatible_points;
    set<int>    allowable_trajs;
    map<int, float> allowable_traj_ts;
    float OVERSHOOT = 1.0f;
    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
        PclPoint& pt = trajectories->data()->at(*it);
        Eigen::Vector2d vec(pt.x - point.x, pt.y - point.y);
        float dot_product = vec.dot(dir);
        
        if (dot_product > OVERSHOOT) {
            continue;
        }
        
        float length = vec.norm();
        float dist_to_road = sqrt(length*length - dot_product*dot_product);
        
        if (dist_to_road > MAX_DIST_TO_ROAD_CENTER) {
            continue;
        }
        
        // Check angle
        float pt_heading = pt.head;
        
        float delta_angle = abs(pt_heading - heading);
        if(delta_angle > 180.0f){
            delta_angle = 360.0f - delta_angle;
        }
        
        bool angle_is_acceptable = true;
        if (is_oneway) {
            if (delta_angle > ANGLE_THRESHOLD) {
                angle_is_acceptable = false;
            }
        }
        else{
            if (delta_angle > 90.0f) {
                delta_angle = 180.0f - delta_angle;
            }
            if (delta_angle > ANGLE_THRESHOLD) {
                angle_is_acceptable = false;
            }
        }
        
        if (angle_is_acceptable) {
            compatible_points.push_back(*it);
            //            feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
            //            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
            allowable_trajs.insert(pt.id_trajectory);
            allowable_traj_ts[pt.id_trajectory] = pt.t;
        }
    }
    
    //    printf("compatible points: %lu\n", compatible_points.size());
    
    // Compute features
    vector<int> k_indices1;
    vector<float> k_dists1;
    vector<int> points_to_consider;
    float MAX_DELTA_T = 100.0f;
    tree->radiusSearch(search_point, 6 * SEARCH_RADIUS, k_indices1, k_dists1);
    for (vector<int>::iterator it=k_indices1.begin(); it != k_indices1.end(); ++it) {
        int id_traj = data->at(*it).id_trajectory;
        const bool is_allowable = allowable_trajs.find(id_traj) != allowable_trajs.end();
        if (is_allowable) {
            float traj_t = allowable_traj_ts.find(id_traj)->second;
            
            float delta_t = abs(traj_t - data->at(*it).t);
            if (delta_t > MAX_DELTA_T) {
                continue;
            }
            
            PclPoint& pt = trajectories->data()->at(*it);
            Eigen::Vector2d vec(pt.x - point.x, pt.y - point.y);
            float dot_product = vec.dot(dir);
            
            if (dot_product < OVERSHOOT) {
                continue;
            }
            points_to_consider.push_back(*it);
            //            feature_vertices_.push_back(SceneConst::getInstance().normalize(data->at(*it).x, data->at(*it).y, Z_FEATURES));
            //            Color orange = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
            //            Color this_color(orange.r*weight, orange.g*weight, orange.b*weight, 1.0f);
            //            feature_colors_.push_back(this_color);
        }
    }
    
    // Compute features
    float DELTA_HEADING = 15.0f; // in degrees
    int N_HEADING_BIN = ceil(360.0f / DELTA_HEADING);
    DELTA_HEADING = 360.0f / N_HEADING_BIN;
    float DELTA_R = 25.0f;
    int N_R_BIN = 4;
    int N_TOTAL_BIN = N_HEADING_BIN * N_R_BIN + 1;
    
    feature.clear();
    feature.resize(N_TOTAL_BIN, 0.0f);
    float heading_in_radius = heading / 180.0f * PI;
    Eigen::Vector3d heading_dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
    for (vector<int>::iterator it = points_to_consider.begin(); it != points_to_consider.end(); ++it) {
        PclPoint& pt = data->at(*it);
        
        float traj_t = allowable_traj_ts.find(pt.id_trajectory)->second;
        float delta_t = abs(traj_t - data->at(*it).t);
        float weight = exp(-2.0f * delta_t * delta_t / MAX_DELTA_T / MAX_DELTA_T);
        
        Eigen::Vector3d vec(pt.x - point.x, pt.y - point.y, 0.0f);
        float r_dist = vec.norm();
        vec.normalize();
        
        float dot_value = vec.dot(heading_dir);
        if(dot_value > 1.0f){
            dot_value = 1.0f;
        }
        if (dot_value < -1.0f) {
            dot_value = -1.0f;
        }
        
        float cross_value = heading_dir.cross(heading_dir)[2];
        float angle = acos(dot_value) * 180.0f / PI;
        if (cross_value < 0) {
            angle = 360.0f - angle;
        }
        
        int r_bin_idx = floor(r_dist / DELTA_R);
        if(r_bin_idx >= N_R_BIN){
            r_bin_idx = N_R_BIN - 1;
        }
        
        int heading_bin_idx = floor(angle / DELTA_HEADING);
        
        int feature_bin_idx = heading_bin_idx + r_bin_idx * N_HEADING_BIN;
        feature[feature_bin_idx] += weight;
    }
    
    //Normalize heading feature
    float sum = 0.0f;
    for (int i = 0; i < N_TOTAL_BIN - 1; ++i) {
        sum += feature[i];
    }
    if (sum > 0.1f) {
        for (int i = 0; i < N_TOTAL_BIN - 1; ++i) {
            feature[i] /= sum;
        }
    }
    if (is_oneway) {
        feature[N_TOTAL_BIN-1] = 0.0f; // oneway
    }
    else{
        feature[N_TOTAL_BIN-1] = 1.0f; // twoway
    }
}

void computeQueryQFeatureAt(PclPoint& point, Trajectories* trajectories, vector<float>& feature, set<int>& allowable_trajs, map<int, float>& allowable_traj_min_ts, map<int, float>& allowable_traj_max_ts, vector<int>& points_to_consider, float heading, bool is_oneway, bool is_reverse_dir){
    float SEARCH_RADIUS = 50.0f;
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    // Compute features
    vector<int> k_indices;
    vector<float> k_dists;
    //vector<int> points_to_consider;
    float MAX_DELTA_T = 100.0f;
    tree->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dists);
   
    float heading_in_radius = heading / 180.0f * PI;
    Eigen::Vector2f dir(cos(heading_in_radius), sin(heading_in_radius));
    if (is_reverse_dir) {
        dir *= -1.0f;
    }
    
    // Compute features
    float DELTA_HEADING = 15.0f; // in degrees
    int N_HEADING_BIN = ceil(360.0f / DELTA_HEADING);
    DELTA_HEADING = 360.0f / N_HEADING_BIN;
    float DELTA_R = 10.0f;
    int N_R_BIN = ceil(SEARCH_RADIUS / DELTA_R);
    int N_TOTAL_BIN = N_HEADING_BIN * N_R_BIN + 1;
    
    feature.clear();
    feature.resize(N_TOTAL_BIN, 0.0f);
    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
        int id_traj = data->at(*it).id_trajectory;
        const bool is_allowable = allowable_trajs.find(id_traj) != allowable_trajs.end();
        if (is_allowable) {
            float d_t_1 = abs(allowable_traj_max_ts[id_traj] - data->at(*it).t);
            float d_t_2 = abs(allowable_traj_min_ts[id_traj] - data->at(*it).t);
            
            float delta_t = (d_t_1 < d_t_2) ? d_t_1 : d_t_2;
            
            if (delta_t > MAX_DELTA_T) {
                continue;
            }
            
            Eigen::Vector2f vec(data->at(*it).x - point.x,
                                data->at(*it).y - point.y);
            float dot_value = vec.dot(dir);
            if (dot_value < 0.5f) {
                continue;
            }
            
            float weight = exp(-2.0f * delta_t * delta_t / MAX_DELTA_T / MAX_DELTA_T);
            points_to_consider.push_back(*it);
            
            Eigen::Vector3f vec1(vec.x(), vec.y(), 0.0f);
            Eigen::Vector3f heading_dir(dir.x(), dir.y(), 0.0f);
            float r_dist = vec.norm();
            vec.normalize();
            
            if(dot_value > 1.0f){
                dot_value = 1.0f;
            }
            if (dot_value < -1.0f) {
                dot_value = -1.0f;
            }
            
            float cross_value = heading_dir.cross(vec1)[2];
            float angle = acos(dot_value) * 180.0f / PI;
            if (cross_value < 0) {
                angle = 360.0f - angle;
            }
            
            int r_bin_idx = floor(r_dist / DELTA_R);
            if(r_bin_idx >= N_R_BIN){
                r_bin_idx = N_R_BIN - 1;
            }
            
            int heading_bin_idx = floor(angle / DELTA_HEADING);
            
            int feature_bin_idx = heading_bin_idx + r_bin_idx * N_HEADING_BIN;
            feature[feature_bin_idx] += weight;
            
        }
    }
    
    //Normalize heading feature
    float sum = 0.0f;
    for (int i = 0; i < N_TOTAL_BIN - 1; ++i) {
        sum += feature[i];
    }
    if (sum > 0.1f) {
        for (int i = 0; i < N_TOTAL_BIN - 1; ++i) {
            feature[i] /= sum;
        }
    }
    if (is_oneway) {
        feature[N_TOTAL_BIN-1] = 0.0f; // oneway
    }
    else{
        feature[N_TOTAL_BIN-1] = 1.0f; // twoway
    }
}

QueryInitFeatureSelector::QueryInitFeatureSelector(QObject* parent, Trajectories* trajectories) : Renderable(parent) {
    trajectories_ = trajectories;
    osmMap_ = NULL;
    features_.clear();
    labels_.clear();
    feature_vertices_.clear();
    feature_colors_.clear();
    n_yes_ = 0;
}

QueryInitFeatureSelector::~QueryInitFeatureSelector(){
}

void QueryInitFeatureSelector::setTrajectories(Trajectories* new_trajectories){
    trajectories_ = new_trajectories;
}

void QueryInitFeatureSelector::computeQueryInitLabelAt(float radius, PclPoint& pt, int& label){
    float SEARCH_RADIUS = radius; // in meter
    float HEADING_THRESHOLD = 20.0f; // in degrees
    
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
    float maximum_delta_heading = 0;
    for (size_t i = 0; i < map_k_indices.size(); ++i){
        PclPoint& nearby_pt = map_sample_points->at(map_k_indices[i]);
        if (nearby_pt.id_trajectory == current_way_id) {
            continue;
        }
        
        if(osmMap_->twoWaysEquivalent(current_way_id, nearby_pt.id_trajectory)){
            continue;
        }
        
        if(!nearby_ways.emplace(nearby_pt.id_trajectory).second){
            continue;
        }
        
        // Check heading
        float delta_heading = abs(deltaHeading1MinusHeading2(pt.head, nearby_pt.head));
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
        label = NON_OBVIOUS_ROAD;
//        if(osmMap_->ways()[current_way_id].isOneway()){
//            label = 0.0f;
//        }
//        else{
//            label = 1.0f;
//        }
    }
    else{
        label = NON_OBVIOUS_ROAD;
    }
}

void QueryInitFeatureSelector::computeQueryInitFeaturesFromMap(float radius, OpenStreetMap *osmMap){
    if (osmMap->isEmpty()) {
        return;
    }
    
    osmMap_ = osmMap;
    
    if (trajectories_ == NULL) {
        return;
    }
    
    features_.clear();
    labels_.clear();
    n_yes_ = 0;
    feature_vertices_.clear();
    feature_colors_.clear();
   
    // Resample map with a point cloud using a grid
    osmMap_->updateMapSearchTree(radius);
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
    
    // Compute Query Init features for each map search point
    for (size_t pt_idx = 0; pt_idx < map_sample_points->size(); ++pt_idx) {
        PclPoint& pt = map_sample_points->at(pt_idx);
        
        vector<float> new_feature;
        computeQueryInitFeatureAt(radius, pt, trajectories_, new_feature, pt.head);
        
        int label = 1; // 0.0: oneway; 1.0: twoway; 2.0: non-road
        computeQueryInitLabelAt(radius, pt, label);
        
        // Add to features
        feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
        
        if (label == NON_OBVIOUS_ROAD) {
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
        }
//        else if(label == 1.0f){
//            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
//        }
        else{
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
        }
        
        features_.push_back(new_feature);
        labels_.push_back(label);
    }
}

bool QueryInitFeatureSelector::save(const string& filename){
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    for (size_t i = 0; i < features_.size(); ++i) {
        output << labels_[i] << endl;
        for (size_t j = 0; j < features_[i].size()-1; ++j) {
            output << features_[i][j] << ", ";
        }
        output << features_[i].back() << endl;
    }
    output.close();
    return true;
}

bool QueryInitFeatureSelector::exportFeatures(float radius, const string& filename){
    features_.clear();
    feature_vertices_.clear();
    feature_colors_.clear();
    
    vector<float> xs;
    vector<float> ys;
    
    // Compute features for each sample points
    PclPointCloud::Ptr& samples = trajectories_->samples();
    features_.clear();
    for (size_t i = 0; i < samples->size(); ++i) {
        PclPoint& pt = samples->at(i);
        vector<float> new_feature;
        computeQueryInitFeatureAt(radius, pt, trajectories_, new_feature, trajectories_->samples()->at(i).head);
        features_.push_back(new_feature);
        feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
        feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
    }
    
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    for (size_t i = 0; i < features_.size(); ++i) {
        for (size_t j = 0; j < features_[i].size()-1; ++j) {
            output << features_[i][j] << ", ";
        }
        output << features_[i].back() << endl;
    }
    output.close();
    return true;
}

bool QueryInitFeatureSelector::loadPrediction(const string& filename){
    ifstream fin(filename);
    if (!fin.good())
        return false;
    labels_.clear();
   
    int n_samples = 0;
    fin >> n_samples;
    for (size_t i = 0; i < n_samples; ++i) {
        float label;
        fin >> label;
        labels_.push_back(label);
        if (label == 0.0f) {
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::RED);
        }
        else if (label == 1.0f){
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::GREEN);
        }
        else{
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
        }
    }
    
    if (labels_.size() != features_.size()) {
        printf("WARNING: labels do not match with features!\n");
    }
    
    fin.close();
    
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
    features_.clear();
    labels_.clear();
    n_yes_ = 0;
}

QueryQFeatureSelector::QueryQFeatureSelector(QObject* parent, Trajectories* trajectories) : Renderable(parent) {
    trajectories_ = trajectories;
    tmp_ = 0;
}

QueryQFeatureSelector::~QueryQFeatureSelector(){
    trajectories_ = NULL;
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
    int FEATURE_DIMENSION = 40;
    float ANGLE_BIN_RESOLUTION = 10.0f; // in degrees
    float SPEED_BIN_RESOLUTION = 5.0f; // in m/s
    
    Eigen::Vector3d canonical_direction = Eigen::Vector3d(loc[2]-loc[0], loc[3]-loc[1], 0.0f);
    canonical_direction.normalize();
    
    // Compute feature
    PclPointCloud::Ptr& gps_point_cloud = trajectories_->data();
    PclSearchTree::Ptr& gps_point_kdtree = trajectories_->tree();
    PclPoint search_point;
    search_point.setCoordinate(loc[2], loc[3], 0.0f);
    vector<int> k_indices;
    vector<float> k_distances;
    gps_point_kdtree->radiusSearch(search_point, SEARCH_RADIUS, k_indices, k_distances);
    
    vector<float> new_feature(FEATURE_DIMENSION, 0.0f);
    
    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
        float heading = static_cast<float>(gps_point_cloud->at(*it).head) * PI / 180.0f;
        float speed = gps_point_cloud->at(*it).speed / 100.0f;
        Eigen::Vector3d pt_dir = Eigen::Vector3d(sin(heading), cos(heading), 0.0f);
        float dot_value = canonical_direction.dot(pt_dir);
        float delta_angle = acos(dot_value) * 180.0f / PI;
        Eigen::Vector3d cross_direction = canonical_direction.cross(pt_dir);
        if (cross_direction[2] < 0) {
            delta_angle = 360 - delta_angle;
        }
        
        int angle_bin = floor(delta_angle / ANGLE_BIN_RESOLUTION);
        new_feature[angle_bin] += 1.0f;
        int speed_bin = floor(speed / SPEED_BIN_RESOLUTION);
        if(speed_bin > 3){
            speed_bin = 3;
        }
        speed_bin += 36;
        new_feature[speed_bin] += 1.0f;
    }
    
    // Normalize the histogram
    for(size_t i = 0; i < new_feature.size(); ++i){
        new_feature[i] /= k_indices.size();
    }
    
    printf("dist square: %.2f\n", k_distances[0]);
    
    features_.push_back(new_feature);
    labels_.push_back(type);
    
    feature_vertices_.push_back(SceneConst::getInstance().normalize(loc[0], loc[1], Z_SELECTION));
    feature_colors_.push_back(Color(0.0f, 0.0f, 1.0f, 1.0f));
    feature_vertices_.push_back(SceneConst::getInstance().normalize(loc[2], loc[3], Z_SELECTION));
    feature_colors_.push_back(Color(0.0f, 1.0f, 1.0f, 1.0f));
}

bool QueryQFeatureSelector::save(const string& filename){
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    for (size_t i = 0; i < features_.size(); ++i) {
        output << labels_[i] << endl;
        for (size_t j = 0; j < features_[i].size()-1; ++j) {
            output << features_[i][j] << ", ";
        }
        output << features_[i].back() << endl;
    }
    output.close();
    return true;
}

bool QueryQFeatureSelector::exportFeatures(const string& filename){
    features_.clear();
    feature_vertices_.clear();
    feature_colors_.clear();
    
    vector<float> xs;
    vector<float> ys;
    PclPointCloud::Ptr& samples = trajectories_->samples();
    
    for (size_t i = 0; i < samples->size(); ++i) {
        PclPoint& pt = samples->at(i);
        
        vector<float> new_feature;
        computeQueryQFeatureAt(pt, trajectories_, new_feature, trajectories_->samples()->at(i).head);
        features_.push_back(new_feature);
        feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
        feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
    }
    
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    for (size_t i = 0; i < features_.size(); ++i) {
        for (size_t j = 0; j < features_[i].size()-1; ++j) {
            output << features_[i][j] << ", ";
        }
        output << features_[i].back() << endl;
    }
    output.close();
    return true;
}

bool QueryQFeatureSelector::loadPrediction(const string& filename){
    ifstream fin(filename);
    if (!fin.good())
        return false;
    labels_.clear();
    
    int n_samples = 0;
    fin >> n_samples;
    for (size_t i = 0; i < n_samples; ++i) {
        int label;
        fin >> label;
        if (label == 0) {
            labels_.push_back(R_GROW);
        }
        else{
            labels_.push_back(R_BRANCH);
        }
        
        if (label == R_GROW) {
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::RED);
        }
        else{
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
        }
    }
    
    if (labels_.size() != features_.size()) {
        printf("WARNING: labels do not match with features!\n");
    }
    
    fin.close();
    
    return true;
}

void QueryQFeatureSelector::draw(){
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
    glPointSize(20);
//    glLineWidth(5);
//    glDrawArrays(GL_LINES, 0, feature_vertices_.size());
    glDrawArrays(GL_POINTS, 0, feature_vertices_.size());
}



void QueryQFeatureSelector::computeQueryQFeaturesFromMap(OpenStreetMap *osmMap){
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
    osmMap_->updateMapSearchTree(25.0f);
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
   
//    int i = tmp_;
    vector<float> feature_headings;
    vector<float> feature_loc_xs;
    vector<float> feature_loc_ys;
    float SEARCH_RADIUS = 10.0f;
    float HEADING_THRESHOLD = 7.5f;
    for (int i = 0; i < map_sample_points->size(); ++i) {
        //Compute ?Q Feature at map sample location
        if (i % 1000 == 0) {
            printf("Now at %d.\n", i);
        }
    
        PclPoint& point = map_sample_points->at(i);
//    feature_vertices_.clear();
//    feature_colors_.clear();
//    feature_vertices_.push_back(SceneConst::getInstance().normalize(point.x, point.y, Z_FEATURES));
//    feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
    
        bool is_oneway = osmMap_->ways()[point.id_trajectory].isOneway();
        float forward_heading = point.head;
        
        // Forward direction
        vector<float> forward_feature;
        set<int> allowable_trajs;
        map<int, float> allowable_traj_min_ts;
        map<int, float> allowable_traj_max_ts;
        vector<int> k_indices;
        vector<float> k_dists;
        float heading_in_radius = point.head / 180.0f * PI;
        Eigen::Vector2f dir(cos(heading_in_radius), sin(heading_in_radius));
        PclPoint search_pt;
        search_pt.setCoordinate(point.x - SEARCH_RADIUS * dir.x(), point.y - SEARCH_RADIUS * dir.y(), 0.0f);
        trajectories_->tree()->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dists);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& data_point = trajectories_->data()->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(data_point.head, point.head));
            if (!is_oneway && delta_heading > 90.0f) {
                delta_heading = 180.0f - delta_heading;
            }
            if (delta_heading < HEADING_THRESHOLD) {
                if (allowable_trajs.emplace(data_point.id_trajectory).second) {
                    allowable_traj_min_ts[data_point.id_trajectory] = data_point.t;
                    allowable_traj_max_ts[data_point.id_trajectory] = data_point.t;
                }
                else{
                    if (allowable_traj_max_ts[data_point.id_trajectory] < data_point.t) {
                        allowable_traj_max_ts[data_point.id_trajectory] = data_point.t;
                    }
                    if (allowable_traj_min_ts[data_point.id_trajectory] > data_point.t) {
                        allowable_traj_min_ts[data_point.id_trajectory] = data_point.t;
                    }
                }
            }
        }
    
        vector<int> related_points;
        computeQueryQFeatureAt(point, trajectories_, forward_feature, allowable_trajs, allowable_traj_min_ts, allowable_traj_max_ts, related_points, forward_heading, is_oneway);
    
//    for (vector<int>::iterator it = related_points.begin(); it != related_points.end(); ++it) {
//        feature_vertices_.push_back(SceneConst::getInstance().normalize(trajectories_->data()->at(*it).x, trajectories_->data()->at(*it).y, Z_FEATURES));
//        Color orange = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
//        Color this_color(orange.r, orange.g, orange.b, 1.0f);
//        feature_colors_.push_back(this_color);
//    }
//    tmp_++;
    
        QueryQLabel label;
        computeQueryQLabelAt(point, label);
        features_.push_back(forward_feature);
        labels_.push_back(label);
        feature_headings.push_back(forward_heading);
        feature_loc_xs.push_back(point.x);
        feature_loc_ys.push_back(point.y);
        
        // Backward direction
        vector<float> backward_feature;
        allowable_trajs.clear();
        allowable_traj_min_ts.clear();
        allowable_traj_max_ts.clear();
        k_indices.clear();
        k_dists.clear();
        search_pt.setCoordinate(point.x + SEARCH_RADIUS * dir.x(), point.y + SEARCH_RADIUS * dir.y(), 0.0f);
        trajectories_->tree()->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dists);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& data_point = trajectories_->data()->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(data_point.head, point.head));
            if (!is_oneway && delta_heading > 90.0f) {
                delta_heading = 180.0f - delta_heading;
            }
            if (delta_heading < HEADING_THRESHOLD) {
                if (allowable_trajs.emplace(data_point.id_trajectory).second) {
                    allowable_traj_min_ts[data_point.id_trajectory] = data_point.t;
                    allowable_traj_max_ts[data_point.id_trajectory] = data_point.t;
                }
                else{
                    if (allowable_traj_max_ts[data_point.id_trajectory] < data_point.t) {
                        allowable_traj_max_ts[data_point.id_trajectory] = data_point.t;
                    }
                    if (allowable_traj_min_ts[data_point.id_trajectory] > data_point.t) {
                        allowable_traj_min_ts[data_point.id_trajectory] = data_point.t;
                    }
                }
            }
        }
        
        related_points.clear();
        computeQueryQFeatureAt(point, trajectories_, backward_feature, allowable_trajs, allowable_traj_min_ts, allowable_traj_max_ts, related_points, forward_heading, is_oneway, true);
        
        //    for (vector<int>::iterator it = related_points.begin(); it != related_points.end(); ++it) {
        //        feature_vertices_.push_back(SceneConst::getInstance().normalize(trajectories_->data()->at(*it).x, trajectories_->data()->at(*it).y, Z_FEATURES));
        //        Color orange = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
        //        Color this_color(orange.r, orange.g, orange.b, 1.0f);
        //        feature_colors_.push_back(this_color);
        //    }
        //    tmp_++;
        
        computeQueryQLabelAt(point, label, true);
        features_.push_back(backward_feature);
        labels_.push_back(label);
        feature_headings.push_back(forward_heading);
        feature_loc_xs.push_back(point.x);
        feature_loc_ys.push_back(point.y);
    }
    
    // Update drawing
//    tmp_++;
    float ARROW_LENGTH = 2.5f; // in meters
    for (int i = 0; i < features_.size(); ++i) {
        float heading_in_radius = feature_headings[i] / 180.0f * PI;
        feature_vertices_.push_back(SceneConst::getInstance().normalize(feature_loc_xs[i], feature_loc_ys[i], Z_FEATURES));
        float dx = ARROW_LENGTH * cos(heading_in_radius);
        float dy = ARROW_LENGTH * sin(heading_in_radius);
        feature_vertices_.push_back(SceneConst::getInstance().normalize(feature_loc_xs[i]+dx, feature_loc_ys[i]+dy, Z_FEATURES));
        
        switch (labels_[i]) {
            case R_GROW:
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
                break;
            case R_BRANCH:
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
                break;
            default:
                break;
        }
    }
}

void QueryQFeatureSelector::computeQueryQLabelAt(PclPoint& point, QueryQLabel& label, bool is_reverse_dir){
    if (osmMap_ == NULL) {
        return;
    }
    
    float SEARCH_RADIUS = 25.0f;
    
    int way_id = point.id_trajectory;
    
    OsmWay& the_way = osmMap_->ways()[way_id];
    set<int> nearby_junctions;
    set<int> related_way_ids;
    for(int i = 0; i < the_way.node_ids().size(); ++i){
        int node_id = the_way.node_ids()[i];
        OsmNode& a_node = osmMap_->nodes()[node_id];
        if(a_node.degree() <= 2){
            continue;
        }
        
        for (int j = 0; j < a_node.way_ids().size(); ++j) {
            related_way_ids.insert(a_node.way_ids()[j]);
        }
    }
    
    float heading_in_radius = point.head / 180.0f * PI;
    Eigen::Vector2d dir(cos(heading_in_radius), sin(heading_in_radius));
    
    vector<int> k_indices;
    vector<float> k_dists;
    osmMap_->map_search_tree()->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dists);
    
    label = R_GROW;
    
    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
        PclPoint& map_pt = osmMap_->map_point_cloud()->at(*it);
        
        Eigen::Vector2d vec(map_pt.x - point.x, map_pt.y - point.y);
        float dot_value = vec.dot(dir);
        bool to_be_considered = false;
        if (is_reverse_dir) {
            if (dot_value <= -1.0) {
                to_be_considered = true;
            }
        }
        else{
            if (dot_value >= 1.0) {
                to_be_considered = true;
            }
        }
        if (to_be_considered) {
            if(map_pt.id_trajectory != way_id){
                if(related_way_ids.find(map_pt.id_trajectory) != related_way_ids.end()){
                    label = R_BRANCH;
                }
            }
        }
    }
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
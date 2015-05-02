//
//  features.cpp
//  junction_reconstruction
//
//  Created by Chen Chen on 1/8/15.
//
//

#include "features.h"
#include <fstream>
#include <pcl/search/impl/flann_search.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/topological_sort.hpp>

using namespace boost;

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
        if (heading_bin_idx == N_HEADING_BINS){
            heading_bin_idx = N_HEADING_BINS - 1;
        }
        
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
    
    dlib::randomize_samples(samples, labels);
    
    query_init_ovo_trainer trainer;
    dlib::krr_trainer<query_init_rbf_kernel> rbf_trainer;
    
    // Cross Validation for choosing gamma
//    for(double gamma = 0.05f; gamma < 0.4f; gamma += 0.05f){
//        cout << "gamma = " << gamma << endl;
//        rbf_trainer.set_kernel(query_init_rbf_kernel(gamma));
//        
//        trainer.set_trainer(rbf_trainer);
//        
//        cout << "\tCross validation: \n" << dlib::cross_validate_multiclass_trainer(trainer, samples, labels, 3) << endl;
//    }
    
    double gamma = 0.15f;
    
    rbf_trainer.set_kernel(query_init_rbf_kernel(gamma));
    trainer.set_trainer(rbf_trainer);
    
    df = trainer.train(samples, labels);
}

void computePointsOnRoad(const vector<RoadPt>& center_line,
                         set<int>& covered_pts,
                         set<int>& covered_trajs,
                         map<int, bool>& aligned_with_road,
                         Trajectories* trajectories){
    // Compute GPS points that fall on the road represented by vector<RoadPt>& center_line
    covered_pts.clear();
    covered_trajs.clear();
    
    if(trajectories == NULL){
        return;
    }
    
    if(center_line.size() < 2){
        cout << "WARNING from computePointsOnRoad: center_line size less than 2" << endl;
        return;
    }

    // Feature parameters
    float HEADING_THRESHOLD         = 10.0f;
    float WIDTH_RATIO               = 0.8f;
    float search_radius             = 15.0f;
    
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    for(size_t i = 0; i < center_line.size(); ++i){
        const RoadPt& r_pt = center_line[i];
        PclPoint pt;
        pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
        
        Eigen::Vector3d dir = headingTo3dVector(r_pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        tree->radiusSearch(pt,
                           search_radius,
                           k_indices,
                           k_dist_sqrs);
       
        for (size_t s = 0; s < k_indices.size(); ++s) {
            int nearby_pt_idx = k_indices[s];
            PclPoint& nearby_pt = data->at(nearby_pt_idx);
            
            if(covered_pts.find(nearby_pt_idx) != covered_pts.end()){
                // Already added the point, skip
                continue;
            }
            
            int nearby_id_traj = nearby_pt.id_trajectory;
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, r_pt.head));
            if (r_pt.is_oneway) {
                // Check heading
                if (delta_heading > HEADING_THRESHOLD) {
                    continue;
                }
                
                // Check parallel distance
                Eigen::Vector3d vec(nearby_pt.x - r_pt.x,
                                    nearby_pt.y - r_pt.y,
                                    0.0f);
                
                float parallel_dist = abs(dir.cross(vec)[2]);
                if(parallel_dist < WIDTH_RATIO * r_pt.n_lanes * LANE_WIDTH){
                    covered_pts.insert(nearby_pt_idx);
                    covered_trajs.insert(nearby_id_traj);
                    aligned_with_road[nearby_id_traj] = true;
                }
            }
            else{
                // Check heading
                // Check parallel distance
                Eigen::Vector3d vec(nearby_pt.x - r_pt.x,
                                    nearby_pt.y - r_pt.y,
                                    0.0f);
                
                float parallel_dist = dir.cross(vec)[2];
                
                float bwd_delta_heading = 180.0f - delta_heading;
                float half_width = 0.5f * r_pt.n_lanes * LANE_WIDTH;
                if (delta_heading < HEADING_THRESHOLD) {
                    if (parallel_dist < LANE_WIDTH) {
                        if(abs(parallel_dist) < WIDTH_RATIO * half_width){
                            covered_pts.insert(nearby_pt_idx);
                            covered_trajs.insert(nearby_id_traj);
                            aligned_with_road[nearby_id_traj] = true;
                        }
                    }
                }
                
                if (bwd_delta_heading < HEADING_THRESHOLD) {
                    if (parallel_dist > -1.0f * LANE_WIDTH) {
                        if(abs(parallel_dist) < WIDTH_RATIO * half_width){
                            covered_pts.insert(nearby_pt_idx);
                            covered_trajs.insert(nearby_id_traj);
                            aligned_with_road[nearby_id_traj] = false;
                        }
                    }
                }
            }
        }
    }
}

void computeConsistentPointSet(Trajectories* trajectories,
                               const vector<RoadPt>& center_line,
                               set<int>& candidate_point_set,
                               map<int, bool>& traj_aligned_with_road,
                               bool grow_backward
                               ){
    candidate_point_set.clear();
    
    if (center_line.size() < 2) {
        cout << "ERROR from computeConsistentPointSet: center_line size less than 2" << endl;
        return;
    }
    
    Parameters& params = Parameters::getInstance();
    
    float MAX_DIST_TO_ORIGIN        = params.branchPredictorExtensionRatio() * params.searchRadius();
    float MAX_T_EXTENSION         = params.branchPredictorMaxTExtension(); // in seconds. Include points that extend current point by at most this value.
    
    set<int> covered_pts;
    set<int> covered_trajs;
    computePointsOnRoad(center_line,
                        covered_pts,
                        covered_trajs,
                        traj_aligned_with_road,
                        trajectories);
    
    map<int, float> traj_min_ts;
    map<int, float> traj_max_ts;
    
    for (set<int>::iterator it = covered_pts.begin(); it != covered_pts.end(); ++it) {
        PclPoint& covered_pt = trajectories->data()->at(*it);
        int covered_traj_idx = covered_pt.id_trajectory;
        if (traj_min_ts.find(covered_traj_idx) != traj_min_ts.end()) {
            if (traj_min_ts[covered_traj_idx] > covered_pt.t) {
                traj_min_ts[covered_traj_idx] = covered_pt.t;
            }
            
            if (traj_max_ts[covered_traj_idx] < covered_pt.t) {
                traj_max_ts[covered_traj_idx] = covered_pt.t;
            }
        }
        else{
            traj_min_ts[covered_traj_idx] = covered_pt.t;
            traj_max_ts[covered_traj_idx] = covered_pt.t;
        }
    }
    
    PclPoint orig_pt;
    if (grow_backward) {
        orig_pt.setCoordinate(center_line.front().x,
                              center_line.front().y,
                              0.0f);
        orig_pt.head = floor(center_line.front().head);
        
    }
    else{
        orig_pt.setCoordinate(center_line.back().x,
                              center_line.back().y,
                              0.0f);
        orig_pt.head = floor(center_line.back().head);
    }
    
    Eigen::Vector2d orig_dir = headingTo2dVector(orig_pt.head);
    
    // Search all possible points
    vector<int> k_indices;
    vector<float> k_dist_sqrs;
    trajectories->tree()->radiusSearch(orig_pt,
                                       MAX_DIST_TO_ORIGIN,
                                       k_indices,
                                       k_dist_sqrs);
    
    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
        PclPoint& nb_pt = trajectories->data()->at(*it);
        
        Eigen::Vector2d vec_to_orig(nb_pt.x - orig_pt.x,
                                    nb_pt.y - orig_pt.y);
        
        float dot_value = orig_dir.dot(vec_to_orig);
        
        if (grow_backward) {
            if (dot_value > 1.0f) {
                continue;
            }
        }
        else{
            if (dot_value < -1.0f) {
                continue;
            }
        }
        
        // Check if the point traj is in covered_trajs.
        if (covered_trajs.find(nb_pt.id_trajectory) != covered_trajs.end()) {
            float delta_t;
            if (traj_aligned_with_road[nb_pt.id_trajectory]) {
                delta_t = nb_pt.t - traj_max_ts[nb_pt.id_trajectory];
            }
            else{
                delta_t = traj_min_ts[nb_pt.id_trajectory] - nb_pt.t;
            }
            
            if (delta_t < 0) {
                continue;
            }
            
            if (delta_t < MAX_T_EXTENSION) {
                candidate_point_set.insert(*it);
            }
        }
    }
}

//void computeConsistentPointSet(float radius,
//                               Trajectories* trajectories,
//                               vector<RoadPt>& center_line,
//                               set<int>& candidate_point_set,
//                               bool grow_backward
//                               ){
//    candidate_point_set.clear();
//    
//    if (center_line.size() < 2) {
//        cout << "ERROR from computeConsistentPointSet: center_line size less than 2" << endl;
//        return;
//    }
//    
//    float MAX_DIST_TO_ORIGIN        = 4.0f * radius;
//    float MAX_T_EXTENSION         = 30.0f; // in seconds. Include points that extend current point by at most this value.
//    
//    set<int> covered_pts;
//    set<int> covered_trajs;
//    computePointsOnRoad(center_line,
//                        covered_pts,
//                        covered_trajs,
//                        trajectories);
//    
//    PclPoint orig_pt;
//    if (grow_backward) {
//        orig_pt.setCoordinate(center_line.front().x,
//                              center_line.front().y,
//                              0.0f);
//        orig_pt.head = floor(center_line.front().head);
//
//    }
//    else{
//        orig_pt.setCoordinate(center_line.back().x,
//                              center_line.back().y,
//                              0.0f);
//        orig_pt.head = floor(center_line.back().head);
//    }
//    
//    Eigen::Vector2d orig_dir = headingTo2dVector(orig_pt.head);
//    
//    set<int> visited_traj_idxs;
//    for(set<int>::iterator pit = covered_pts.begin(); pit != covered_pts.end(); ++pit){
//        PclPoint& nearby_pt = trajectories->data()->at(*pit);
//        int nearby_traj_idx = nearby_pt.id_trajectory;
//        if (visited_traj_idxs.find(nearby_traj_idx) != visited_traj_idxs.end()) {
//            // This trajectory has already been visited
//            continue;
//        }
//        
//        // Find closest point on the trajectory to orig_pt
//        const vector<int>& traj_pt_idx = trajectories->trajectories()[nearby_traj_idx];
//        set<int>& traj_on_road_idxs = covered_trajs[nearby_traj_idx];
//        
//        float min_dist = 1e10;
//        int min_idx = -1;
//        for (set<int>::iterator pit = traj_on_road_idxs.begin(); pit != traj_on_road_idxs.end(); ++pit) {
//            int pt_idx = traj_pt_idx[*pit];
//            
//            PclPoint& tmp_pt = trajectories->data()->at(pt_idx);
//            Eigen::Vector2d vec(tmp_pt.x - orig_pt.x,
//                                tmp_pt.y - orig_pt.y);
//            float dist = vec.norm();
//            if (dist < min_dist) {
//                min_dist = dist;
//                min_idx = *pit;
//            }
//        }
//        
//        if (min_idx != -1) {
//            PclPoint& closest_pt = trajectories->data()->at(traj_pt_idx[min_idx]);
//            float min_dist = 1e10;
//            int min_dist_idx = -1;
//            for(size_t s = 0; s < center_line.size(); ++s){
//                float delta_x = closest_pt.x - center_line[s].x;
//                float delta_y = closest_pt.y - center_line[s].y;
//                float dist = sqrt(delta_x*delta_x + delta_y*delta_y);
//                if (dist < min_dist) {
//                    min_dist = dist;
//                    min_dist_idx = s;
//                }
//            }
//            
//            float t0 = closest_pt.t;
//            float delta_heading = abs(deltaHeading1MinusHeading2(closest_pt.head, center_line[min_dist_idx].head));
//            
//            if (grow_backward) {
//                if (delta_heading > 90.0f) {
//                    // Opposite direction
//                    for (size_t s = min_idx; s < traj_pt_idx.size(); ++s) {
//                        PclPoint& tmp_pt = trajectories->data()->at(traj_pt_idx[s]);
//                        float delta_t = (tmp_pt.t - t0);
//                        
//                        if (delta_t < 0) {
//                            continue;
//                        }
//                        
//                        if (delta_t > MAX_T_EXTENSION) {
//                            continue;
//                        }
//                        
//                        Eigen::Vector2d vec(tmp_pt.x - orig_pt.x,
//                                            tmp_pt.y - orig_pt.y);
//                        float dist = vec.norm();
//                        if (dist > MAX_DIST_TO_ORIGIN) {
//                            continue;
//                        }
//                        
//                        float dot_value = vec.dot(orig_dir);
//                        
//                        if (dot_value < 1.0f) {
//                            candidate_point_set.insert(traj_pt_idx[s]);
//                        }
//                    }
//                }
//                else{
//                    // Same direction
//                    for (int s = min_idx; s >=0; --s) {
//                        PclPoint& tmp_pt = trajectories->data()->at(traj_pt_idx[s]);
//                        float delta_t = t0 - tmp_pt.t;
//                        
//                        if (delta_t < 0) {
//                            continue;
//                        }
//                        
//                        if (delta_t > MAX_T_EXTENSION) {
//                            continue;
//                        }
//                        
//                        Eigen::Vector2d vec(tmp_pt.x - orig_pt.x,
//                                            tmp_pt.y - orig_pt.y);
//                        float dist = vec.norm();
//                        if (dist > MAX_DIST_TO_ORIGIN) {
//                            continue;
//                        }
//                        
//                        float dot_value = vec.dot(orig_dir);
//                        
//                        if (dot_value < 1.0f) {
//                            candidate_point_set.insert(traj_pt_idx[s]);
//                        }
//                    }
//                }
//            }
//            else{
//                if (delta_heading > 90.0f) {
//                    // Opposite direction
//                    for (int s = min_idx; s >=0; --s) {
//                        PclPoint& tmp_pt = trajectories->data()->at(traj_pt_idx[s]);
//                        float delta_t = t0 - tmp_pt.t;
//                        
//                        if (delta_t < 0) {
//                            continue;
//                        }
//                        
//                        if (delta_t > MAX_T_EXTENSION) {
//                            continue;
//                        }
//                        
//                        Eigen::Vector2d vec(tmp_pt.x - orig_pt.x,
//                                            tmp_pt.y - orig_pt.y);
//                        float dist = vec.norm();
//                        if (dist > MAX_DIST_TO_ORIGIN) {
//                            continue;
//                        }
//                        
//                        float dot_value = vec.dot(orig_dir);
//                        
//                        if (dot_value > -1.0f) {
//                            candidate_point_set.insert(traj_pt_idx[s]);
//                        }
//                    }
//                }
//                else{
//                    // Same direction
//                    for (size_t s = min_idx; s < traj_pt_idx.size(); ++s) {
//                        PclPoint& tmp_pt = trajectories->data()->at(traj_pt_idx[s]);
//                        float delta_t = (tmp_pt.t - t0);
//                        
//                        if (delta_t < 0) {
//                            continue;
//                        }
//                        
//                        if (delta_t > MAX_T_EXTENSION) {
//                            continue;
//                        }
//                        
//                        Eigen::Vector2d vec(tmp_pt.x - orig_pt.x,
//                                            tmp_pt.y - orig_pt.y);
//                        float dist = vec.norm();
//                        if (dist > MAX_DIST_TO_ORIGIN) {
//                            continue;
//                        }
//                        
//                        float dot_value = vec.dot(orig_dir);
//                        
//                        if (dot_value > -1.0f) {
//                            candidate_point_set.insert(traj_pt_idx[s]);
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

//void computeConsistentPointSet(float radius,
//                               Trajectories* trajectories,
//                               vector<RoadPt>& center_line,
//                               set<int>& candidate_point_set,
//                               bool is_oneway,
//                               bool grow_backward
//                               ){
//    candidate_point_set.clear();
//    
//    // (R, theta)distribution of consistent trajectories points + intersection distribution for points that have bigger than 7.5 degree heading diff.
//    if (center_line.size() < 2) {
//        cout << "size less than 2" << endl;
//        return;
//    }
//    
//    PclPointCloud::Ptr& data = trajectories->data();
//    PclSearchTree::Ptr& tree = trajectories->tree();
//    
//    PclPoint orig_pt;
//    if (grow_backward) {
//        orig_pt.setCoordinate(center_line.front().x,
//                              center_line.front().y,
//                              0.0f);
//        orig_pt.head = floor(center_line.front().head);
//        
//    }
//    else{
//        orig_pt.setCoordinate(center_line.back().x,
//                              center_line.back().y,
//                              0.0f);
//        orig_pt.head = floor(center_line.back().head);
//    }
//    
//    Eigen::Vector3d orig_dir = headingTo3dVector(orig_pt.head);
//    
//    // Feature parameters
//    float HEADING_THRESHOLD         = 15.0f;
//    float MAX_DIST_TO_ROAD_CENTER   = 0.5f * radius; // in meters
//    float DELTA_T_EXTENSION         = 30.0f; // in seconds. Include points that extend current point by at most this value.
////    float MAX_EXTENSION             = 3.0f * radius; // we will trace the centerline for up to this distance to find trajectory fall on this road
//    
//    // Find trajectories that falls on this road, and compute angle distribution
//    PclPoint pt;
//    if (grow_backward) {
//        // backward direction
//        float cum_length = 0.0f; // the maximum extension length we will look at
//        
//        set<int> visited_traj_idxs;
//        for (int i = 0; i < center_line.size(); ++i) {
//            if (i != 0) {
//                float delta_x = center_line[i].x - center_line[i-1].x;
//                float delta_y = center_line[i].y - center_line[i-1].y;
//                cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
//            }
//            
//            pt.setCoordinate(center_line[i].x, center_line[i].y, 0.0f);
//            Eigen::Vector3d dir = headingTo3dVector(center_line[i].head);
//            
//            // Look nearby
//            vector<int> k_indices;
//            vector<float> k_dist_sqrs;
//            tree->radiusSearch(pt,
//                               radius,
//                               k_indices,
//                               k_dist_sqrs);
//            for (size_t s = 0; s < k_indices.size(); ++s) {
//                int nearby_pt_idx = k_indices[s];
//                PclPoint& nearby_pt = data->at(nearby_pt_idx);
//                int nearby_traj_idx = nearby_pt.id_trajectory;
//                if (visited_traj_idxs.find(nearby_traj_idx) != visited_traj_idxs.end()) {
//                    // This trajectory has already been visited
//                    continue;
//                }
//                
//                Eigen::Vector3d vec(nearby_pt.x - pt.x,
//                                    nearby_pt.y - pt.y,
//                                    0.0f);
//                
//                Eigen::Vector3d vec0(nearby_pt.x - orig_pt.x,
//                                     nearby_pt.y - orig_pt.y,
//                                     0.0f);
//                
//                // Check position
//                if (vec0.dot(orig_dir) < 1.0f) {
//                    continue;
//                }
//                
//                // Check heading
//                float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, center_line[i].head));
//                float corrected_heading = delta_heading;
//                if (!is_oneway) {
//                    if (delta_heading > 90.0f) {
//                        corrected_heading = 180.0f - delta_heading;
//                    }
//                }
//                if (corrected_heading > HEADING_THRESHOLD) {
//                    continue;
//                }
//                
//                // Check perpendicular dist
//                float perp_dist = dir.cross(vec)[2];
//                if (abs(perp_dist) > MAX_DIST_TO_ROAD_CENTER) {
//                    continue;
//                }
//                
//                visited_traj_idxs.insert(nearby_traj_idx);
//                
//                bool look_backward = true;
//                if (!is_oneway) {
//                    if (delta_heading > 90.0f) {
//                        look_backward = false;
//                    }
//                }
//                
//                const vector<int>& traj_pt_idxs = trajectories->trajectories()[nearby_traj_idx];
//                if (traj_pt_idxs.size() == 0) {
//                    continue;
//                }
//                
//                int id_sample = nearby_pt.id_sample;
//                float start_time = nearby_pt.t;
//                
//                if (!look_backward) {
//                    bool passed_query_point = false;
//                    for (int l = id_sample; l < traj_pt_idxs.size(); ++l) {
//                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
//                        
//                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
//                                                 this_pt.y - orig_pt.y,
//                                                 0.0f);
//                        
//                        // Check position
//                        if(!passed_query_point){
//                            if (this_vec.dot(orig_dir) < -1.0f) {
//                                passed_query_point = true;
//                            }
//                        }
//                        
//                        if (passed_query_point) {
//                            float delta_t = abs(this_pt.t - start_time);
//                            float delta_x = this_pt.x - orig_pt.x;
//                            float delta_y = this_pt.y - orig_pt.y;
//                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
//                            
//                            if (dist_to_orig < 4 * radius &&
//                                delta_t < DELTA_T_EXTENSION) {
//                                // Insert to candidate point set
//                                candidate_point_set.insert(traj_pt_idxs[l]);
//                            }
//                            else{
//                                break;
//                            }
//                        }
//                    }
//                }
//                else{
//                    bool passed_query_point = false;
//                    for (int l = id_sample; l >= 0; --l) {
//                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
//                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
//                                                 this_pt.y - orig_pt.y,
//                                                 0.0f);
//                        
//                        // Check position
//                        if(!passed_query_point){
//                            if (this_vec.dot(orig_dir) < -1.0f) {
//                                passed_query_point = true;
//                            }
//                        }
//                        if (passed_query_point) {
//                            float delta_t = abs(this_pt.t - start_time);
//                            float delta_x = this_pt.x - orig_pt.x;
//                            float delta_y = this_pt.y - orig_pt.y;
//                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
//                            if (dist_to_orig < 4 * radius &&
//                                delta_t < DELTA_T_EXTENSION) {
//                                // Insert to candidate point set
//                                candidate_point_set.insert(traj_pt_idxs[l]);
//                            }
//                            else{
//                                break;
//                            }
//                        }
//                    }
//                }
//            }
////            if (cum_length > MAX_EXTENSION) {
////                break;
////            }
//        }
//    }
//    else{
//        // forward direction
//        float cum_length = 0.0f; // the extension length we will look at
//        
//        set<int> visited_traj_idxs;
//        for (int i = center_line.size()-1; i >= 0; --i) {
//            if (i != center_line.size() - 1) {
//                float delta_x = center_line[i].x - center_line[i+1].x;
//                float delta_y = center_line[i].y - center_line[i+1].y;
//                cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
//            }
//            
//            pt.setCoordinate(center_line[i].x, center_line[i].y, 0.0f);
//            Eigen::Vector3d dir = headingTo3dVector(center_line[i].head);
//            
//            // Look nearby
//            vector<int> k_indices;
//            vector<float> k_dist_sqrs;
//            tree->radiusSearch(pt,
//                               radius,
//                               k_indices,
//                               k_dist_sqrs);
//            for (size_t s = 0; s < k_indices.size(); ++s) {
//                int nearby_pt_idx = k_indices[s];
//                PclPoint& nearby_pt = data->at(nearby_pt_idx);
//                int nearby_traj_idx = nearby_pt.id_trajectory;
//                
//                if (visited_traj_idxs.find(nearby_traj_idx) != visited_traj_idxs.end()) {
//                    // This trajectory has already been visited
//                    continue;
//                }
//                
//                Eigen::Vector3d vec(nearby_pt.x - pt.x,
//                                    nearby_pt.y - pt.y,
//                                    0.0f);
//                
//                Eigen::Vector3d vec0(nearby_pt.x - orig_pt.x,
//                                     nearby_pt.y - orig_pt.y,
//                                     0.0f);
//                
//                // Check position
//                if (vec0.dot(orig_dir) > -1.0f) {
//                    continue;
//                }
//                
//                // Check heading
//                float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, center_line[i].head));
//                float corrected_heading = delta_heading;
//                if (!is_oneway) {
//                    if (delta_heading > 90.0f) {
//                        corrected_heading = 180.0f - delta_heading;
//                    }
//                }
//                if (corrected_heading > HEADING_THRESHOLD) {
//                    continue;
//                }
//                
//                // Check perpendicular dist
//                float perp_dist = dir.cross(vec)[2];
//                if (abs(perp_dist) > MAX_DIST_TO_ROAD_CENTER) {
//                    continue;
//                }
//                
//                // Now, this trajectory is compatible with current road, compute its heading change cross the search region
//                // Find the first point that's outside the region
//                visited_traj_idxs.insert(nearby_traj_idx);
//                
//                bool look_forward = true;
//                if (!is_oneway) {
//                    if (delta_heading > 90.0f) {
//                        look_forward = false;
//                    }
//                }
//                
//                const vector<int>& traj_pt_idxs = trajectories->trajectories()[nearby_traj_idx];
//                if (traj_pt_idxs.size() == 0) {
//                    continue;
//                }
//                
//                int id_sample = nearby_pt.id_sample;
//                float start_time = nearby_pt.t;
//                
//                if (look_forward) {
//                    bool passed_query_point = false;
//                    for (int l = id_sample; l < traj_pt_idxs.size(); ++l) {
//                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
//                        
//                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
//                                                 this_pt.y - orig_pt.y,
//                                                 0.0f);
//                        
//                        // Check position
//                        if(!passed_query_point){
//                            if (this_vec.dot(orig_dir) > 1.0f) {
//                                passed_query_point = true;
//                            }
//                        }
//                        
//                        if (passed_query_point) {
//                            float delta_t = abs(this_pt.t - start_time);
//                            float delta_x = this_pt.x - orig_pt.x;
//                            float delta_y = this_pt.y - orig_pt.y;
//                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
//                            if (dist_to_orig < 4 * radius &&
//                                delta_t < DELTA_T_EXTENSION) {
//                                // Insert to candidate point set
//                                candidate_point_set.insert(traj_pt_idxs[l]);
//                            }
//                            else{
//                                break;
//                            }
//                        }
//                    }
//                }
//                else{
//                    bool passed_query_point = false;
//                    for (int l = id_sample; l >= 0; --l) {
//                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
//                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
//                                                 this_pt.y - orig_pt.y,
//                                                 0.0f);
//                        
//                        // Check position
//                        if(!passed_query_point){
//                            if (this_vec.dot(orig_dir) > 1.0f) {
//                                passed_query_point = true;
//                            }
//                        }
//                        if (passed_query_point) {
//                            float delta_t = abs(this_pt.t - start_time);
//                            float delta_x = this_pt.x - orig_pt.x;
//                            float delta_y = this_pt.y - orig_pt.y;
//                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
//                            if (dist_to_orig < 4 * radius &&
//                                delta_t < DELTA_T_EXTENSION) {
//                                // Insert to candidate point set
//                                candidate_point_set.insert(traj_pt_idxs[l]);
//                            }
//                            else{
//                                break;
//                            }
//                        }
//                    }
//                }
//            }
////            if (cum_length > MAX_EXTENSION) {
////                break;
////            }
//        }
//    }
//}

bool computeQueryQFeatureAt(float radius,
                            Trajectories* trajectories,
                            query_q_sample_type& feature,
                            vector<RoadPt>& center_line,
                            bool grow_backward){
    // (R, theta)distribution of consistent trajectories points + intersection distribution for points that have bigger than 7.5 degree heading diff.
    if (center_line.size() < 2) {
        cout << "size less than 2" << endl;
        return false;
    }
    
    PclPointCloud::Ptr& data = trajectories->data();
    
    PclPoint orig_pt;
    if (grow_backward) {
        orig_pt.setCoordinate(center_line.front().x,
                              center_line.front().y,
                              0.0f);
        orig_pt.head = floor(center_line.front().head);
        
    }
    else{
        orig_pt.setCoordinate(center_line.back().x,
                              center_line.back().y,
                              0.0f);
        orig_pt.head = floor(center_line.back().head);
    }
    
    Eigen::Vector3d orig_dir = headingTo3dVector(orig_pt.head);
    
    // Feature parameters
        // Heading histogram parameters
    int N_HEADING_BINS = 8;
    float DELTA_HEADING_BIN = 360.0f / N_HEADING_BINS;
    
        // Speed histogram
    int N_SPEED_BINS = 4;
    float MAX_LOW_SPEED = 5.0f; // meter per second
    float MAX_MID_LOW_SPEED = 10.0f; // meter per second
    float MAX_MID_HIGH_SPEED = 20.0f; // meter per second
    
        // Second part of the feature: intersection distribution to the canonical line, positive ahead
    int     N_INTERSECTION_BIN      = 12;
    float   delta_intersection_bin  = 10.0f; // in meters
    
    // Find trajectories that falls on this road, and compute angle distribution
    PclPoint pt;
    set<int> candidate_point_set;
    map<int, bool> traj_aligned_with_road;
    computeConsistentPointSet(trajectories,
                              center_line,
                              candidate_point_set,
                              traj_aligned_with_road,
                              grow_backward);
    
    // Now candidate points are stored in candidate_point_set, we can start computing angle distributions of these points w.r.t. the original query point.
    // Parameters:
        // N_ANGLE_BINS, delta_angle_bin, N_R_BINS, delta_r_bin
        // N_INTERSECTION_BIN, delta_intersection_bin, MIN_HEADING_DIFF
    
    if(candidate_point_set.size() < 5){
        return false;
    }
    
    int N_FIRST_PART_BINS = N_HEADING_BINS * N_SPEED_BINS;
    vector<double> speed_heading_hist(N_FIRST_PART_BINS, 0.0f);
    vector<double> snd_part_hist(N_INTERSECTION_BIN, 0.0f);
    
    for (set<int>::iterator it = candidate_point_set.begin(); it != candidate_point_set.end(); ++it) {
        PclPoint& this_pt = data->at(*it);
        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
                                 this_pt.y - orig_pt.y,
                                 0.0f);
        float this_vec_length = this_vec.norm();
        if(this_vec_length > 1e-3){
            this_vec /= this_vec_length;
        }
        else{
            this_vec *= 0.0f;
        }
        
        Eigen::Vector3d corrected_orig_dir = orig_dir;
        
        // corrected_orig_dir is the cannonical direction
        if (grow_backward) {
            corrected_orig_dir *= -1.0f;
        }
        
        float dot_value = abs(this_vec.dot(corrected_orig_dir));
        if(dot_value > 1.0f){
            dot_value = 1.0f;
        }
        
        // Compute speed heading histogram
        float speed = this_pt.speed * 1.0f / 100.0f;
        float delta_heading = deltaHeading1MinusHeading2(this_pt.head, orig_pt.head);
        
        if (delta_heading < 0.0f) {
            delta_heading += 360.0f;
        }
        
        int heading_bin_idx = floor(delta_heading / DELTA_HEADING_BIN);
        if (heading_bin_idx >= N_HEADING_BINS) {
            heading_bin_idx = N_HEADING_BINS - 1;
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
        
        // Compute the first part of the descriptor
        float cross_value = corrected_orig_dir.cross(this_vec)[2];
        
        // Compute the second part of the descriptor
        float projected_x = -1.0f * this_vec_length * cross_value;
        float projected_y = this_vec_length * this_vec.dot(corrected_orig_dir);
        
        float this_pt_heading_in_radius = this_pt.head * PI / 180.0f;
        Eigen::Vector3d this_pt_heading_dir(cos(this_pt_heading_in_radius),
                                            sin(this_pt_heading_in_radius),
                                            0.0f);
        float cos_theta = this_pt_heading_dir.dot(corrected_orig_dir);
        float sin_theta = sqrt(1.0f - cos_theta*cos_theta);
        
        float infinity = 1e6;
        float y_intersect = 0.0f;
        if (abs(sin_theta) < 1e-3) {
            y_intersect = infinity;
        }
        else{
            float l = abs(projected_x) * cos_theta / sin_theta;
            y_intersect = projected_y - l;
        }
        
        int tmp_proj_idx = floor(abs(y_intersect) / delta_intersection_bin);
        int snd_part_idx = -1;
        if (tmp_proj_idx < 0.5f * N_INTERSECTION_BIN) {
            if (y_intersect > 0) {
                snd_part_idx = tmp_proj_idx + (N_INTERSECTION_BIN / 2);
            }
            else{
                snd_part_idx = (N_INTERSECTION_BIN / 2) - tmp_proj_idx;
            }
            if (snd_part_idx < 0) {
                snd_part_idx = 0;
            }
            if (snd_part_idx >= N_INTERSECTION_BIN) {
                snd_part_idx = N_INTERSECTION_BIN - 1;
            }
            
            snd_part_hist[snd_part_idx] += 1.0f;
        }
    }
    
    // Normalize speed heading histogram
    float cum_sum = 0.0f;
    for (size_t i = 0; i < speed_heading_hist.size(); ++i) {
        cum_sum += speed_heading_hist[i];
    }
    if (cum_sum > 1e-3) {
        for (size_t i = 0; i < speed_heading_hist.size(); ++i) {
            speed_heading_hist[i] /= cum_sum;
        }
    }
    
    // Normalize snd_part_hist
    cum_sum = 0.0f;
    for (size_t i = 0; i < snd_part_hist.size(); ++i) {
        cum_sum += snd_part_hist[i];
    }
    if (cum_sum > 1e-3) {
        for (size_t i = 0; i < snd_part_hist.size(); ++i) {
            snd_part_hist[i] /= cum_sum;
        }
    }
    
    // Store speed heading histogram into feature
    for (int i = 0; i < speed_heading_hist.size(); ++i) {
        feature(i) = speed_heading_hist[i];
    }
    
    // Store snd_part_hist to feature
//    int start_offset = speed_heading_hist.size();
//    for (int i = 0; i < snd_part_hist.size(); ++i) {
//        int idx = i + start_offset;
//        feature(idx) = snd_part_hist[i];
//    }
    
    return true;
}

bool computeQueryQFeatureAtForVisualization(float radius,
                                            Trajectories* trajectories,
                                            query_q_sample_type& feature,
                                            vector<RoadPt>& center_line,
                                            set<int>& candidate_point_set,
                                            bool grow_backward){
    candidate_point_set.clear();
    
    // (R, theta)distribution of consistent trajectories points + intersection distribution for points that have bigger than 7.5 degree heading diff.
    if (center_line.size() < 2) {
        cout << "size less than 2" << endl;
        return false;
    }
    
    map<int, bool> traj_aligned_with_road;
    computeConsistentPointSet(trajectories,
                              center_line,
                              candidate_point_set,
                              traj_aligned_with_road,
                              grow_backward);
    
    return true;
}

bool tmpCompare(const pair<int, float>& firstElem, const pair<int, float>& secondElem){
    return firstElem.second < secondElem.second;
}

double growModelScore(vector<float>& xs,
                      vector<float>& ys,
                      int at_idx,
                      PclPointCloud::Ptr& points,
                      vector<vector<int> >& partition,
                      double dx,
                      double dy){
    double score = 0.0f;
    if(xs.size() < 2){
        return score;
    }
    
    int N = xs.size();
    double LAMBDA = 0.1f;
    // Distance score
    float new_xs = xs[at_idx] + dx;
    float new_ys = ys[at_idx] + dy;
    
    vector<int>& pts_to_this_point = partition[at_idx];
    for(size_t i = 0; i < pts_to_this_point.size(); ++i){
        PclPoint& pt = points->at(pts_to_this_point[i]);
        float delta_x = pt.x - new_xs;
        float delta_y = pt.y - new_ys;
        score += sqrt(delta_x*delta_x + delta_y*delta_y);
    }
    
    if(at_idx > 0){
        vector<int>& prev_seg_point = partition[at_idx + N - 1];
        Eigen::Vector3d prev_seg_dir(new_xs - xs[at_idx-1],
                                     new_ys - ys[at_idx-1],
                                     0.0f);
        
        double prev_seg_dir_length = prev_seg_dir.norm();
        if(prev_seg_dir_length > 1e-3){
            prev_seg_dir /= prev_seg_dir_length;
            for(size_t i = 0; i < prev_seg_point.size(); ++i){
                PclPoint& pt = points->at(prev_seg_point[i]);
                
                Eigen::Vector3d vec(pt.x - new_xs,
                                    pt.y - new_ys,
                                    0.0f);
                
                score += abs(prev_seg_dir.cross(vec)[2]);
            }
        }
    }
    
    if(at_idx < N - 1){
        vector<int>& nxt_seg_point = partition[at_idx + N];
        Eigen::Vector3d nxt_seg_dir(xs[at_idx + 1] - new_xs,
                                    ys[at_idx + 1] - new_ys,
                                     0.0f);
        
        double nxt_seg_dir_length = nxt_seg_dir.norm();
        if(nxt_seg_dir_length > 1e-3){
            nxt_seg_dir /= nxt_seg_dir_length;
            for(size_t i = 0; i < nxt_seg_point.size(); ++i){
                PclPoint& pt = points->at(nxt_seg_point[i]);
                
                Eigen::Vector3d vec(pt.x - new_xs,
                                    pt.y - new_ys,
                                    0.0f);
                
                score += abs(nxt_seg_dir.cross(vec)[2]);
            }
        }
    }
    
    score /= points->size();
    
    // Smoothness term
    if(at_idx > 0 && at_idx < N - 1){
        Eigen::Vector2d vec0(xs[at_idx + 1] - new_xs,
                             ys[at_idx + 1] - new_ys);
        Eigen::Vector2d vec1(xs[at_idx - 1] - new_xs,
                             ys[at_idx - 1] - new_ys);
        float vec0_length = vec0.norm();
        float vec1_length = vec1.norm();
        if(vec0_length > 1e-3 && vec1_length > 1e-3){
            vec0 /= vec0_length;
            vec1 /= vec1_length;
            double length = sqrt(vec0_length*vec1_length);
            float dot_value = vec0.dot(vec1);
            score += (LAMBDA * length * (1 + dot_value));
        }
    }
    
    return score;
}

float optimizeGrowModel(vector<float>& xs,
                        vector<float>& ys,
                        PclPointCloud::Ptr& points,
                        PclSearchTree::Ptr& search_tree){
    if(xs.size() < 2){
        return 0.0f;
    }
    
    int N = xs.size();
    int K = 2 * N - 1;
    int max_iter = 20;
    
    for(int i_iter = 0; i_iter < max_iter; ++i_iter){
        // Partition the points, remove outliers
        vector<vector<int> > partition(K, vector<int>());
        for (size_t i = 0; i < points->size(); ++i) {
            PclPoint& pt = points->at(i);
            float min_dist = 1e10;
            int   proj_idx = -1;
            for (int s = 1; s < xs.size(); ++s) {
                Eigen::Vector2d vec0(pt.x - xs[s-1], pt.y - ys[s-1]);
                Eigen::Vector2d vec1(pt.x - xs[s], pt.y - ys[s]);
                Eigen::Vector2d dir(xs[s] - xs[s-1], ys[s] - ys[s-1]);
                float dir_length = dir.norm();
                float length0 = vec0.norm();
                float length1 = vec1.norm();
                
                if (dir_length < 0.1f) {
                    if (min_dist > length0) {
                        proj_idx = s - 1;
                        min_dist = length0;
                        continue;
                    }
                }
                
                dir /= dir_length;
                float dot_value = dir.dot(vec0);
               
                if (dot_value < 0.0f) {
                    if (min_dist > length0) {
                        proj_idx = s-1;
                        min_dist = length0;
                    }
                }
                else if(dot_value > dir_length) {
                    if (min_dist > length1) {
                        proj_idx = s;
                        min_dist = length1;
                    }
                }
                else{
                    float dist = sqrt(length0*length0 - dot_value*dot_value);
                    if (min_dist > dist) {
                        proj_idx = s + N - 1;
                        min_dist = dist;
                    }
                }
            }
            
            if (min_dist > 25.0f) {
                continue;
            }
            partition[proj_idx].push_back(i);
        }
        
        // Gradient descent
        int MAX_K = 50;
        int g_iter = 0;
        while(g_iter < MAX_K){
            float cum_change = 0.0f;
            for(int s = 1; s < xs.size(); ++s){
                // Compute gradient at index s
                double dx = 1e-3;
                double dy = 1e-3;
                
                double g0 = growModelScore(xs,
                                           ys,
                                           s,
                                           points,
                                           partition);
                
                double dgdx = growModelScore(xs,
                                             ys,
                                             s,
                                             points,
                                             partition,
                                             dx) - g0;
                dgdx /= dx;
                double dgdy = growModelScore(xs,
                                             ys,
                                             s,
                                             points,
                                             partition,
                                             0.0f,
                                             dy) - g0;
                dgdy /= dy;
                Eigen::Vector2d g_dir(dgdx, dgdy);
                float g_dir_length = g_dir.norm();
                if(g_dir_length < 1e-3){
                    continue;
                }
                g_dir /= g_dir_length;
                xs[s] -= g_dir.x();
                ys[s] -= g_dir.y();
                cum_change += 1.0f;
            }
            if(cum_change < 1e-3){
                break;
            }
            g_iter++;
        }
    }
    
    float score = 0.0f;
    
    // Partition the points
    for (size_t i = 0; i < points->size(); ++i) {
        PclPoint& pt = points->at(i);
        float min_dist = 1e10;
        for (int s = 1; s < xs.size(); ++s) {
            Eigen::Vector2d vec0(pt.x - xs[s-1], pt.y - ys[s-1]);
            Eigen::Vector2d vec1(pt.x - xs[s], pt.y - ys[s]);
            Eigen::Vector2d dir(xs[s] - xs[s-1], ys[s] - ys[s-1]);
            float dir_length = dir.norm();
            float length0 = vec0.norm();
            float length1 = vec1.norm();
            
            if (dir_length < 0.1f) {
                if (min_dist > length0) {
                    min_dist = length0;
                    continue;
                }
            }
            
            dir /= dir_length;
            float dot_value = dir.dot(vec0);
            
            if (dot_value < 0.0f) {
                if (min_dist > length0) {
                    min_dist = length0;
                }
            }
            else if(dot_value > dir_length) {
                if (min_dist > length1) {
                    min_dist = length1;
                }
            }
            else{
                float dist = sqrt(length0*length0 - dot_value*dot_value);
                if (min_dist > dist) {
                    min_dist = dist;
                }
            }
        }
        
        score += (min_dist*min_dist);
    }
    
    return score;
}

class dfs_depth_visitor : public default_dfs_visitor{
public:
    dfs_depth_visitor(vector<int>& depth, vector<int>& parents) : depth_(depth),
                                                                parents_(parents){
    }
    
    template< typename Edge, typename Graph >
    void tree_edge(Edge e, const Graph & g) {
        int source_idx = source(e, g);
        int target_idx = target(e, g);
        depth_[target_idx] = depth_[source_idx] + 1;
        parents_[target_idx] = source_idx;
    }
    
private:
    vector<int>& depth_;
    vector<int>& parents_;
};

//float growModelFitting(RoadPt& start_point,
//                       PclPointCloud::Ptr& points,
//                       PclSearchTree::Ptr& search_tree,
//                       vector<RoadPt>& extension,
//                       bool grow_backward){
//    if (points->size() == 0) {
//        return 0.0f;
//    }
//    
//    bool is_oneway = start_point.is_oneway;
//    
//    // Move point around as skeleton extraction
//    PclPointCloud::Ptr tmp_points(new PclPointCloud);
//    PclSearchTree::Ptr tmp_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//   
//    for (size_t i = 0; i < points->size(); ++i) {
//        tmp_points->push_back(points->at(i));
//    }
//
//    tmp_search_tree->setInputCloud(tmp_points);
//    
//    float cum_change;
//    int MAX_ITER = 30;
//    float HEADING_THRESHOLD = 15.0f;
//    int i_iter = 0;
//    float search_radius = 15.0f;
//    while (i_iter < MAX_ITER) {
//        cum_change = 0.0f;
//        
//        for (size_t i = 0; i < tmp_points->size(); ++i) {
//            PclPoint& pt = tmp_points->at(i);
//            Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
//            Eigen::Vector3d pt_perp_dir(-pt_dir[1], pt_dir[0], 0.0f);
//            
//            vector<int> k_indices;
//            vector<float> k_dist_sqrs;
//            tmp_search_tree->radiusSearch(pt, search_radius, k_indices, k_dist_sqrs);
//            
//            float cum_perp_proj = 0.0f;
//            int   count = 0;
//            for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//                PclPoint& nb_pt = tmp_points->at(*it);
//                float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
//                if (!is_oneway && delta_heading > 90.0f) {
//                    delta_heading = 180.0f - delta_heading;
//                }
//                
//                if (delta_heading < HEADING_THRESHOLD) {
//                    Eigen::Vector3d vec(nb_pt.x - pt.x, nb_pt.y - pt.y, 0.0f);
//                    
//                    cum_perp_proj += pt_dir.cross(vec)[2];
//                    
//                    count++;
//                }
//            }
//            
//            if (count != 0) {
//                cum_perp_proj /= count;
//                
//                cum_change += abs(cum_perp_proj);
//                
//                Eigen::Vector3d v0(pt.x, pt.y, 0.0f);
//                v0 += (cum_perp_proj * pt_perp_dir);
//                pt.x = v0.x();
//                pt.y = v0.y();
//            }
//        }
//        
//        tmp_search_tree->setInputCloud(tmp_points);
//        
//        if (cum_change < 1.0f) {
//            break;
//        }
//        
//        i_iter++;
//    }
//    
//    // Move point around as skeleton extraction
//    PclPointCloud::Ptr simplified_points(new PclPointCloud);
//    PclSearchTree::Ptr simplified_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    
//    vector<int> is_covered(tmp_points->size(), false);
//    for (size_t i = 0; i < tmp_points->size(); ++i) {
//        if (is_covered[i]) {
//            continue;
//        }
//        
//        is_covered[i] = true;
//        PclPoint& pt = tmp_points->at(i);
//        simplified_points->push_back(pt);
//        
//        vector<int> k_indices;
//        vector<float> k_dist_sqrs;
//        
//        tmp_search_tree->radiusSearch(pt, 5.0f, k_indices, k_dist_sqrs);
//        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//            if (is_covered[*it]) {
//                continue;
//            }
//            
//            PclPoint& nearby_pt = tmp_points->at(*it);
//            float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, pt.head));
//            
//            if (!is_oneway && delta_heading > 90.0f) {
//                delta_heading = 180.0f - delta_heading;
//            }
//            
//            if (delta_heading < HEADING_THRESHOLD) {
//                is_covered[*it] = true;
//            }
//        }
//    }
//    
//    simplified_search_tree->setInputCloud(simplified_points);
//    
//    // Build a undirected graph
//    typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_distance_t, float>, property < edge_weight_t, float> > graph_t;
//    typedef typename graph_t::edge_property_type Weight;
//   
//    typedef graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
//    typedef graph_traits<graph_t>::edge_descriptor edge_descriptor;
//    typedef pair<int, int> E;
//    vector<float> edge_weights;
//    vector<E>     edge_list;
//    
//    vector<int> k_indices;
//    vector<float> k_dist_sqrs;
//    PclPoint pt;
//    pt.setCoordinate(start_point.x, start_point.y, 0.0f);
//    pt.head = start_point.head;
//    
//    Eigen::Vector2d start_pt_dir = headingTo2dVector(pt.head);
//    
//    simplified_search_tree->radiusSearch(pt, 1.2f * search_radius, k_indices, k_dist_sqrs);
//    
//    for (size_t i = 0; i != k_indices.size(); ++i) {
//        PclPoint nb_pt = simplified_points->at(k_indices[i]);
//        Eigen::Vector2d nb_pt_dir = headingTo2dVector(nb_pt.head);
//        
//        float dot_value = nb_pt_dir.dot(start_pt_dir);
//        if (!is_oneway) {
//            dot_value = abs(dot_value);
//        }
//   
//        float edge_weight = sqrt(k_dist_sqrs[i]) * (2.0f - dot_value);
//        
//        edge_list.push_back(E(0, k_indices[i]+1));
//        edge_weights.push_back(edge_weight);
//    }
//    
//    for (size_t i = 0; i < simplified_points->size(); ++i) {
//        PclPoint& this_pt = simplified_points->at(i);
//        Eigen::Vector2d this_pt_dir = headingTo2dVector(this_pt.head);
//        vector<int> k_inds;
//        vector<float> k_dists;
//        
//        simplified_search_tree->radiusSearch(this_pt, 1.2f * search_radius, k_inds, k_dists);
//        
//        for (size_t s = 0; s != k_inds.size(); ++s) {
//            if (k_inds[s] == i) {
//                continue;
//            }
//            
//            PclPoint nb_pt = simplified_points->at(k_inds[s]);
//            Eigen::Vector2d nb_pt_dir = headingTo2dVector(nb_pt.head);
//            
//            float dot_value = nb_pt_dir.dot(this_pt_dir);
//            if (!is_oneway) {
//                dot_value = abs(dot_value);
//            }
//            
//            float edge_weight = sqrt(k_dists[s]) * (2.0f - dot_value);
//            
//            edge_list.push_back(E(i+1, k_inds[s]+1));
//            edge_weights.push_back(edge_weight);
//        }
//    }
//    
//    graph_t G;
//    
//    for(size_t i = 0; i < edge_list.size(); ++i){
//        add_edge(edge_list[i].first, edge_list[i].second, edge_weights[i], G);
//    }
//    
//    vector<edge_descriptor> spanning_tree;
//    //kruskal_minimum_spanning_tree(G, back_inserter(spanning_tree));;
//    
//    vector<vertex_descriptor> p(num_vertices(G));
//    vector<float> d(num_vertices(G));
//    dijkstra_shortest_paths(G, 0,
//                            predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, G))).
//                            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, G))));
//    
//    property_map<graph_t, edge_weight_t>::type w = get(edge_weight, G);
//    
//    extension.clear();
//    int count = 0;
//    
//    // Construct the spanning tree
//    graph_t MST;
//    
//    for (vector<edge_descriptor>::iterator ei = spanning_tree.begin();
//         ei != spanning_tree.end(); ++ei) {
//        count ++;
//        int source_idx = source(*ei, G);
//        int target_idx = target(*ei, G);
//        
//        add_edge(source_idx, target_idx, get(w, *ei),MST);
//    }
//    
//    vector<int> depth(num_vertices(MST), 0);
//    vector<int> parents(num_vertices(MST), -1);
//    if (depth.size() > 0) {
//        dfs_depth_visitor vis(depth, parents);
//        
//        depth_first_search(MST, visitor(vis).root_vertex(0));
//        
//        int max_depth = -1;
//        int max_depth_idx = -1;
//        for(size_t i = 0; i < depth.size(); ++i){
//            if(depth[i] > max_depth){
//                max_depth = depth[i];
//                max_depth_idx = i;
//            }
//        }
//        
//        vector<int> path;
//        path.push_back(max_depth_idx);
//        int p = parents[max_depth_idx];
//        int cur_node = max_depth_idx;
//        while(p != -1){
//            cur_node = p;
//            path.insert(path.begin(), p);
//            p = parents[cur_node];
//        }
//        for(size_t i = 0; i < path.size()-1; ++i){
//            int source_idx = path[i];
//            int target_idx = path[i+1];
//            if (source_idx == 0) {
//                extension.push_back(RoadPt(start_point.x,
//                                           start_point.y,
//                                           start_point.head,
//                                           is_oneway));
//            }
//            else{
//                PclPoint& pt = simplified_points->at(source_idx-1);
//                extension.push_back(RoadPt(pt.x,
//                                           pt.y,
//                                           pt.head,
//                                           is_oneway));
//            }
//            
//            if (target_idx == 0) {
//                extension.push_back(RoadPt(start_point.x,
//                                           start_point.y,
//                                           start_point.head,
//                                           is_oneway));
//            }
//            else{
//                PclPoint& pt = simplified_points->at(target_idx-1);
//                extension.push_back(RoadPt(pt.x,
//                                           pt.y,
//                                           pt.head,
//                                           is_oneway));
//            }
//        }
//    }
//    
//    smoothCurve(extension, true);
//   
//    float score = 0.0f;
//    for (size_t i = 0; i < points->size(); ++i) {
//        PclPoint& pt = points->at(i);
//        float min_dist = 1e10;
//        for (int s = 1; s < extension.size(); ++s) {
//            Eigen::Vector2d vec0(pt.x - extension[s-1].x, pt.y - extension[s-1].y);
//            Eigen::Vector2d vec1(pt.x - extension[s].x, pt.y - extension[s].y);
//            Eigen::Vector2d dir(extension[s].x - extension[s-1].x,
//                                extension[s].y - extension[s-1].y);
//            
//            float dir_length = dir.norm();
//            float length0 = vec0.norm();
//            float length1 = vec1.norm();
//            
//            if (dir_length < 0.1f) {
//                if (min_dist > length0) {
//                    min_dist = length0;
//                    continue;
//                }
//            }
//            
//            dir /= dir_length;
//            float dot_value = dir.dot(vec0);
//            
//            if (dot_value < 0.0f) {
//                if (min_dist > length0) {
//                    min_dist = length0;
//                }
//            }
//            else if(dot_value > dir_length) {
//                if (min_dist > length1) {
//                    min_dist = length1;
//                }
//            }
//            else{
//                float dist = sqrt(length0*length0 - dot_value*dot_value);
//                if (min_dist > dist) {
//                    min_dist = dist;
//                }
//            }
//        }
//        
//        score += (min_dist*min_dist);
//    }
//    
//    if(points->size() > 0){
//        score /= points->size();
//    }
//    
//    return score;
//}

/*
float growModelFitting(RoadPt& start_point,
                       PclPointCloud::Ptr& points,
                       PclSearchTree::Ptr& search_tree,
                       vector<RoadPt>& extension,
                       bool grow_backward){
    if (points->size() == 0) {
        return 0.0f;
    }
    
    bool is_oneway = start_point.is_oneway;
    float DELTA_LENGTH = 10.0f;
    
    // Initial guess using heuristics
    // Trace the road
    float search_radius = 25.0f;
    PclPoint search_pt;
    search_pt.setCoordinate(start_point.x, start_point.y, 0.0f);
    search_pt.head = start_point.head;
    
    extension.clear();
    extension.push_back(RoadPt(search_pt.x,
                               search_pt.y,
                               search_pt.head,
                               is_oneway));
    
    while (true) {
        Eigen::Vector3d search_pt_dir = headingTo3dVector(search_pt.head);
        Eigen::Vector3d search_pt_per_dir(-search_pt_dir[1], search_pt_dir[0], 0.0f);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        search_tree->radiusSearch(search_pt,
                                  search_radius,
                                  k_indices,
                                  k_dist_sqrs);
        
        // Heading Votes
        float delta_heading_bin = 30.0f;
        float max_heading = 120; // in degrees
        int N_BINS = floor(2 * max_heading / delta_heading_bin);
        
        vector<vector<int> > votes(N_BINS, vector<int>());
        vector<float> cum_vote_values(N_BINS, 0.0f);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = points->at(*it);
            Eigen::Vector3d vec(nb_pt.x - search_pt.x,
                                nb_pt.y - search_pt.y,
                                0.0f);
            
            float vec_length = vec.norm();
            if(vec_length < 1.0f){
                continue;
            }
            
            float dot_value = search_pt_dir.dot(vec) / vec_length;
            if(dot_value < -0.5f){
                continue;
            }
            
            float signed_delta_heading = deltaHeading1MinusHeading2(nb_pt.head, search_pt.head);
            float delta_heading = abs(signed_delta_heading);
            
            if (!is_oneway && delta_heading > 90.0f) {
                delta_heading = 180.0f - delta_heading;
                int tmp_head = (nb_pt.head + 180) % 360;
                signed_delta_heading = deltaHeading1MinusHeading2(tmp_head, search_pt.head);
            }
            
            delta_heading = abs(signed_delta_heading);
            if (delta_heading > max_heading) {
                continue;
            }
            
            int bin_idx = floor((max_heading + signed_delta_heading) / delta_heading_bin + 0.5f);
            
            if (bin_idx >= N_BINS || bin_idx < 0) {
                continue;
            }
            
            votes[bin_idx].push_back(*it);
            cum_vote_values[bin_idx] += signed_delta_heading;
        }
        
        int max_vote_count = 1;
        int max_idx = -1;
        for (int k = 0; k < votes.size(); ++k) {
            if (max_vote_count < votes[k].size()) {
                max_vote_count = votes[k].size();
                max_idx = k;
            }
        }
        
        if (max_idx == -1) {
            break;
        }
        
        vector<int> max_vote_bin = votes[max_idx];
        float cum_perp_proj = 0.0f;
        Eigen::Vector3d avg_dir(0.0f, 0.0f, 0.0f);
        int closeby_pt_count = 0;
        for (size_t k = 0; k < max_vote_bin.size(); ++k) {
            int pt_idx = max_vote_bin[k];
            PclPoint& nb_pt = points->at(pt_idx);
            Eigen::Vector3d vec(nb_pt.x - search_pt.x,
                                nb_pt.y - search_pt.y,
                                0.0f);
            float parallel_dist = abs(vec.dot(search_pt_dir));
            if (parallel_dist < 5.0f) {
                closeby_pt_count++;
            }
            
            Eigen::Vector3d nb_dir = headingTo3dVector(nb_pt.head);
            float perp_proj = search_pt_dir.cross(vec)[2];
            cum_perp_proj += perp_proj;
            float dot_value = search_pt_dir.dot(nb_dir);
            if (dot_value < 0) {
                avg_dir -= nb_dir;
            }
            else{
                avg_dir += nb_dir;
            }
        }
        
        int new_heading;
        int max_bin_delta_heading = floor(cum_vote_values[max_idx] / max_vote_bin.size() + 0.5f);
        new_heading = increaseHeadingBy(max_bin_delta_heading, search_pt.head);
        avg_dir = headingTo3dVector(new_heading);
        
        cum_perp_proj /= max_vote_bin.size();
        Eigen::Vector3d pt0(search_pt.x, search_pt.y, 0.0f);
        Eigen::Vector3d new_pt1 = pt0 + DELTA_LENGTH * search_pt_dir + cum_perp_proj * search_pt_per_dir;
        Eigen::Vector3d new_pt2 = pt0 + DELTA_LENGTH * avg_dir;
        Eigen::Vector3d new_pt = 0.5 * (new_pt1 + new_pt2);
        
        float delta_x = search_pt.x - new_pt.x();
        float delta_y = search_pt.y - new_pt.y();
        float delta_l = sqrt(delta_x*delta_x + delta_y*delta_y);
        if (delta_l < 3.0f) {
            break;
        }
        
        search_pt.x = new_pt.x();
        search_pt.y = new_pt.y();
        search_pt.head = new_heading;
        
        extension.push_back(RoadPt(search_pt.x,
                                   search_pt.y,
                                   search_pt.head,
                                   is_oneway));
    }
    
    smoothCurve(extension, true);
    
    
    // Store the result
    if (extension.size() < 2) {
        return POSITIVE_INFINITY;
    }
    
    // Compute score
    float score = 0.0f;
    for (size_t i = 0; i < points->size(); ++i) {
        PclPoint& pt = points->at(i);
        float min_dist = 1e10;
        for (int s = 1; s < extension.size(); ++s) {
            Eigen::Vector2d vec0(pt.x - extension[s-1].x, pt.y - extension[s-1].y);
            Eigen::Vector2d vec1(pt.x - extension[s].x, pt.y - extension[s].y);
            Eigen::Vector2d dir(extension[s].x - extension[s-1].x,
                                extension[s].y - extension[s-1].y);
            
            float dir_length = dir.norm();
            float length0 = vec0.norm();
            float length1 = vec1.norm();
            
            if (dir_length < 0.1f) {
                if (min_dist > length0) {
                    min_dist = length0;
                    continue;
                }
            }
            
            dir /= dir_length;
            float dot_value = dir.dot(vec0);
            
            if (dot_value < 0.0f) {
                if (min_dist > length0) {
                    min_dist = length0;
                }
            }
            else if(dot_value > dir_length) {
                if (min_dist > length1) {
                    min_dist = length1;
                }
            }
            else{
                float dist = sqrt(length0*length0 - dot_value*dot_value);
                if (min_dist > dist) {
                    min_dist = dist;
                }
            }
        }
        
        score += (min_dist*min_dist);
    }
    
    cout << "score is: " << score << endl;
    
    return score;
}
 */

//float branchFitting(RoadPt& start_point,
//                    PclPointCloud::Ptr& points,
//                    PclSearchTree::Ptr& search_tree,
//                    Trajectories*       trajectories,
//                    vector<vector<RoadPt> >& branches,
//                    bool grow_backward){
//    if (points->size() == 0) {
//        return 0.0f;
//    }
//    
//    branches.clear();
//    
//    bool is_oneway = start_point.is_oneway;
//    
//    float search_radius = Parameters::getInstance().searchRadius();
//    
//    // Move point around as skeleton extraction
//    PclPointCloud::Ptr tmp_points(new PclPointCloud);
//    PclSearchTree::Ptr tmp_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    
//    // Strip off the points consistent with the first branch
//    vector<float> tmp_point_scores;
//    for (size_t i = 0; i < points->size(); ++i) {
//        tmp_points->push_back(points->at(i));
//    }
//    
//    if(tmp_points->size() == 0){
//        return 0.0f;
//    }
//    
//    tmp_search_tree->setInputCloud(tmp_points);
//    
//    float HEADING_THRESHOLD = 15.0f;
//    float delta_bin = 2.5f; // in meters
//    float sigma_s = 5.0f; // in meters
//    for(int t = 0; t < 3; ++t){
//        for (size_t i = 0; i < tmp_points->size(); ++i) {
//            PclPoint& pt = tmp_points->at(i);
//            RoadPt r_pt(start_point);
//            r_pt.x = pt.x;
//            r_pt.y = pt.y;
//            r_pt.head = pt.head;
//            adjustRoadCenterAt(r_pt,
//                               points,
//                               search_tree,
//                               search_radius,
//                               HEADING_THRESHOLD,
//                               delta_bin,
//                               sigma_s,
//                               true);
//            pt.x = r_pt.x;
//            pt.y = r_pt.y;
//        }
//        
//        tmp_search_tree->setInputCloud(tmp_points);
//    }
//    
//    PclPointCloud::Ptr simplified_points(new PclPointCloud);
//    PclSearchTree::Ptr simplified_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    
//    vector<int> is_covered(tmp_points->size(), false);
//    for (size_t i = 0; i < tmp_points->size(); ++i) {
//        if (is_covered[i]) {
//            continue;
//        }
//        
//        is_covered[i] = true;
//        PclPoint& pt = tmp_points->at(i);
//        
//        int count = 0;
//        float cum_x = 0.0f;
//        float cum_y = 0.0f;
//        
//        vector<int> k_indices;
//        vector<float> k_dist_sqrs;
//        
//        tmp_search_tree->radiusSearch(pt, 10.0f, k_indices, k_dist_sqrs);
//        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//            PclPoint& nearby_pt = tmp_points->at(*it);
//            float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, pt.head));
//            
//            if (!is_oneway && delta_heading > 90.0f) {
//                delta_heading = 180.0f - delta_heading;
//            }
//            
//            if (delta_heading < HEADING_THRESHOLD) {
//                is_covered[*it] = true;
//                cum_x += (nearby_pt.id_sample * nearby_pt.x);
//                cum_y += (nearby_pt.id_sample * nearby_pt.y);
//                count += nearby_pt.id_sample;
//            }
//        }
//        
//        if(count > 0){
//            cum_x /= count;
//            cum_y /= count;
//            pt.x = cum_x;
//            pt.y = cum_y;
//        }
//        
//        simplified_points->push_back(pt);
//    }
//    
//    simplified_search_tree->setInputCloud(simplified_points);
//    
//    // Build a undirected graph
//    typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_distance_t, float>, property < edge_weight_t, float> > graph_t;
//    typedef typename graph_t::edge_property_type Weight;
//    
//    typedef graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
//    typedef graph_traits<graph_t>::edge_descriptor edge_descriptor;
//    typedef graph_traits<graph_t>::vertex_iterator vertex_iter;
//    typedef graph_traits<graph_t>::edge_iterator edge_iter;
//    typedef pair<int, int> E;
//    vector<float> edge_weights;
//    vector<E>     edge_list;
//    
//    PclPoint pt;
//    pt.setCoordinate(start_point.x, start_point.y, 0.0f);
//    pt.head = start_point.head;
//    
//    float ratio = 2.0f;
//    
//    // Add edges from start_point to simplified_points
//    vector<int> k_indices;
//    vector<float> k_dist_sqrs;
//    simplified_search_tree->radiusSearch(pt, ceil(ratio * search_radius), k_indices, k_dist_sqrs);
//    for (size_t i = 0; i != k_indices.size(); ++i) {
//        PclPoint nb_pt = simplified_points->at(k_indices[i]);
//        Eigen::Vector2d nb_pt_dir = headingTo2dVector(nb_pt.head);
//        
//        Eigen::Vector2d vec(nb_pt.x - start_point.x,
//                            nb_pt.y - start_point.y);
//        
//        float vec_length = vec.norm();
//        if(vec_length > 1e-3){
//            vec /= vec_length;
//        }
//        else{
//            vec *= 0.0f;
//        }
//        
//        float dot_value = vec.dot(nb_pt_dir);
//        
//        if (!is_oneway) {
//            dot_value = abs(dot_value);
//        }
//        
//        float edge_weight = sqrt(k_dist_sqrs[i]) * (2.0f - 1.0f * dot_value);
//        
//        edge_list.push_back(E(0, k_indices[i]+1));
//        edge_weights.push_back(edge_weight);
//    }
//    
//    // Add edges among simplified_points
//    for (size_t i = 0; i < simplified_points->size(); ++i) {
//        PclPoint& this_pt = simplified_points->at(i);
//        Eigen::Vector2d this_pt_dir = headingTo2dVector(this_pt.head);
//        
//        vector<int> k_inds;
//        vector<float> k_dists;
//        simplified_search_tree->radiusSearch(this_pt,
//                                             ceil(ratio * search_radius),
//                                             k_inds,
//                                             k_dists);
//        
//        for (size_t s = 0; s < k_inds.size(); ++s) {
//            if (k_inds[s] <= i) {
//                continue;
//            }
//            
//            PclPoint nb_pt = simplified_points->at(k_inds[s]);
//            Eigen::Vector2d nb_pt_dir = headingTo2dVector(nb_pt.head);
//           
//            Eigen::Vector2d vec(nb_pt.x - this_pt.x,
//                                nb_pt.y - this_pt.y);
//            
//            float vec_length = vec.norm();
//            if(vec_length > 1e-3){
//                vec /= vec_length;
//            }
//            else{
//                vec *= 0.0f;
//            }
//            
//            float dot_value1 = vec.dot(nb_pt_dir);
//            float dot_value2 = -1.0f * vec.dot(this_pt_dir);
//            
//            if (!is_oneway) {
//                dot_value1 = abs(dot_value1);
//                dot_value2 = abs(dot_value2);
//            }
//            
//            float dot_value = (dot_value1 > dot_value2) ? dot_value1 : dot_value2;
//            
//            float edge_weight = sqrt(k_dists[s]) * (2.0f - dot_value); //- dot_value1 - dot_value2);
//            
//            edge_list.push_back(E(i+1, k_inds[s]+1));
//            edge_weights.push_back(edge_weight);
//        }
//    }
//    
//    graph_t G;
//    
//    for(size_t i = 0; i < edge_list.size(); ++i){
//        add_edge(edge_list[i].first, edge_list[i].second, edge_weights[i], G);
//    }
//    
//    // Compute connected component
//    vector<int> component(num_vertices(G));
//    connected_components(G, &component[0]);
//    
//    vector<edge_descriptor> spanning_tree;
//    kruskal_minimum_spanning_tree(G, back_inserter(spanning_tree));;
//    
//    property_map<graph_t, edge_weight_t>::type w = get(edge_weight, G);
//    
//    // Construct the spanning tree
//    graph_t MST;
//    for (vector<edge_descriptor>::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
//        int source_idx = source(*ei, G);
//        int target_idx = target(*ei, G);
//        if (component[source_idx] != component[0]) {
//            continue;
//        }
//        
//        add_edge(source_idx, target_idx, get(w, *ei), MST);
//        
////        if (source_idx == 0) {
////            RoadPt r_pt(start_point);
////            branch.push_back(r_pt);
////        }
////        else{
////            PclPoint pt = simplified_points->at(source_idx - 1);
////            RoadPt r_pt;
////            r_pt.x = pt.x;
////            r_pt.y = pt.y;
////            r_pt.head = pt.head;
////            branch.push_back(r_pt);
////        }
////        
////        if (target_idx == 0) {
////            RoadPt r_pt(start_point);
////            branch.push_back(start_point);
////        }
////        else{
////            PclPoint pt = simplified_points->at(target_idx - 1);
////            RoadPt r_pt;
////            r_pt.x = pt.x;
////            r_pt.y = pt.y;
////            r_pt.head = pt.head;
////            branch.push_back(r_pt);
////        }
//    }
//    
//    int n_vertices = num_vertices(MST);
//    vector<int> depth(n_vertices, 0);
//    vector<int> parents(n_vertices, -1);
//    
//    float MIN_BRANCH_LENGTH = 30.0f;
//    
//    vector<vector<RoadPt> > raw_branches;
//    if (n_vertices > 0) {
//        dfs_depth_visitor vis(depth, parents);
//        
//        depth_first_search(MST, visitor(vis).root_vertex(0));
//        
//        int max_depth = -1;
//        int max_depth_idx = -1;
//        for(size_t i = 0; i < depth.size(); ++i){
//            if(depth[i] > max_depth){
//                max_depth = depth[i];
//                max_depth_idx = i;
//            }
//        }
//        
//        vector<vector<int> > paths;
//        vector<int> first_path;
//        vector<bool> marked(n_vertices, false);
//        first_path.push_back(max_depth_idx);
//        marked[max_depth_idx] = true;
//        int p = parents[max_depth_idx];
//        int cur_node = max_depth_idx;
//        while(p != -1){
//            cur_node = p;
//            first_path.insert(first_path.begin(), p);
//            marked[cur_node] = true;
//            p = parents[cur_node];
//        }
//        paths.push_back(first_path);
//        
//        int n_cur_path = 1;
//        while (true) {
//            float cum_max_branch_length = 0.0f;
//            
//            int cur_max_idx = -1;
//            for (int s = 0; s < n_vertices; ++s) {
//                if (marked[s]) {
//                    continue;
//                }
//                
//                // Find path
//                int cur_node = s;
//                int p = parents[s];
//                int cur_length = 0;
//                float cur_cum_length = 0.0f;
//                while(true){
//                    if(p == -1){
//                        break;
//                    }
//                    
//                    cur_length++;
//                   
//                    float start_x;
//                    float start_y;
//                    if (cur_node == 0) {
//                        start_x = start_point.x;
//                        start_y = start_point.y;
//                    }
//                    else{
//                        PclPoint& pt = simplified_points->at(cur_node-1);
//                        start_x = pt.x;
//                        start_y = pt.y;
//                    }
//                    
//                    float end_x;
//                    float end_y;
//                    
//                    if (p == 0) {
//                        end_x = start_point.x;
//                        end_y = start_point.y;
//                    }
//                    else{
//                        PclPoint& pt = simplified_points->at(p-1);
//                        end_x = pt.x;
//                        end_y = pt.y;
//                    }
//                    
//                    float delta_x = end_x - start_x;
//                    float delta_y = end_y - start_y;
//                    cur_cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
//                    
//                    if (marked[p]) {
//                        break;
//                    }
//                    
//                    cur_node = p;
//                    p = parents[p];
//                }
//                
//                if (cum_max_branch_length < cur_cum_length) {
//                    cum_max_branch_length = cur_cum_length;
//                    cur_max_idx = s;
//                }
//            }
//            
//            if (cum_max_branch_length < MIN_BRANCH_LENGTH) {
//                break;
//            }
//            
//            // Trace path
//            vector<int> this_path;
//            this_path.push_back(cur_max_idx);
//            marked[cur_max_idx] = true;
//            int p = parents[cur_max_idx];
//            while(p != -1){
//                this_path.insert(this_path.begin(), p);
//                
//                if (marked[p]) {
//                    break;
//                }
//                
//                marked[p] = true;
//                p = parents[p];
//            }
//            paths.push_back(this_path);
//            n_cur_path++;
//        
//            if(n_cur_path > 3){
//                break;
//            }
//        }
//        
//        for(size_t s = 0; s < paths.size(); ++s){
//            vector<int>& this_path = paths[s];
//            
//            vector<RoadPt> branch;
//            for(int k = 0; k < this_path.size() - 1; ++k){
//                int source_idx = this_path[k];
//                int target_idx = this_path[k+1];
//                if (source_idx == 0) {
//                    RoadPt r_pt(start_point);
//                    branch.push_back(r_pt);
//                }
//                else{
//                    PclPoint pt = simplified_points->at(source_idx - 1);
//                    RoadPt r_pt;
//                    r_pt.x = pt.x;
//                    r_pt.y = pt.y;
//                    r_pt.head = pt.head;
//                    branch.push_back(r_pt);
//                }
//                
//                if (target_idx == 0) {
//                    RoadPt r_pt(start_point);
//                    branch.push_back(start_point);
//                }
//                else{
//                    PclPoint pt = simplified_points->at(target_idx - 1);
//                    RoadPt r_pt;
//                    r_pt.x = pt.x;
//                    r_pt.y = pt.y;
//                    r_pt.head = pt.head;
//                    branch.push_back(r_pt);
//                }
//            }
//            raw_branches.push_back(branch);
////            branches.push_back(branch);
//        }
//    }
//    
//    if (raw_branches.size() == 0) {
//        return false;
//    }
//    
//    vector<bool> branch_valid(raw_branches.size(), false);
//    for (size_t s = 0; s < raw_branches.size(); ++s) {
//        
//    }
//    
//    float junc_x = 0.0f;
//    float junc_y = 0.0f;
//    int n_junc = 0;
//    if (raw_branches.size() > 2) {
//        junc_x = raw_branches[1][0].x;
//        junc_y = raw_branches[1][0].y;
//        n_junc = 1;
//        for (int s = 2; s < raw_branches.size(); ++s) {
//            if (roadPtDistance(raw_branches[1][0], raw_branches[s][0]) < 10.0f) {
//                junc_x += raw_branches[s][0].x;
//                junc_y += raw_branches[s][0].y;
//                ++n_junc;
//            }
//        }
//        junc_x /= n_junc;
//        junc_y /= n_junc;
//    }
//    
//    // update raw_branches
//    vector<vector<RoadPt> > updated_branches;
//    if (n_junc != 0) {
//        // move junction location
//        
//        // update each branch geometry
//    }
//    else{
//        // update the main branch geometry
//    }
//    
//    
//    float delta_growing_length = Parameters::getInstance().deltaGrowingLength();
//    float delta_heading_bin = 5.0f; // in degrees
//    float max_delta_heading_threshold = 45.0f;
//    if (raw_branches.size() > 1) {
//        RoadPt junc_loc(raw_branches[1][0]);
//        if(roadPtDistance(junc_loc, start_point) > 1.5f * delta_growing_length){
//            vector<RoadPt>& first_raw_branch = raw_branches[0];
//            // Resample first_raw_branch point heading
//            PclPoint pt;
//            for (size_t s = 0; s < first_raw_branch.size(); ++s) {
//                RoadPt& r_pt = first_raw_branch[s];
//                adjustRoadPtHeading(r_pt,
//                                    points,
//                                    search_tree,
//                                    search_radius,
//                                    max_delta_heading_threshold,
//                                    delta_heading_bin,
//                                    true);
//            }
//            
//            vector<RoadPt> branch;
//            float cum_length = 0.0f;
//            branch.push_back(first_raw_branch[0]);
//            for (int s = 1; s < first_raw_branch.size(); ++s) {
//                cum_length += roadPtDistance(first_raw_branch[s], first_raw_branch[s-1]);
//                if (cum_length > delta_growing_length) {
//                    break;
//                }
//                
//                branch.push_back(first_raw_branch[s]);
//            }
//            
//            smoothCurve(branch);
//            
//            branches.push_back(branch);
//        }
//        else{
//            // Refit branch model
//            if (raw_branches.size() == 2) {
//                // This may be a Y-split
//            }
//            else{
//                
//            }
//            
//            for (size_t s = 0; s < raw_branches.size(); ++s) {
//                branches.push_back(raw_branches[s]);
//            }
//        }
//    }
//    else{
//        vector<RoadPt> branch;
//        vector<RoadPt>& first_raw_branch = raw_branches[0];
//        float cum_length = 0.0f;
//        branch.push_back(first_raw_branch[0]);
//        for (int s = 1; s < first_raw_branch.size(); ++s) {
//            cum_length += roadPtDistance(first_raw_branch[s], first_raw_branch[s-1]);
//            if (cum_length > delta_growing_length) {
//                break;
//            }
//            
//            branch.push_back(first_raw_branch[s]);
//        }
//        
//        // Resample point heading
//        for (size_t s = 0; s < first_raw_branch.size(); ++s) {
//            RoadPt& r_pt = first_raw_branch[s];
//            
//            adjustRoadPtHeading(r_pt,
//                                points,
//                                search_tree,
//                                search_radius,
//                                max_delta_heading_threshold,
//                                delta_heading_bin,
//                                true);
//        }
//        
//        smoothCurve(branch);
//        
//        branches.push_back(branch);
//    }
//    
//    /*
//    vector<RoadPt> simp_points;
//    for(size_t i = 0; i < simplified_points->size(); ++i){
//        PclPoint& pt = simplified_points->at(i);
//        RoadPt r_pt(start_point);
//        r_pt.x = pt.x;
//        r_pt.y = pt.y;
//        r_pt.head = pt.head;
//        simp_points.push_back(r_pt);
//    }
//    
//    branches.push_back(simp_points);
//    vector<RoadPt> simp_points_dir;
//    for(size_t i = 0; i < simplified_points->size(); ++i){
//        PclPoint& pt = simplified_points->at(i);
//        RoadPt r_pt(start_point);
//        r_pt.x = pt.x;
//        r_pt.y = pt.y;
//        r_pt.head = pt.head;
//        simp_points_dir.push_back(r_pt);
//        
//        Eigen::Vector2d dir = headingTo2dVector(pt.head);
//       
//        RoadPt r_pt1(start_point);
//        r_pt1.x = pt.x + 10.0f * dir[0];
//        r_pt1.y = pt.y + 10.0f * dir[1];
//        r_pt1.head = pt.head;
//        simp_points_dir.push_back(r_pt1);
//    }
//    
//    branches.push_back(simp_points_dir);
//     */
//    
//    return 0.0f;
//}

bool tmpComparePairs(const pair<int, float>& v1, const pair<int, float>& v2){
    return v1.second < v2.second;
}

bool branchPrediction(const vector<RoadPt>&     road_centerline,
                      set<int>&           candidate_set,
                      Trajectories*       trajectories,
                      RoadPt&             junction_loc,
                      vector<vector<RoadPt> >&     branches,
                      bool                grow_backward){
    branches.clear();
    
    // Initialize point cloud
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    // Add points to point cloud
    map<int, int> pt_idx_mapping;
    for (set<int>::iterator it = candidate_set.begin(); it != candidate_set.end(); ++it) {
        PclPoint& pt = trajectories->data()->at(*it);
        pt_idx_mapping[*it] = points->size();
        points->push_back(pt);
    }
    
    if(points->size() == 0){
        return false;
    }
    
    search_tree->setInputCloud(points);
    
    branchFitting(road_centerline,
                  points,
                  search_tree,
                  trajectories,
                  branches);
    
    return true;
}

void tensor_decomposition(const Eigen::Matrix2d& T,
                          Eigen::Vector2d& e1,
                          Eigen::Vector2d& e2,
                          double& lambda1,
                          double& lambda2){
    double k11 = T(0, 0);
    double k12 = T(0, 1);
    double k22 = T(1, 1);
    
    double t = (k11 + k22) / 2;
    double a = k11 - t;
    double b = k12;
    
    double ab2 = sqrt(a*a + b*b);
    double l1 = t + ab2;
    double l2 = t - ab2;
    
    double y = ab2 - a;
    double x = b;
    double r = sqrt(x*x + y*y);
    
    double theta = 0.0f;
    if (r > 1e-5) {
        theta = acos(x / r);
    }
    else{
        lambda1 = 0.0f;
        lambda2 = 0.0f;
    }
    
    if (abs(l1) > abs(l2)) {
        e1[0] = cos(theta);
        e1[1] = sin(theta);
        e2[0] = -sin(theta);
        e2[1] = cos(theta);
        lambda1 = l1;
        lambda2 = l2;
    }
    else{
        e2[0] = cos(theta);
        e2[1] = sin(theta);
        e1[0] = -sin(theta);
        e1[1] = cos(theta);
        lambda1 = l2;
        lambda2 = l1;
    }
}

float branchFitting(const vector<RoadPt>&     centerline,
                    PclPointCloud::Ptr& points,
                    PclSearchTree::Ptr& search_tree,
                    Trajectories*       trajectories,
                    vector<vector<RoadPt> >& branches,
                    bool grow_backward){
    if (points->size() == 0) {
        return 0.0f;
    }
    
    branches.clear();
    
    PclPointCloud::Ptr simplified_points(new PclPointCloud);
    PclSearchTree::Ptr simplified_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    sampleGPSPoints(5.0f,
                    7.5f,
                    points,
                    search_tree,
                    simplified_points,
                    simplified_search_tree);
    
    // Road voting
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*simplified_points, min_pt, max_pt);
    min_pt[0] -= 10.0f;
    max_pt[0] += 10.0f;
    min_pt[1] -= 10.0f;
    max_pt[1] += 10.0f;
    
    PclPointCloud::Ptr grid_points(new PclPointCloud);
    PclSearchTree::Ptr grid_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    float delta = Parameters::getInstance().roadVoteGridSize();
    int n_x = floor((max_pt[0] - min_pt[0]) / delta + 0.5f) + 1;
    int n_y = floor((max_pt[1] - min_pt[1]) / delta + 0.5f) + 1;
    float sigma_h = Parameters::getInstance().roadSigmaH();
    float sigma_w = Parameters::getInstance().roadSigmaW();
    
    int half_search_window = floor(sigma_h / delta + 0.5f);
    int N_ANGLE_BINS = 24;
    map<int, vector<float> > grid_angle_votes;
    
    float max_vote = 0.0f;
    for (size_t i = 0; i < simplified_points->size(); ++i) {
        PclPoint& pt = simplified_points->at(i);
        Eigen::Vector2d pt_dir = headingTo2dVector(pt.head);
        int heading_bin_idx = pt.head / 15;
        
        int pt_i = floor((pt.x - min_pt[0]) / delta);
        int pt_j = floor((pt.y - min_pt[1]) / delta);
        
        for(int pi = pt_i - half_search_window; pi <= pt_i + half_search_window; ++pi){
            if (pi < 0) {
                continue;
            }
            if(pi >= n_x){
                continue;
            }
            for(int pj = pt_j - half_search_window; pj <= pt_j + half_search_window; ++pj){
                if (pj < 0) {
                    continue;
                }
                if(pj >= n_y){
                    continue;
                }
                
                int grid_pt_idx = pj + n_y * pi;
                
                float grid_pt_x = (pi + 0.5f) * delta + min_pt[0];
                float grid_pt_y = (pj + 0.5f) * delta + min_pt[1];
                
                Eigen::Vector2d vec(grid_pt_x - pt.x,
                                    grid_pt_y - pt.y);
                
                float dot_value_sqr = pow(vec.dot(pt_dir), 2.0f);
                float perp_value_sqr = vec.dot(vec) - dot_value_sqr;
                
                float vote = pt.id_sample *
                exp(-1.0f * dot_value_sqr / 2.0f / sigma_h / sigma_h) *
                exp(-1.0f * perp_value_sqr / 2.0f / sigma_w / sigma_w);
                
                if (vote > 1e-2) {
                    for (int s = heading_bin_idx - 1; s <= heading_bin_idx + 1; ++s) {
                        int corresponding_bin_idx = s;
                        if (s < 0) {
                            corresponding_bin_idx += N_ANGLE_BINS;
                        }
                        if(s >= N_ANGLE_BINS){
                            corresponding_bin_idx %= N_ANGLE_BINS;
                        }
                        
                        float bin_center = (corresponding_bin_idx + 0.5f) * 15.0f;
                        float delta_angle = deltaHeading1MinusHeading2(pt.head, bin_center);
                        float angle_base = exp(-1.0f * delta_angle * delta_angle / 2.0f / 15.0f / 15.0f);
                        
                        
                        if (grid_angle_votes.find(grid_pt_idx) == grid_angle_votes.end()) {
                            grid_angle_votes[grid_pt_idx] = vector<float>(N_ANGLE_BINS, 0.0f);
                        }
                        
                        grid_angle_votes[grid_pt_idx][corresponding_bin_idx] += vote * angle_base;
                        if(grid_angle_votes[grid_pt_idx][corresponding_bin_idx] > max_vote){
                            max_vote = grid_angle_votes[grid_pt_idx][corresponding_bin_idx];
                        }
                    }
                }
            }
        }
    }
    
    vector<float> grid_votes;
    
    // add original centerline to grid
    for(size_t i = 0; i < centerline.size(); ++i){
        PclPoint pt;
        pt.setCoordinate(centerline[i].x, centerline[i].y, 0.0f);
        pt.head = centerline[i].head;
        grid_points->push_back(pt);
        grid_votes.push_back(1.0f);
    }
    
    if(max_vote > 1e-3){
        for(map<int, vector<float> >::iterator it = grid_angle_votes.begin(); it != grid_angle_votes.end(); ++it){
            int grid_pt_idx = it->first;
            vector<float>& votes = it->second;
            int max_idx = -1;
            findMaxElement(votes, max_idx);
            
            if (max_idx != -1 && votes[max_idx] >= 0.005f) {
                PclPoint pt;
                
                int pt_i = grid_pt_idx / n_y;
                int pt_j = grid_pt_idx % n_y;
                float pt_x = (pt_i + 0.5f) * delta + min_pt[0];
                float pt_y = (pt_j + 0.5f) * delta + min_pt[1];
                pt.setCoordinate(pt_x, pt_y, 0.0f);
                pt.head = floor((max_idx + 0.5f) * 15.0f);
                pt.head %= 360;
                grid_points->push_back(pt);
                grid_votes.push_back(votes[max_idx] / max_vote);
            }
        }
    }
    
    if(grid_points->size() > 0){
        grid_point_search_tree->setInputCloud(grid_points);
    }
    
    // Generate paths
    float vote_threshold = Parameters::getInstance().roadVoteThreshold();
    vector<bool> is_local_maximum(grid_points->size(), false);
    for(size_t i = 0; i < centerline.size(); ++i){
        is_local_maximum[i] = true;
    }
    
    for (size_t i = centerline.size(); i < grid_points->size(); ++i) {
        if(grid_votes[i] < vote_threshold){
            continue;
        }
        PclPoint& g_pt = grid_points->at(i);
        Eigen::Vector2d dir = headingTo2dVector(g_pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        grid_point_search_tree->radiusSearch(g_pt, 15.0f, k_indices, k_dist_sqrs);
        bool is_lateral_max = true;
        for (size_t j = 0; j < k_indices.size(); ++j) {
            if (k_indices[j] == i) {
                continue;
            }
            
            PclPoint& nb_g_pt = grid_points->at(k_indices[j]);
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, g_pt.head));
            
            if (delta_heading > 7.5f) {
                continue;
            }
            
            Eigen::Vector2d vec(nb_g_pt.x - g_pt.x,
                                nb_g_pt.y - g_pt.y);
            
            if (abs(vec.dot(dir)) < 1.5f) {
                if (grid_votes[k_indices[j]] > grid_votes[i]) {
                    is_lateral_max = false;
                    break;
                }
            }
        }
        
        if (is_lateral_max) {
            is_local_maximum[i] = true;
        }
    }
    
    // Sort grid_votes_ with index
    vector<pair<int, float> > final_grid_votes;
    for(size_t i = 0; i < grid_votes.size(); ++i){
        final_grid_votes.push_back(pair<int, float>(i, grid_votes[i]));
    }
    
    sort(final_grid_votes.begin(), final_grid_votes.end(), pairCompareDescend);
    
    float stopping_threshold = Parameters::getInstance().roadVoteThreshold();
    
    typedef adjacency_list<vecS, vecS, undirectedS >    graph_t;
    typedef graph_traits<graph_t>::vertex_descriptor    vertex_descriptor;
    typedef graph_traits<graph_t>::edge_descriptor      edge_descriptor;
    
    graph_t G(final_grid_votes.size());
    
    {
        using namespace boost;
        float search_radius = 10.0f;
        vector<bool> grid_pt_visited(final_grid_votes.size(), false);
        
        // Add centerline path
        for (size_t i = 0; i < centerline.size() - 1; ++i) {
            add_edge(i, i+1, G);
        }
        
        
        for (int i = 0; i < final_grid_votes.size(); ++i) {
            int grid_pt_idx = final_grid_votes[i].first;
            if(!is_local_maximum[grid_pt_idx]){
                continue;
            }
            
            if (grid_pt_idx < centerline.size()) {
                continue;
            }
            
            grid_pt_visited[grid_pt_idx] = true;
            float grid_pt_vote = final_grid_votes[i].second;
            if(grid_pt_vote < stopping_threshold){
                break;
            }
            
            PclPoint& g_pt = grid_points->at(grid_pt_idx);
            Eigen::Vector2d g_pt_dir = headingTo2dVector(g_pt.head);
            
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            grid_point_search_tree->radiusSearch(g_pt,
                                                 search_radius,
                                                 k_indices,
                                                 k_dist_sqrs);
            
            // Find the best candidate to connect
            int best_fwd_candidate = -1;
            float closest_fwd_distance = 1e6;
            int best_bwd_candidate = -1;
            float closest_bwd_distance = 1e6;
            for(size_t k = 0; k < k_indices.size(); ++k){
                if(k_indices[k] == grid_pt_idx){
                    continue;
                }
                
                if(!grid_pt_visited[k_indices[k]]){
                    continue;
                }
                
                PclPoint& nb_g_pt = grid_points->at(k_indices[k]);
                
                float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, g_pt.head));
                if(delta_heading > 15.0f){
                    continue;
                }
                
                Eigen::Vector2d vec(nb_g_pt.x - g_pt.x,
                                    nb_g_pt.y - g_pt.y);
                
                float dot_value = vec.dot(g_pt_dir);
                float perp_dist = sqrt(vec.dot(vec) - dot_value*dot_value);
                
                if (dot_value > 0) {
                    float this_fwd_distance = dot_value + perp_dist;
                    if(closest_fwd_distance > this_fwd_distance){
                        closest_fwd_distance = this_fwd_distance;
                        best_fwd_candidate = k_indices[k];
                    }
                }
                
                if(dot_value < 0){
                    float this_bwd_distance = abs(dot_value) + perp_dist;
                    if (closest_bwd_distance > this_bwd_distance) {
                        closest_bwd_distance = this_bwd_distance;
                        best_bwd_candidate = k_indices[k];
                    }
                }
            }
            
            if (best_fwd_candidate != -1) {
                auto es = out_edges(best_fwd_candidate, G);
                int n_edge = 0;
                bool is_compatible = true;
                for (auto eit = es.first; eit != es.second; ++eit){
                    n_edge++;
                    int target_idx = target(*eit, G);
                    PclPoint& target_g_pt = grid_points->at(target_idx);
                    PclPoint& source_g_pt = grid_points->at(best_fwd_candidate);
                    Eigen::Vector2d edge_dir(target_g_pt.x - source_g_pt.x,
                                             target_g_pt.y - source_g_pt.y);
                    if (edge_dir.dot(g_pt_dir) < 0) {
                        is_compatible = false;
                    }
                }
                
                if (is_compatible && n_edge < 2) {
                    // Add edge
                    add_edge(grid_pt_idx, best_fwd_candidate, G);
                }
            }
            
            if (best_bwd_candidate != -1) {
                auto es = in_edges(best_bwd_candidate, G);
                int n_edge = 0;
                bool is_compatible = true;
                for (auto eit = es.first; eit != es.second; ++eit){
                    n_edge++;
                    int source_idx = source(*eit, G);
                    PclPoint& source_g_pt = grid_points->at(source_idx);
                    PclPoint& target_g_pt = grid_points->at(best_bwd_candidate);
                    Eigen::Vector2d edge_dir(target_g_pt.x - source_g_pt.x,
                                             target_g_pt.y - source_g_pt.y);
                    if (edge_dir.dot(g_pt_dir) < 0) {
                        is_compatible = false;
                    }
                }
                
                if (is_compatible && n_edge < 2) {
                    // Add edge
                    add_edge(grid_pt_idx, best_bwd_candidate, G);
                }
            }
        }
    }
    
    // Compute connected component
    vector<int> component(num_vertices(G));
    int num = connected_components(G, &component[0]);
    
    vector<vector<int> > clusters(num, vector<int>());
    for (int i = 0; i != component.size(); ++i){
        clusters[component[i]].push_back(i);
    }
    
    // Trace roads
    vector<vector<int > > sorted_clusters;
    vector<vector<RoadPt> > roads;
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<int>& cluster = clusters[i];
        if(cluster.size() < 10){
            continue;
        }
        
        vector<int> sorted_cluster;
        
        // Find source
        int source_idx = -1;
        for (size_t j = 0; j < cluster.size(); ++j) {
            source_idx = cluster[j];
            PclPoint& source_pt = grid_points->at(source_idx);
            if(out_degree(source_idx, G) == 1){
                // Check edge direction
                auto es = out_edges(source_idx, G);
                bool is_head = true;
                for (auto eit = es.first; eit != es.second; ++eit){
                    int target_idx = target(*eit, G);
                    PclPoint& target_pt = grid_points->at(target_idx);
                    Eigen::Vector2d vec(target_pt.x - source_pt.x,
                                        target_pt.y - source_pt.y);
                    Eigen::Vector2d source_pt_dir = headingTo2dVector(source_pt.head);
                    if (source_pt_dir.dot(vec) < 0.1f) {
                        is_head = false;
                    }
                }
                if(is_head){
                    break;
                }
            }
        }
        
        if(source_idx == -1){
            continue;
        }
        
        int cur_idx = source_idx;
        int last_idx = -1;
        while(true){
            sorted_cluster.push_back(cur_idx);
            auto es = out_edges(cur_idx, G);
            bool new_edge_discovered = false;
            for (auto eit = es.first; eit != es.second; ++eit){
                int target_idx = target(*eit, G);
                if(last_idx != target_idx){
                    last_idx = cur_idx;
                    cur_idx = target_idx;
                    new_edge_discovered = true;
                    break;
                }
            }
            if(!new_edge_discovered){
                break;
            }
        }
        
        sorted_clusters.push_back(sorted_cluster);
        // Generate roads
        vector<RoadPt> a_road;
        for(size_t j = 0; j < sorted_cluster.size(); ++j){
            PclPoint& pt = grid_points->at(sorted_cluster[j]);
            RoadPt r_pt(pt.x,
                        pt.y,
                        pt.head);
            
//            adjustRoadCenterAt(r_pt,
//                               simplified_points,
//                               simplified_search_tree,
//                               15.0f,
//                               7.5f,
//                               2.5f,
//                               2.5f,
//                               true);
            
            a_road.push_back(r_pt);
        }
        
        smoothCurve(a_road);
        // Check road length
        float cum_length = 0.0f;
        for(size_t j = 1; j < a_road.size(); ++j){
            float delta_x = a_road[j].x - a_road[j-1].x;
            float delta_y = a_road[j].y - a_road[j-1].y;
            
            cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
        }
        
        if(cum_length >= 50.0f){
            roads.push_back(a_road);
        }
    }
    
    /*
     // Visualize voting field of the grid_points
    vector<RoadPt> branch;
    for (size_t i = 0; i < grid_points->size(); ++i) {
        RoadPt r_pt;
        PclPoint& g_pt = grid_points->at(i);
        r_pt.x = g_pt.x;
        r_pt.y = g_pt.y;
        r_pt.n_lanes = floor(grid_votes[i] * 100.0f);
        
        branch.push_back(r_pt);
    }
    branches.push_back(branch);
     */
    
    for (size_t i = 0; i < roads.size(); ++i) {
        branches.push_back(roads[i]);
    }
    
    return 0.0f;
}

//float branchFitting(RoadPt& start_point,
//                    PclPointCloud::Ptr& points,
//                    PclSearchTree::Ptr& search_tree,
//                    Trajectories*       trajectories,
//                    vector<vector<RoadPt> >& branches,
//                    bool grow_backward){
//    if (points->size() == 0) {
//        return 0.0f;
//    }
//    
//    branches.clear();
//    
//    bool is_oneway = start_point.is_oneway;
//    
//    PclPointCloud::Ptr simplified_points(new PclPointCloud);
//    PclSearchTree::Ptr simplified_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    float search_radius = Parameters::getInstance().searchRadius();
//    
//    float HEADING_THRESHOLD = 15.0f;
//    float delta_perp_bin = 2.5f; // in meters
//    float sigma_perp_s = 5.0f; // in meters
//    float delta_head_bin = 5.0f;
//    float sigma_head_s = 5.0f;
//    int   min_n = 1;
//    sampleRoadSkeletonPoints(search_radius,
//                             HEADING_THRESHOLD,
//                             delta_perp_bin,
//                             sigma_perp_s,
//                             delta_head_bin,
//                             sigma_head_s,
//                             min_n,
//                             is_oneway,
//                             points,
//                             search_tree,
//                             simplified_points,
//                             simplified_search_tree);
//    if (simplified_points->size() == 0) {
//        return 0.0f;
//    }
//    
//    vector<RoadPt> simp_points;
//    for(size_t i = 0; i < simplified_points->size(); ++i){
//        PclPoint& pt = simplified_points->at(i);
//        RoadPt r_pt(start_point);
//        r_pt.x = pt.x;
//        r_pt.y = pt.y;
//        r_pt.head = pt.head;
//        simp_points.push_back(r_pt);
//    }
//    
//    branches.push_back(simp_points);
//    vector<RoadPt> simp_points_dir;
//    for(size_t i = 0; i < simplified_points->size(); ++i){
//        PclPoint& pt = simplified_points->at(i);
//        RoadPt r_pt(start_point);
//        r_pt.x = pt.x;
//        r_pt.y = pt.y;
//        r_pt.head = pt.head;
//        simp_points_dir.push_back(r_pt);
//        
//        Eigen::Vector2d dir = headingTo2dVector(pt.head);
//        
//        RoadPt r_pt1(start_point);
//        r_pt1.x = pt.x + 10.0f * dir[0];
//        r_pt1.y = pt.y + 10.0f * dir[1];
//        r_pt1.head = pt.head;
//        simp_points_dir.push_back(r_pt1);
//    }
//    
//    branches.push_back(simp_points_dir);
//    
//    // Build a undirected graph
//    typedef adjacency_list<vecS, vecS, undirectedS, property<vertex_distance_t, float>, property < edge_weight_t, float> > graph_t;
//    typedef typename graph_t::edge_property_type Weight;
//    
//    typedef graph_traits<graph_t>::vertex_descriptor vertex_descriptor;
//    typedef graph_traits<graph_t>::edge_descriptor edge_descriptor;
//    typedef graph_traits<graph_t>::vertex_iterator vertex_iter;
//    typedef graph_traits<graph_t>::edge_iterator edge_iter;
//    typedef pair<int, int> E;
//    vector<float> edge_weights;
//    vector<E>     edge_list;
//    
//    PclPoint pt;
//    pt.setCoordinate(start_point.x, start_point.y, 0.0f);
//    pt.head = start_point.head;
//    
//    float ratio = 1.0f;
//    
//    vector<int> k_indices;
//    vector<float> k_dist_sqrs;
//    simplified_search_tree->radiusSearch(pt, ceil(ratio * search_radius), k_indices, k_dist_sqrs);
//    float high_angle_penalty = 2.0f;
//    // Add edges from start_point to simplified_points
//    for (size_t i = 0; i != k_indices.size(); ++i) {
//        PclPoint nb_pt = simplified_points->at(k_indices[i]);
//        Eigen::Vector2d nb_pt_dir = headingTo2dVector(nb_pt.head);
//        
//        Eigen::Vector2d vec(nb_pt.x - start_point.x,
//                            nb_pt.y - start_point.y);
//        
//        float vec_length = vec.norm();
//        if(vec_length > 1e-3){
//            vec /= vec_length;
//        }
//        else{
//            vec *= 0.0f;
//        }
//        
//        float dot_value = vec.dot(nb_pt_dir);
//        
//        if (is_oneway) {
//            if (dot_value > -0.1f) {
//                float edge_weight = sqrt(k_dist_sqrs[i]) * (1.0f + high_angle_penalty * (1.0f - dot_value));
//                edge_list.push_back(E(0, k_indices[i]+1));
//                edge_weights.push_back(edge_weight);
//            }
//        }
//        else{
//            dot_value = abs(dot_value);
//            float edge_weight = sqrt(k_dist_sqrs[i]) * (1.0f + high_angle_penalty * (1.0f - dot_value));
//            
//            edge_list.push_back(E(0, k_indices[i]+1));
//            edge_weights.push_back(edge_weight);
//        }
//    }
//    
//    // Add edges among simplified_points
//    for (size_t i = 0; i < simplified_points->size(); ++i) {
//        PclPoint& this_pt = simplified_points->at(i);
//        Eigen::Vector2d this_pt_dir = headingTo2dVector(this_pt.head);
//        
//        vector<int> k_inds;
//        vector<float> k_dists;
//        simplified_search_tree->radiusSearch(this_pt,
//                                             ceil(ratio * search_radius),
//                                             k_inds,
//                                             k_dists);
//        
//        for (size_t s = 0; s < k_inds.size(); ++s) {
//            if (k_inds[s] <= i) {
//                continue;
//            }
//            
//            PclPoint nb_pt = simplified_points->at(k_inds[s]);
//            Eigen::Vector2d nb_pt_dir = headingTo2dVector(nb_pt.head);
//            
//            Eigen::Vector2d vec(nb_pt.x - this_pt.x,
//                                nb_pt.y - this_pt.y);
//            
//            float vec_length = vec.norm();
//            if(vec_length > 1e-3){
//                vec /= vec_length;
//            }
//            else{
//                vec *= 0.0f;
//            }
//            
//            float d_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, this_pt.head));
//            if (d_heading > 135) {
//                continue;
//            }
//            
//            float dot_value1 = abs(vec.dot(nb_pt_dir));
//            float dot_value2 = abs(vec.dot(this_pt_dir));
//            float avg_dot_value = 0.5f * (dot_value1 + dot_value2);
//            
//            float edge_weight = sqrt(k_dists[s]) * (1.0f + high_angle_penalty * (1.0f - avg_dot_value));
//            
//            edge_list.push_back(E(i+1, k_inds[s]+1));
//            edge_weights.push_back(edge_weight);
//            
//        }
//    }
//    
//    graph_t G;
//    
//    for(size_t i = 0; i < edge_list.size(); ++i){
//        add_edge(edge_list[i].first, edge_list[i].second, edge_weights[i], G);
//    }
//    
//    // Compute connected component
//    vector<int> component(num_vertices(G));
//    connected_components(G, &component[0]);
//    
//    vector<edge_descriptor> spanning_tree;
//    kruskal_minimum_spanning_tree(G, back_inserter(spanning_tree));;
//    property_map<graph_t, edge_weight_t>::type w = get(edge_weight, G);
//    
//    // Construct the spanning tree
//    graph_t MST;
//    vector<RoadPt> branch;
//    for (vector<edge_descriptor>::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei) {
//        int source_idx = source(*ei, G);
//        int target_idx = target(*ei, G);
//        
//        if (component[source_idx] != component[0]) {
//            continue;
//        }
//        
//        add_edge(source_idx, target_idx, get(w, *ei), MST);
//        
//        if (source_idx == 0) {
//            RoadPt r_pt(start_point);
//            branch.push_back(r_pt);
//        }
//        else{
//            PclPoint pt = simplified_points->at(source_idx - 1);
//            RoadPt r_pt;
//            r_pt.x = pt.x;
//            r_pt.y = pt.y;
//            r_pt.head = pt.head;
//            branch.push_back(r_pt);
//        }
//        
//        if (target_idx == 0) {
//            RoadPt r_pt(start_point);
//            branch.push_back(start_point);
//        }
//        else{
//            PclPoint pt = simplified_points->at(target_idx - 1);
//            RoadPt r_pt;
//            r_pt.x = pt.x;
//            r_pt.y = pt.y;
//            r_pt.head = pt.head;
//            branch.push_back(r_pt);
//        }
//    }
//    
//    branches.push_back(branch);
//    
//    return 0.0f;
//    
//    int n_vertices = num_vertices(MST);
//    vector<int> depth(n_vertices, 0);
//    vector<int> parents(n_vertices, -1);
//    
//    float MIN_BRANCH_LENGTH = 30.0f;
//    
//    vector<vector<RoadPt> > raw_branches;
//    if (n_vertices > 0) {
//        dfs_depth_visitor vis(depth, parents);
//        
//        depth_first_search(MST, visitor(vis).root_vertex(0));
//        
//        int max_depth = -1;
//        int max_depth_idx = -1;
//        for(size_t i = 0; i < depth.size(); ++i){
//            if(depth[i] > max_depth){
//                max_depth = depth[i];
//                max_depth_idx = i;
//            }
//        }
//        
//        vector<vector<int> > paths;
//        vector<int> first_path;
//        vector<bool> marked(n_vertices, false);
//        first_path.push_back(max_depth_idx);
//        marked[max_depth_idx] = true;
//        int p = parents[max_depth_idx];
//        int cur_node = max_depth_idx;
//        while(p != -1){
//            cur_node = p;
//            first_path.insert(first_path.begin(), p);
//            marked[cur_node] = true;
//            p = parents[cur_node];
//        }
//        paths.push_back(first_path);
//        
//        int n_cur_path = 1;
//        while (true) {
//            float cum_max_branch_length = 0.0f;
//            
//            int cur_max_idx = -1;
//            for (int s = 0; s < n_vertices; ++s) {
//                if (marked[s]) {
//                    continue;
//                }
//                
//                // Find path
//                int cur_node = s;
//                int p = parents[s];
//                int cur_length = 0;
//                float cur_cum_length = 0.0f;
//                while(true){
//                    if(p == -1){
//                        break;
//                    }
//                    
//                    cur_length++;
//                    
//                    float start_x;
//                    float start_y;
//                    if (cur_node == 0) {
//                        start_x = start_point.x;
//                        start_y = start_point.y;
//                    }
//                    else{
//                        PclPoint& pt = simplified_points->at(cur_node-1);
//                        start_x = pt.x;
//                        start_y = pt.y;
//                    }
//                    
//                    float end_x;
//                    float end_y;
//                    
//                    if (p == 0) {
//                        end_x = start_point.x;
//                        end_y = start_point.y;
//                    }
//                    else{
//                        PclPoint& pt = simplified_points->at(p-1);
//                        end_x = pt.x;
//                        end_y = pt.y;
//                    }
//                    
//                    float delta_x = end_x - start_x;
//                    float delta_y = end_y - start_y;
//                    cur_cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
//                    
//                    if (marked[p]) {
//                        break;
//                    }
//                    
//                    cur_node = p;
//                    p = parents[p];
//                }
//                
//                if (cum_max_branch_length < cur_cum_length) {
//                    cum_max_branch_length = cur_cum_length;
//                    cur_max_idx = s;
//                }
//            }
//            
//            if (cum_max_branch_length < MIN_BRANCH_LENGTH) {
//                break;
//            }
//            
//            // Trace path
//            vector<int> this_path;
//            this_path.push_back(cur_max_idx);
//            marked[cur_max_idx] = true;
//            int p = parents[cur_max_idx];
//            while(p != -1){
//                this_path.insert(this_path.begin(), p);
//                
//                if (marked[p]) {
//                    break;
//                }
//                
//                marked[p] = true;
//                p = parents[p];
//            }
//            paths.push_back(this_path);
//            n_cur_path++;
//            
//            if(n_cur_path > 3){
//                break;
//            }
//        }
//        
//        for(size_t s = 0; s < paths.size(); ++s){
//            vector<int>& this_path = paths[s];
//            
//            vector<RoadPt> branch;
//            for(int k = 0; k < this_path.size() - 1; ++k){
//                int source_idx = this_path[k];
//                int target_idx = this_path[k+1];
//                if (source_idx == 0) {
//                    RoadPt r_pt(start_point);
//                    branch.push_back(r_pt);
//                }
//                else{
//                    PclPoint pt = simplified_points->at(source_idx - 1);
//                    RoadPt r_pt;
//                    r_pt.x = pt.x;
//                    r_pt.y = pt.y;
//                    r_pt.head = pt.head;
//                    branch.push_back(r_pt);
//                }
//                
//                if (target_idx == 0) {
//                    RoadPt r_pt(start_point);
//                    branch.push_back(start_point);
//                }
//                else{
//                    PclPoint pt = simplified_points->at(target_idx - 1);
//                    RoadPt r_pt;
//                    r_pt.x = pt.x;
//                    r_pt.y = pt.y;
//                    r_pt.head = pt.head;
//                    branch.push_back(r_pt);
//                }
//            }
//            raw_branches.push_back(branch);
//            //branches.push_back(branch);
//        }
//    }
//    
//    if (raw_branches.size() == 0) {
//        return 0.0f;
//    }
//    
//    vector<vector<RoadPt> > updated_branches;
//    // Remove abrupt turns
//    for (size_t i = 0; i < raw_branches.size(); ++i) {
//        vector<RoadPt>& cur_raw_branch = raw_branches[i];
//        vector<RoadPt> new_branch;
//        new_branch.push_back(cur_raw_branch[0]);
//    
//        float cum_length = 0.0f;
//        int last_recorded_idx = 0;
//        int s = 1;
//        Eigen::Vector2d last_dir = headingTo2dVector(cur_raw_branch[0].head);
//        while (s < cur_raw_branch.size()) {
//            Eigen::Vector2d vec(cur_raw_branch[s].x - cur_raw_branch[last_recorded_idx].x,
//                                cur_raw_branch[s].y - cur_raw_branch[last_recorded_idx].y);
//            if (last_dir.dot(vec) < -0.1f) {
//                ++s;
//                continue;
//            }
//            else{
//                // Insert
//                last_dir = vec;
//                new_branch.push_back(cur_raw_branch[s]);
//                cum_length += roadPtDistance(cur_raw_branch[s], cur_raw_branch[last_recorded_idx]);
//                last_recorded_idx = s;
//                ++s;
//            }
//        }
//        if (cum_length > 25.0f) {
//            updated_branches.push_back(new_branch);
//        }
//    }
//    vector<bool> updated_branch_is_valid(raw_branches.size(), true);
//    
//    float junc_x = 0.0f;
//    float junc_y = 0.0f;
//    int n_junc = 0;
//    if (updated_branches.size() > 1) {
//        junc_x = updated_branches[1][0].x;
//        junc_y = updated_branches[1][0].y;
//        n_junc = 1;
//        for (int s = 2; s < updated_branches.size(); ++s) {
//            if (roadPtDistance(updated_branches[1][0], updated_branches[s][0]) < 10.0f) {
//                junc_x += updated_branches[s][0].x;
//                junc_y += updated_branches[s][0].y;
//                ++n_junc;
//            }
//            else{
//                updated_branch_is_valid[s] = false;
//            }
//        }
//        junc_x /= n_junc;
//        junc_y /= n_junc;
//    }
//    
//    // update junction location
//    if (n_junc > 0) {
//        int i_iter = 0;
//        int max_iter = 100;
//        float cur_score = 0.0f;
//        for (int s = 0; s < updated_branches.size(); ++s) {
//            if (!updated_branch_is_valid[s]) {
//                continue;
//            }
//            vector<RoadPt>& cur_raw_branch = updated_branches[s];
//            for (int t = 0; t < cur_raw_branch.size(); ++t) {
//                float d_x_to_junc = cur_raw_branch[t].x - junc_x;
//                float d_y_to_junc = cur_raw_branch[t].y - junc_y;
//                float dist_to_junc = sqrt(d_x_to_junc*d_x_to_junc + d_y_to_junc*d_y_to_junc);
//                if (dist_to_junc > 50.0f && dist_to_junc < 100.0f) {
//                    Eigen::Vector2d vec1(d_x_to_junc, d_y_to_junc);
//                    Eigen::Vector2d pt_dir = headingTo2dVector(cur_raw_branch[t].head);
//                    float dot_value = abs(vec1.dot(pt_dir));
//                    cur_score += 1.0f - dot_value;
//                }
//            }
//        }
//        while (i_iter < max_iter) {
//            float score = 0.0f;
//            
//            float dx = 2.0f * double(rand()) / RAND_MAX - 1.0f;
//            float dy = 2.0f * double(rand()) / RAND_MAX - 1.0f;
//            
//            float new_junc_x = junc_x + dx;
//            float new_junc_y = junc_y + dy;
//            
//            int count = 0;
//            for (int s = 0; s < updated_branches.size(); ++s) {
//                if (!updated_branch_is_valid[s]) {
//                    continue;
//                }
//                vector<RoadPt>& cur_raw_branch = updated_branches[s];
//                for (int t = 0; t < cur_raw_branch.size(); ++t) {
//                    float d_x_to_junc = cur_raw_branch[t].x - new_junc_x;
//                    float d_y_to_junc = cur_raw_branch[t].y - new_junc_y;
//                    float dist_to_junc = sqrt(d_x_to_junc*d_x_to_junc + d_y_to_junc*d_y_to_junc);
//                    if (dist_to_junc > 50.0f && dist_to_junc < 100.0f) {
//                        ++count;
//                        Eigen::Vector2d vec1(d_x_to_junc, d_y_to_junc);
//                        Eigen::Vector2d pt_dir = headingTo2dVector(cur_raw_branch[t].head);
//                        float dot_value = abs(vec1.dot(pt_dir));
//                        score += 1.0f - dot_value;
//                    }
//                }
//            }
//            
//            if (count == 0) {
//                score = 1e10;
//            }
//            
//            if (score < cur_score) {
//                cur_score = score;
//                junc_x = new_junc_x;
//                junc_y = new_junc_y;
//            }
//            
//            ++i_iter;
//        }
//    }
//    
////    for (size_t s = 0; s < updated_branches.size(); ++s) {
////        if (updated_branch_is_valid[s]) {
////            branches.push_back(updated_branches[s]);
////        }
////    }
//    
//        for (size_t s = 0; s < raw_branches.size(); ++s) {
//            branches.push_back(raw_branches[s]);
//        }
//    
////    vector<RoadPt> simp_points;
//    for(size_t i = 0; i < simplified_points->size(); ++i){
//        PclPoint& pt = simplified_points->at(i);
//        RoadPt r_pt(start_point);
//        r_pt.x = pt.x;
//        r_pt.y = pt.y;
//        r_pt.head = pt.head;
//        simp_points.push_back(r_pt);
//    }
//    
////    vector<RoadPt> b;
////    RoadPt p(start_point);
////    p.x = junc_x;
////    p.y = junc_y;
////    b.push_back(p);
////    branches.push_back(b);
//    
//    branches.push_back(simp_points);
////    vector<RoadPt> simp_points_dir;
//    for(size_t i = 0; i < simplified_points->size(); ++i){
//        PclPoint& pt = simplified_points->at(i);
//        RoadPt r_pt(start_point);
//        r_pt.x = pt.x;
//        r_pt.y = pt.y;
//        r_pt.head = pt.head;
//        simp_points_dir.push_back(r_pt);
//        
//        Eigen::Vector2d dir = headingTo2dVector(pt.head);
//        
//        RoadPt r_pt1(start_point);
//        r_pt1.x = pt.x + 10.0f * dir[0];
//        r_pt1.y = pt.y + 10.0f * dir[1];
//        r_pt1.head = pt.head;
//        simp_points_dir.push_back(r_pt1);
//    }
//    
//    branches.push_back(simp_points_dir);
//    
//    return 0.0f;
//    
//    float delta_growing_length = Parameters::getInstance().deltaGrowingLength();
//    float delta_heading_bin = 5.0f; // in degrees
//    float max_delta_heading_threshold = 45.0f;
//    if (raw_branches.size() > 1) {
//        RoadPt junc_loc(raw_branches[1][0]);
//        if(roadPtDistance(junc_loc, start_point) > 1.5f * delta_growing_length){
//            vector<RoadPt>& first_raw_branch = raw_branches[0];
//            
//            vector<RoadPt> branch;
//            float cum_length = 0.0f;
//            branch.push_back(first_raw_branch[0]);
//            int last_recorded_idx = 0;
//            int s = 1;
//            Eigen::Vector2d last_dir = headingTo2dVector(first_raw_branch[0].head);
//            while (s < first_raw_branch.size()) {
//                Eigen::Vector2d vec(first_raw_branch[s].x - first_raw_branch[last_recorded_idx].x,
//                                    first_raw_branch[s].y - first_raw_branch[last_recorded_idx].y);
//                if (last_dir.dot(vec) < 0) {
//                    ++s;
//                    continue;
//                }
//                else{
//                    // Insert
//                    last_dir = vec;
//                    branch.push_back(first_raw_branch[s]);
//                    cum_length += roadPtDistance(first_raw_branch[s], first_raw_branch[last_recorded_idx]);
//                    last_recorded_idx = s;
//                    ++s;
//                }
//                
//                if (cum_length > delta_growing_length) {
//                    break;
//                }
//            }
//            
//            smoothCurve(branch);
//            
//            branches.push_back(branch);
//        }
//        else{
//            // Now comes the real branch
//            for (size_t s = 0; s < raw_branches.size(); ++s) {
//                branches.push_back(raw_branches[s]);
//            }
//        }
//    }
//    else{
//        vector<RoadPt> branch;
//        vector<RoadPt>& first_raw_branch = raw_branches[0];
//        float cum_length = 0.0f;
//        branch.push_back(first_raw_branch[0]);
//        for (int s = 1; s < first_raw_branch.size(); ++s) {
//            cum_length += roadPtDistance(first_raw_branch[s], first_raw_branch[s-1]);
//            if (cum_length > delta_growing_length) {
//                break;
//            }
//            
//            branch.push_back(first_raw_branch[s]);
//        }
//        
//        // Resample point heading
//        for (size_t s = 0; s < first_raw_branch.size(); ++s) {
//            RoadPt& r_pt = first_raw_branch[s];
//            
//            adjustRoadPtHeading(r_pt,
//                                points,
//                                search_tree,
//                                search_radius,
//                                max_delta_heading_threshold,
//                                delta_heading_bin,
//                                true);
//        }
//        
//        smoothCurve(branch);
//        
//        branches.push_back(branch);
//    }
//    
//    return 0.0f;
//}

//bool branchPrediction(RoadPt&             start_point,
//                      set<int>&           candidate_set,
//                      map<int, bool>&     traj_aligned_with_road,
//                      Trajectories*       trajectories,
//                      RoadPt&             junction_loc,
//                      vector<vector<RoadPt> >&     branches,
//                      bool                grow_backward){
//    branches.clear();
//    // Initialize point cloud
//    // Initialize point cloud
//    PclPointCloud::Ptr orig_points(new PclPointCloud);
//    PclSearchTree::Ptr orig_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    
//    // Add points to point cloud
//    map<int, int> pt_idx_mapping;
//    for (set<int>::iterator it = candidate_set.begin(); it != candidate_set.end(); ++it) {
//        PclPoint& pt = trajectories->data()->at(*it);
//        pt_idx_mapping[*it] = orig_points->size();
//        orig_points->push_back(pt);
//    }
//    
//    if(orig_points->size() == 0){
//        return false;
//    }
//    
//    orig_search_tree->setInputCloud(orig_points);
//    
//    PclPointCloud::Ptr points(new PclPointCloud);
//    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    
//    vector<int> is_covered(orig_points->size(), false);
//    
//    // Point heading will be aligned to the maximum histogram bin
//    for (size_t i = 0; i < orig_points->size(); ++i) {
//        if (is_covered[i]) {
//            continue;
//        }
//        
//        PclPoint& pt = orig_points->at(i);
//        is_covered[i] = true;
//        
//        float pt_speed = pt.speed / 100.0f;
//        if(pt_speed < 4.0f){
//            continue;
//        }
//        
//        vector<int> k_indices;
//        vector<float> k_dist_sqrs;
//        
//        set<int> traj_support;
//        orig_search_tree->radiusSearch(pt, 5.0f, k_indices, k_dist_sqrs);
//        
//        // Mark corresponding nearby points
//        int weight = 0;
//        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//            PclPoint& nearby_pt = orig_points->at(*it);
//            float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, pt.head));
//            if (delta_heading < 7.5f) {
//                weight++;
//                is_covered[*it] = true;
//            }
//        }
//        
//        pt.id_sample = weight;
//        points->push_back(pt);
//    }
//    
//    if(points->size() == 0){
//        return false;
//    }
//    
//    search_tree->setInputCloud(points);
//    
//    bool is_oneway = start_point.is_oneway;
//    if (is_oneway) {
//        // Branching for oneway
//        vector<RoadPt> grow_extension;
////        float grow_score = growModelFitting(start_point, points, search_tree, grow_extension, false);
////        cout << "Grow score: " << grow_score << endl;
//        
//        branchFitting(start_point,
//                      points,
//                      search_tree,
//                      trajectories,
//                      branches);
//    }
//    else{
//        vector<RoadPt> grow_extension;
////        float grow_score = growModelFitting(start_point, points, search_tree, grow_extension, false);
////        cout << "Grow score: " << grow_score << endl;
//        
//        branchFitting(start_point,
//                      points,
//                      search_tree,
//                      trajectories,
//                      branches);
//    }
//    
//    if(branches.size() > 1){
//        return true;
//    }
//    else{
//        return false;
//    }
//}

//bool branchPrediction(float               search_radius,
//                      RoadPt&             start_point,
//                      set<int>&           candidate_set,
//                      Trajectories*       trajectories,
//                      RoadPt&             junction_loc,
//                      vector<vector<RoadPt> >&     branches,
//                      bool                grow_backward){
//    
//    /*
//     Predict if we need to do branch.
//     - return value:
//     true: branch
//     false: grow
//     */
//    
//    if(candidate_set.size() == 0){
//        return false;
//    }
//    
//    bool is_oneway = start_point.is_oneway;
//    
//    // Initialize point cloud
//    PclPointCloud::Ptr points(new PclPointCloud);
//    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
//    
//    // Add points to point cloud
//    for (set<int>::iterator it = candidate_set.begin(); it != candidate_set.end(); ++it) {
//        PclPoint& pt = trajectories->data()->at(*it);
//        points->push_back(pt);
//    }
//    
//    search_tree->setInputCloud(points);
//    
//    vector<int> is_covered(points->size(), false);
//    
//    float delta = 10.0f; // in meters
//    // Trace the first branch
//    vector<RoadPt> first_branch;
//    PclPoint search_pt;
//    search_pt.setCoordinate(start_point.x, start_point.y, 0.0f);
//    search_pt.head = start_point.head;
//    
//    first_branch.push_back(RoadPt(search_pt.x,
//                                  search_pt.y,
//                                  search_pt.head,
//                                  is_oneway));
//    while (true) {
//        Eigen::Vector3d start_dir = headingTo3dVector(search_pt.head);
//        Eigen::Vector3d start_perp_dir(-start_dir[1], start_dir[0], 0.0f);
//        Eigen::Vector3d prev_search_pt(search_pt.x, search_pt.y, 0.0f);
//        Eigen::Vector3d tmp_search_pt = prev_search_pt + delta * start_dir;
//        
//        search_pt.x = tmp_search_pt.x();
//        search_pt.y = tmp_search_pt.y();
//        
//        vector<int> k_indices;
//        vector<float> k_dist_sqrs;
//        search_tree->radiusSearch(search_pt,
//                                  search_radius,
//                                  k_indices,
//                                  k_dist_sqrs);
//        
//        int pt_count = 0;
//        int closeby_pt_count = 0;
//        Eigen::Vector3d avg_dir(0.0f, 0.0f, 0.0f);
//        float cum_perp_proj = 0.0f;
//        float delta_bin = 5.0f;
//        int N_BINS = 2 * floor(25.0f / delta_bin);
//        vector<float> votes(N_BINS, 0.0f);
//        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//            PclPoint& nb_pt = points->at(*it);
//            
//            float signed_delta_heading = deltaHeading1MinusHeading2(nb_pt.head, search_pt.head);
//            float delta_heading = abs(signed_delta_heading);
//            if (delta_heading > 90.0f) {
//                delta_heading = 180.0f - delta_heading;
//                int tmp_head = (nb_pt.head + 180) % 360;
//                signed_delta_heading = deltaHeading1MinusHeading2(tmp_head, search_pt.head);
//            }
//            
//            if (delta_heading < 25.0f) {
//                int bin_idx = floor((25.0f + signed_delta_heading) / delta_bin);
//                if (bin_idx >= N_BINS) {
//                    bin_idx = N_BINS-1;
//                }
//                if (bin_idx < 0) {
//                    bin_idx = 0;
//                }
//                
//                votes[bin_idx] += 1.0f;
//                
//                pt_count++;
//                is_covered[*it] = true;
//                // Compute perp projection
//                Eigen::Vector3d vec(nb_pt.x - search_pt.x,
//                                    nb_pt.y - search_pt.y,
//                                    0.0f);
//                float parallel_dir = abs(vec.dot(start_dir));
//                if (parallel_dir < 5.0f) {
//                    closeby_pt_count++;
//                }
//                Eigen::Vector3d nb_dir = headingTo3dVector(nb_pt.head);
//                float perp_proj = start_dir.cross(vec)[2];
//                cum_perp_proj += perp_proj;
//                
//                float dot_value = start_dir.dot(nb_dir);
//                if (dot_value < 0) {
//                    avg_dir -= nb_dir;
//                }
//                else{
//                    avg_dir += nb_dir;
//                }
//            }
//        }
//        
//        if (closeby_pt_count < 1) {
//            break;
//        }
//        
//        float max_vote = 0.0f;
//        int max_idx = -1.0f;
//        for (int k = 0; k < votes.size(); ++k) {
//            if (max_vote < votes[k]) {
//                max_vote = votes[k];
//                max_idx = k;
//            }
//        }
//        
//        int new_heading;
//        if (max_idx != -1) {
//            int max_bin_delta_heading = floor(-25.0f + (max_idx + 0.5f) * delta_bin);
//            new_heading = increaseHeadingBy(max_bin_delta_heading, search_pt.head);
//            avg_dir = headingTo3dVector(new_heading);
//        }
//        else{
//            avg_dir.normalize();
//            new_heading = vector3dToHeading(avg_dir);
//        }
//        
//        cum_perp_proj /= pt_count;
//        Eigen::Vector3d new_pt1 = tmp_search_pt + cum_perp_proj * start_perp_dir;
//        Eigen::Vector3d new_pt2 = prev_search_pt + delta * avg_dir;
//        Eigen::Vector3d new_pt = 0.5 * (new_pt1 + new_pt2);
//        search_pt.x = new_pt.x();
//        search_pt.y = new_pt.y();
//        search_pt.head = new_heading;
//        first_branch.push_back(RoadPt(search_pt.x,
//                                      search_pt.y,
//                                      search_pt.head,
//                                      is_oneway));
//    }
//    
//    smoothCurve(first_branch, true);
//    
//    // Partition the points
//    PclPointCloud::Ptr left_points(new PclPointCloud);
//    PclPointCloud::Ptr right_points(new PclPointCloud);
//    
//    for (int i = 0; i < points->size(); ++i) {
//        if (is_covered[i]) {
//            continue;
//        }
//        
//        PclPoint& pt = points->at(i);
//        // Find the closest point to the first branch
//        int closest_idx = -1;
//        float closest_dist = 1e6;
//        for (int j = 0; j < first_branch.size(); ++j) {
//            RoadPt& this_pt = first_branch[j];
//            Eigen::Vector3d vec(pt.x - this_pt.x,
//                                pt.y - this_pt.y,
//                                0.0f);
//            float length = vec.norm();
//            if(length < closest_dist){
//                closest_dist = length;
//                closest_idx = j;
//            }
//        }
//        
//        RoadPt& this_pt = first_branch[closest_idx];
//        Eigen::Vector3d vec(pt.x - this_pt.x,
//                            pt.y - this_pt.y,
//                            0.0f);
//        Eigen::Vector3d this_pt_dir = headingTo3dVector(this_pt.head);
//        if (this_pt_dir.cross(vec)[2] > 0) {
//            left_points->push_back(pt);
//        }
//        else{
//            right_points->push_back(pt);
//        }
//    }
//    
//    // Vote for junction
//    vector<float> votes(first_branch.size(), 0.0f);
//    for (int i = 0; i < left_points->size(); ++i) {
//        PclPoint& pt = left_points->at(i);
//        float speed = pt.speed / 100.0f;
//        if(speed < 5.0f){
//            continue;
//        }
//        Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
//        
//        int vote_idx = -1;
//        float max_vote = 0.0f;
//        for (int j = 0; j < first_branch.size(); ++j) {
//            RoadPt& r_pt = first_branch[j];
//            Eigen::Vector3d vec(pt.x - r_pt.x,
//                                pt.y - r_pt.y,
//                                0.0f);
//            float length = vec.norm();
//            if (length > 1e-3) {
//                vec /= length;
//            }
//            
//            float dot_value = abs(pt_dir.dot(vec));
//            if(dot_value < 0.8f){
//                continue;
//            }
//            
//            if (dot_value > max_vote) {
//                max_vote = dot_value;
//                vote_idx = j;
//            }
//        }
//        
//        if(vote_idx != -1){
//            votes[vote_idx] += 1.0f;
//        }
//    }
//    
//    for (int i = 0; i < right_points->size(); ++i) {
//        PclPoint& pt = right_points->at(i);
//        float speed = pt.speed / 100.0f;
//        if(speed < 5.0f){
//            continue;
//        }
//        
//        Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
//        
//        int vote_idx = -1;
//        float max_vote = 0.0f;
//        for (int j = 0; j < first_branch.size(); ++j) {
//            RoadPt& r_pt = first_branch[j];
//            Eigen::Vector3d vec(pt.x - r_pt.x,
//                                pt.y - r_pt.y,
//                                0.0f);
//            float length = vec.norm();
//            if (length > 1e-3) {
//                vec /= length;
//            }
//            
//            float dot_value = abs(pt_dir.dot(vec));
//            if(dot_value < 0.8f){
//                continue;
//            }
//            if (dot_value > max_vote) {
//                max_vote = dot_value;
//                vote_idx = j;
//            }
//        }
//        
//        if (vote_idx != -1) {
//            votes[vote_idx] += 1.0f;
//        }
//    }
//    
//    int max_idx = -1;
//    float max_value = 1.5f;
//    for(int i = 0; i < votes.size(); ++i){
//        if (max_value < votes[i]) {
//            max_value = votes[i];
//            max_idx = i;
//        }
//    }
//    
//    if (max_idx != -1) {
//        // We have junction
//        RoadPt& r_pt = first_branch[max_idx];
//        junction_loc.x = r_pt.x;
//        junction_loc.y = r_pt.y;
//        
//        Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
//        Eigen::Vector2d r_perp_dir(-r_pt_dir[1], r_pt_dir[0]);
//        Eigen::Vector2d pt(r_pt.x, r_pt.y);
//        
//        Eigen::Vector2d branch_pt(r_pt.x, r_pt.y);
//        
//        // Trace left branch
//        bool has_left_branch = false;
//        bool has_right_branch = false;
//        if (left_points->size() > 5) {
//            PclPoint search_pt;
//            search_pt.setCoordinate(branch_pt.x(), branch_pt.y(), 0.0f);
//            
//            // Trace left branch
//            Eigen::Vector2d avg_dir(0.0f, 0.0f);
//            for (int i = 0; i < left_points->size(); ++i) {
//                PclPoint& nb_pt = left_points->at(i);
//                Eigen::Vector2d vec(nb_pt.x - branch_pt.x(),
//                                    nb_pt.y - branch_pt.y());
//                float vec_length = vec.norm();
//                if(vec_length > 1e-3){
//                    vec /= vec_length;
//                    avg_dir += vec;
//                }
//            }
//            
//            int first_heading = vector2dToHeading(avg_dir);
//            search_pt.head = first_heading;
//            
//            vector<RoadPt> left_branch;
//            left_branch.push_back(RoadPt(search_pt.x,
//                                         search_pt.y,
//                                         search_pt.head,
//                                         false));
//            
//            Eigen::Vector2d left_branch_dir = headingTo2dVector(search_pt.head);
//            Eigen::Vector2d nxt_loc = branch_pt + 10.0f * left_branch_dir;
//            
//            left_branch.push_back(RoadPt(nxt_loc.x(),
//                                         nxt_loc.y(),
//                                         search_pt.head,
//                                         false));
//            
//            branches.push_back(left_branch);
//            has_left_branch = true;
//        }
//        
//        // Trace right branch
//        if (right_points->size() > 5) {
//            // Trace right branch
//            PclPoint search_pt;
//            search_pt.setCoordinate(branch_pt.x(), branch_pt.y(), 0.0f);
//            
//            // Trace right branch
//            Eigen::Vector2d avg_dir(0.0f, 0.0f);
//            for (int i = 0; i < right_points->size(); ++i) {
//                PclPoint& nb_pt = right_points->at(i);
//                Eigen::Vector2d vec(nb_pt.x - branch_pt.x(),
//                                    nb_pt.y - branch_pt.y());
//                float vec_length = vec.norm();
//                if(vec_length > 1e-3){
//                    vec /= vec_length;
//                    avg_dir += vec;
//                }
//            }
//            
//            int first_heading = vector2dToHeading(avg_dir);
//            search_pt.head = first_heading;
//            
//            vector<RoadPt> right_branch;
//            right_branch.push_back(RoadPt(search_pt.x,
//                                          search_pt.y,
//                                          search_pt.head,
//                                          false));
//            
//            Eigen::Vector2d right_branch_dir = headingTo2dVector(search_pt.head);
//            Eigen::Vector2d nxt_loc = branch_pt + 10.0f * right_branch_dir;
//            right_branch.push_back(RoadPt(nxt_loc.x(),
//                                          nxt_loc.y(),
//                                          search_pt.head,
//                                          false));
//            
//            branches.push_back(right_branch);
//            has_right_branch = true;
//        }
//        
//        if (has_left_branch || has_right_branch) {
//            return true;
//        }
//        else{
//            return false;
//        }
//    }
//    else{
//        branches.push_back(first_branch);
//        return false;
//    }
//}

bool branchPredictionWithDebug(float               search_radius,
                               RoadPt&             start_point,
                               set<int>&           candidate_set,
                               Trajectories*       trajectories,
                               RoadPt&             junction_loc,
                               vector<vector<RoadPt> >&     branches,
                               vector<Vertex>&     points_to_draw,
                               vector<Color>&      point_colors,
                               vector<Vertex>&     line_to_draw,
                               vector<Color>&      line_colors,
                               bool                grow_backward){
    
    /*
     Predict if we need to do branch.
        - return value:
            true: branch
            false: grow
     */
    
    if(candidate_set.size() == 0){
        return false;
    }
    
    bool is_oneway = start_point.is_oneway;
    
    bool DEBUG = true;
    
    // Initialize point cloud
    PclPointCloud::Ptr orig_points(new PclPointCloud);
    PclSearchTree::Ptr orig_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    // Add points to point cloud
    for (set<int>::iterator it = candidate_set.begin(); it != candidate_set.end(); ++it) {
        PclPoint& pt = trajectories->data()->at(*it);
        orig_points->push_back(pt);
    }
    
    orig_search_tree->setInputCloud(orig_points);
    
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    vector<int> is_covered(orig_points->size(), false);
    for (size_t i = 0; i < orig_points->size(); ++i) {
        if (is_covered[i]) {
            continue;
        }
        
        is_covered[i] = true;
        PclPoint& pt = orig_points->at(i);
        points->push_back(pt);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
       
        orig_search_tree->radiusSearch(pt, 5.0f, k_indices, k_dist_sqrs);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            if (is_covered[*it]) {
                continue;
            }
            
            PclPoint& nearby_pt = orig_points->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, pt.head));
            if (delta_heading < 10.0f) {
                is_covered[*it] = true;
            }
        }
    }
   
    search_tree->setInputCloud(points);
    
    float delta = 10.0f; // in meters
    // Trace the first branch
    vector<RoadPt> first_branch;
    PclPoint search_pt;
    search_pt.setCoordinate(start_point.x, start_point.y, 0.0f);
    search_pt.head = start_point.head;
    
    first_branch.push_back(RoadPt(search_pt.x,
                                  search_pt.y,
                                  search_pt.head,
                                  is_oneway));
    while (true) {
        Eigen::Vector3d start_dir = headingTo3dVector(search_pt.head);
        Eigen::Vector3d start_perp_dir(-start_dir[1], start_dir[0], 0.0f);
        Eigen::Vector3d prev_search_pt(search_pt.x, search_pt.y, 0.0f);
        Eigen::Vector3d tmp_search_pt = prev_search_pt + delta * start_dir;
        
        search_pt.x = tmp_search_pt.x();
        search_pt.y = tmp_search_pt.y();
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        search_tree->radiusSearch(search_pt,
                                  search_radius,
                                  k_indices,
                                  k_dist_sqrs);
        
        int pt_count = 0;
        int closeby_pt_count = 0;
        Eigen::Vector3d avg_dir(0.0f, 0.0f, 0.0f);
        float cum_perp_proj = 0.0f;
        float delta_bin = 5.0f;
        int N_BINS = 2 * floor(25.0f / delta_bin);
        vector<float> votes(N_BINS, 0.0f);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = points->at(*it);
            
            float signed_delta_heading = deltaHeading1MinusHeading2(nb_pt.head, search_pt.head);
            float delta_heading = abs(signed_delta_heading);
            if (delta_heading > 90.0f) {
                delta_heading = 180.0f - delta_heading;
                int tmp_head = (nb_pt.head + 180) % 360;
                signed_delta_heading = deltaHeading1MinusHeading2(tmp_head, search_pt.head);
            }
            
            if (delta_heading < 25.0f) {
                int bin_idx = floor((25.0f + signed_delta_heading) / delta_bin);
                if (bin_idx >= N_BINS) {
                    bin_idx = N_BINS-1;
                }
                if (bin_idx < 0) {
                    bin_idx = 0;
                }
                
                votes[bin_idx] += 1.0f;
                
                pt_count++;
                is_covered[*it] = true;
                // Compute perp projection
                Eigen::Vector3d vec(nb_pt.x - search_pt.x,
                                    nb_pt.y - search_pt.y,
                                    0.0f);
                float parallel_dir = abs(vec.dot(start_dir));
                if (parallel_dir < 5.0f) {
                    closeby_pt_count++;
                }
                Eigen::Vector3d nb_dir = headingTo3dVector(nb_pt.head);
                float perp_proj = start_dir.cross(vec)[2];
                cum_perp_proj += perp_proj;
                
                float dot_value = start_dir.dot(nb_dir);
                if (dot_value < 0) {
                    avg_dir -= nb_dir;
                }
                else{
                    avg_dir += nb_dir;
                }
            }
        }
        
        if (closeby_pt_count < 1) {
            break;
        }
        
        float max_vote = 0.0f;
        int max_idx = -1.0f;
        for (int k = 0; k < votes.size(); ++k) {
            if (max_vote < votes[k]) {
                max_vote = votes[k];
                max_idx = k;
            }
        }
        
        int new_heading;
        if (max_idx != -1) {
            int max_bin_delta_heading = floor(-25.0f + (max_idx + 0.5f) * delta_bin);
            new_heading = increaseHeadingBy(max_bin_delta_heading, search_pt.head);
            avg_dir = headingTo3dVector(new_heading);
        }
        else{
            avg_dir.normalize();
            new_heading = vector3dToHeading(avg_dir);
        }
        
        cum_perp_proj /= pt_count;
        Eigen::Vector3d new_pt1 = tmp_search_pt + cum_perp_proj * start_perp_dir;
        Eigen::Vector3d new_pt2 = prev_search_pt + delta * avg_dir;
        Eigen::Vector3d new_pt = 0.5 * (new_pt1 + new_pt2);
        search_pt.x = new_pt.x();
        search_pt.y = new_pt.y();
        search_pt.head = new_heading;
        first_branch.push_back(RoadPt(search_pt.x,
                                      search_pt.y,
                                      search_pt.head,
                                      is_oneway));
    }
    
    smoothCurve(first_branch, true);
    
    if (DEBUG) {
        // Display first branch
        for(int i = 1; i < first_branch.size(); ++i){
            RoadPt& v1 = first_branch[i-1];
            RoadPt& v2 = first_branch[i];
            line_to_draw.push_back(SceneConst::getInstance().normalize(v1.x, v1.y, Z_DEBUG));
            line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::YELLOW));
            line_to_draw.push_back(SceneConst::getInstance().normalize(v2.x, v2.y, Z_DEBUG));
            line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::YELLOW));
        }
        
        for(int i = 0; i < first_branch.size(); ++i){
            RoadPt& v = first_branch[i];
            points_to_draw.push_back(SceneConst::getInstance().normalize(v.x, v.y, Z_DEBUG));
            point_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::YELLOW));
        }
    }
    
    // Partition the points
    PclPointCloud::Ptr left_points(new PclPointCloud);
    PclPointCloud::Ptr right_points(new PclPointCloud);
    
    for (int i = 0; i < points->size(); ++i) {
        if (is_covered[i]) {
            continue;
        }
        
        PclPoint& pt = points->at(i);
        
        float speed = pt.speed / 100.0f;
        if(speed < 5.0f){
            continue;
        }
        
        // Find the closest point to the first branch
        int closest_idx = -1;
        float closest_dist = 1e6;
        for (int j = 0; j < first_branch.size(); ++j) {
            RoadPt& this_pt = first_branch[j];
            Eigen::Vector3d vec(pt.x - this_pt.x,
                                pt.y - this_pt.y,
                                0.0f);
            float length = vec.norm();
            if(length < closest_dist){
                closest_dist = length;
                closest_idx = j;
            }
        }
        
        RoadPt& this_pt = first_branch[closest_idx];
        Eigen::Vector3d vec(pt.x - this_pt.x,
                            pt.y - this_pt.y,
                            0.0f);
        Eigen::Vector3d this_pt_dir = headingTo3dVector(this_pt.head);
        if (this_pt_dir.cross(vec)[2] > 0) {
            left_points->push_back(pt);
        }
        else{
            right_points->push_back(pt);
        }
    }
    
    if (DEBUG) {
        for(int i = 0; i < left_points->size(); ++i){
            PclPoint& pt = left_points->at(i);
            points_to_draw.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
            point_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::PINK));
        }
        
        for(int i = 0; i < right_points->size(); ++i){
            PclPoint& pt = right_points->at(i);
            points_to_draw.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
            point_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
        }
    }
    
    // Vote for junction
    vector<float> votes(first_branch.size(), 0.0f);
    for (int i = 0; i < left_points->size(); ++i) {
        PclPoint& pt = left_points->at(i);
        float speed = pt.speed / 100.0f;
        if(speed < 5.0f){
            continue;
        }
        Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
        
        int vote_idx = -1;
        float max_vote = 0.0f;
        for (int j = 0; j < first_branch.size(); ++j) {
            RoadPt& r_pt = first_branch[j];
            Eigen::Vector3d vec(pt.x - r_pt.x,
                                pt.y - r_pt.y,
                                0.0f);
            float length = vec.norm();
            if (length > 1e-3) {
                vec /= length;
            }
            
            float dot_value = abs(pt_dir.dot(vec));
            if(dot_value < 0.8f){
                continue;
            }
            
            if (dot_value > max_vote) {
                max_vote = dot_value;
                vote_idx = j;
            }
        }
        
        if(vote_idx != -1){
            votes[vote_idx] += 1.0f;
        }
    }
    
    for (int i = 0; i < right_points->size(); ++i) {
        PclPoint& pt = right_points->at(i);
        float speed = pt.speed / 100.0f;
        if(speed < 5.0f){
            continue;
        }
        
        Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
        
        int vote_idx = -1;
        float max_vote = 0.0f;
        for (int j = 0; j < first_branch.size(); ++j) {
            RoadPt& r_pt = first_branch[j];
            Eigen::Vector3d vec(pt.x - r_pt.x,
                                pt.y - r_pt.y,
                                0.0f);
            float length = vec.norm();
            if (length > 1e-3) {
                vec /= length;
            }
            
            float dot_value = abs(pt_dir.dot(vec));
            if(dot_value < 0.8f){
                continue;
            }
            if (dot_value > max_vote) {
                max_vote = dot_value;
                vote_idx = j;
            }
        }
       
        if (vote_idx != -1) {
            votes[vote_idx] += 1.0f;
        }
    }
    
    int max_idx = -1;
    float max_value = 1.5f;
    for(int i = 0; i < votes.size(); ++i){
        if (max_value < votes[i]) {
            max_value = votes[i];
            max_idx = i;
        }
    }
    
    if (max_idx != -1) {
        // We have junction
        RoadPt& r_pt = first_branch[max_idx];
        junction_loc.x = r_pt.x;
        junction_loc.y = r_pt.y;
        
        Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
        Eigen::Vector2d r_perp_dir(-r_pt_dir[1], r_pt_dir[0]);
        Eigen::Vector2d pt(r_pt.x, r_pt.y);
        
        if (DEBUG) {
            // Visualize junction
            Eigen::Vector2d v1 = pt + 5.0f * r_perp_dir;
            Eigen::Vector2d v2 = pt - 5.0f * r_perp_dir;
            Eigen::Vector2d v3 = pt + 5.0f * r_pt_dir;
            Eigen::Vector2d v4 = pt - 5.0f * r_pt_dir;
            line_to_draw.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_DEBUG));
            line_to_draw.push_back(SceneConst::getInstance().normalize(v2.x(), v2.y(), Z_DEBUG));
            line_to_draw.push_back(SceneConst::getInstance().normalize(v3.x(), v3.y(), Z_DEBUG));
            line_to_draw.push_back(SceneConst::getInstance().normalize(v4.x(), v4.y(), Z_DEBUG));
            line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
        }
        
        Eigen::Vector2d branch_pt(r_pt.x, r_pt.y);
        
        // Trace left branch
        if (left_points->size() > 5) {
            PclPoint search_pt;
            search_pt.setCoordinate(branch_pt.x(), branch_pt.y(), 0.0f);
            
            // Trace left branch
            Eigen::Vector2d avg_dir(0.0f, 0.0f);
            for (int i = 0; i < left_points->size(); ++i) {
                PclPoint& nb_pt = left_points->at(i);
                Eigen::Vector2d vec(nb_pt.x - branch_pt.x(),
                                    nb_pt.y - branch_pt.y());
                float vec_length = vec.norm();
                if(vec_length > 1e-3){
                    vec /= vec_length;
                    avg_dir += vec;
                }
            }
            
            int first_heading = vector2dToHeading(avg_dir);
            search_pt.head = first_heading;
            
            vector<RoadPt> left_branch;
            left_branch.push_back(RoadPt(search_pt.x,
                                         search_pt.y,
                                         search_pt.head,
                                         false));
            
            Eigen::Vector2d left_branch_dir = headingTo2dVector(search_pt.head);
            Eigen::Vector2d nxt_loc = branch_pt + 20.0f * left_branch_dir;
            
            left_branch.push_back(RoadPt(nxt_loc.x(),
                                         nxt_loc.y(),
                                         search_pt.head,
                                         false));
            
            branches.push_back(left_branch);
            
            if (DEBUG) {
                line_to_draw.push_back(SceneConst::getInstance().normalize(left_branch[0].x, left_branch[0].y, Z_DEBUG + 0.05f));
                line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::PINK));
                line_to_draw.push_back(SceneConst::getInstance().normalize(left_branch[1].x, left_branch[1].y, Z_DEBUG + 0.05f));
                line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::PINK));
            }
        }
        
        // Trace right branch
        if (right_points->size() > 5) {
            // Trace right branch
            PclPoint search_pt;
            search_pt.setCoordinate(branch_pt.x(), branch_pt.y(), 0.0f);
            
            // Trace right branch
            Eigen::Vector2d avg_dir(0.0f, 0.0f);
            for (int i = 0; i < right_points->size(); ++i) {
                PclPoint& nb_pt = right_points->at(i);
                Eigen::Vector2d vec(nb_pt.x - branch_pt.x(),
                                    nb_pt.y - branch_pt.y());
                float vec_length = vec.norm();
                if(vec_length > 1e-3){
                    vec /= vec_length;
                    avg_dir += vec;
                }
            }
            
            int first_heading = vector2dToHeading(avg_dir);
            search_pt.head = first_heading;
            
            vector<RoadPt> right_branch;
            right_branch.push_back(RoadPt(search_pt.x,
                                          search_pt.y,
                                          search_pt.head,
                                          false));
            
            Eigen::Vector2d right_branch_dir = headingTo2dVector(search_pt.head);
            Eigen::Vector2d nxt_loc = branch_pt + 20.0f * right_branch_dir;
            right_branch.push_back(RoadPt(nxt_loc.x(),
                                          nxt_loc.y(),
                                          search_pt.head,
                                          false));
            
            branches.push_back(right_branch);
            
            if (DEBUG) {
                line_to_draw.push_back(SceneConst::getInstance().normalize(right_branch[0].x, right_branch[0].y, Z_DEBUG + 0.05f));
                line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
                line_to_draw.push_back(SceneConst::getInstance().normalize(right_branch[1].x, right_branch[1].y, Z_DEBUG + 0.05f));
                line_colors.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
            }
        }
        
        return true;
    }
    else{
        branches.push_back(first_branch);
        return false;
    }
}

void trainQueryQClassifier(vector<query_q_sample_type>& samples,
                           vector<int>& orig_labels,
                           query_q_decision_function& df){
    if (samples.size() != orig_labels.size()) {
        cout << "WARNING: sample and label do not have same size." << endl;
        return;
    }
    
    cout << "Start query q classifier training..." << endl;
    cout << "\tSample size: " << samples.size() << endl;
    
    vector<double> labels(orig_labels.size()); // convert int label to double (for dlib)
    for(int i = 0; i < orig_labels.size(); ++i){
        if(orig_labels[i] == R_GROW){
            labels[i] = -1.0f;
        }
        else{
            labels[i] = 1.0f;
        }
    }
    
    dlib::randomize_samples(samples, labels);
    
    query_q_trainer trainer;
    
    // Cross validation portion
//    for(double gamma = 0.1; gamma <= 0.31; gamma += 0.1){
//        for (double nu = 0.1; nu <= 0.3; nu += 0.05) {
//            trainer.set_kernel(query_q_rbf_kernel(gamma));
//            trainer.set_nu(nu);
//            cout << "gamma: " << gamma << "    nu: " << nu;
//            cout << "     cross validation accuracy: "
//            << dlib::cross_validate_trainer(trainer, samples, labels, 3);
//        }
//    }
//    return;
    // End of: Cross validation portion
    
    double gamma = 0.3f;
    double nu = 0.2f;
    
    trainer.set_kernel(query_q_rbf_kernel(gamma));
    trainer.set_nu(nu);
    
    df = trainer.train(samples, labels);
    // print out the number of support vectors in the resulting decision function
    cout << "\nnumber of support vectors in our learned_function is "
         << df.basis_vectors.size() << endl;
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
    
    decision_function_valid_ = false;
    
    features_.clear();
    labels_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
}

QueryQFeatureSelector::QueryQFeatureSelector(QObject* parent, Trajectories* trajectories) : Renderable(parent) {
    trajectories_ = trajectories;
    osmMap_ = NULL;
    
    tmp_ = 0;
    ARROW_LENGTH_ = 5.0f; // in meters
    
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
    glPointSize(10);
    glDrawArrays(GL_POINTS, 0, feature_vertices_.size());
    
    // Draw sample heading
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&feature_headings_[0], 3*feature_headings_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&feature_heading_colors_[0], 4*feature_heading_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    glDrawArrays(GL_LINES, 0, feature_headings_.size());
}

void QueryQFeatureSelector::extractTrainingSamplesFromMap(float radius, OpenStreetMap *osmMap){
    if (osmMap->isEmpty()) {
        return;
    }
    
    osmMap_ = osmMap;
    
    if (trajectories_ == NULL) {
        return;
    }
    
    bool DEBUG_MODE = true;
    
    features_.clear();
    feature_locs_.clear();
    feature_is_front_.clear();
    labels_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
    feature_headings_.clear();
    feature_heading_colors_.clear();
    
    // Resample map with a point cloud using a grid
    osmMap_->updateMapSearchTree(0.5f * radius);
    
    PclPointCloud::Ptr& map_sample_points = osmMap_->map_point_cloud();
    
    if (DEBUG_MODE) {
        int pt_idx = tmp_;
        bool forward = true;
        PclPoint& pt = map_sample_points->at(pt_idx);
        int way_id = pt.id_trajectory;
        
        // Compute center line for this road
        vector<int>& way_point_idxs = osmMap_->way_point_idxs()[way_id];
        int this_point_idx = static_cast<int>(pt.car_id);
        if (forward) {
            // Compute Forward Direction Feature and labels
            vector<RoadPt> fwd_center_line;
            for (int s = 0; s <= this_point_idx; ++s) {
                int prev_pt_idx = way_point_idxs[s];
                PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
               
                fwd_center_line.push_back(RoadPt(prev_pt.x, prev_pt.y, prev_pt.head, osmMap_->ways()[way_id].isOneway()));
                
                feature_vertices_.push_back(SceneConst::getInstance().normalize(prev_pt.x, prev_pt.y, Z_FEATURES));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            }
            
            if (fwd_center_line.size() > 2) {
                query_q_sample_type fwd_feature;
                set<int> candidate_point_set;
                bool has_feature = computeQueryQFeatureAtForVisualization(radius,
                                                          trajectories_,
                                                          fwd_feature,
                                                          fwd_center_line,
                                                        candidate_point_set);
                if (!has_feature) {
                    return;
                }
                
                for(set<int>::iterator it = candidate_point_set.begin(); it != candidate_point_set.end(); ++it){
                    PclPoint& pt = trajectories_->data()->at(*it);
                    feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
                    feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
                }
                
                feature_headings_.clear();
                feature_heading_colors_.clear();
                RoadPt start_pt(pt.x, pt.y, pt.head, osmMap_->ways()[way_id].isOneway());
                RoadPt junction_loc;
                vector<vector<RoadPt> > branches;
                branchPredictionWithDebug(radius,
                                          start_pt,
                                          candidate_point_set,
                                          trajectories_,
                                          junction_loc,
                                          branches,
                                          feature_vertices_,
                                          feature_colors_,
                                          feature_headings_,
                                          feature_heading_colors_);
            }
        }
        else{
            vector<RoadPt> bwd_center_line;
            for (int s = this_point_idx; s < way_point_idxs.size(); ++s) {
                int prev_pt_idx = way_point_idxs[s];
                PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
                
                bwd_center_line.push_back(RoadPt(prev_pt.x, prev_pt.y, prev_pt.head, osmMap_->ways()[way_id].isOneway()));
                
                feature_vertices_.push_back(SceneConst::getInstance().normalize(prev_pt.x, prev_pt.y, Z_FEATURES));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            }
            
            if (bwd_center_line.size() > 2) {
                query_q_sample_type bwd_feature;
                set<int> candidate_point_set;
                bool has_feature = computeQueryQFeatureAtForVisualization           (radius,
                    trajectories_,
                    bwd_feature,
                    bwd_center_line,
                    candidate_point_set,
                    true);
                
                if (!has_feature) {
                    return;
                }
                
                for(set<int>::iterator it = candidate_point_set.begin(); it != candidate_point_set.end(); ++it){
                    PclPoint& pt = trajectories_->data()->at(*it);
                    feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
                    feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
                }
            }
        }
        
        tmp_++;
        return;
    }
    
    // Compute Query Q features for each map sample point
    for (size_t pt_idx = 0; pt_idx < map_sample_points->size(); ++pt_idx) {
        PclPoint& pt = map_sample_points->at(pt_idx);
        int way_id = pt.id_trajectory;
        float heading_in_radius = pt.head * PI / 180.0f;
        
        // Compute center line for this road
        vector<int>& way_point_idxs = osmMap_->way_point_idxs()[way_id];
        int this_point_idx = static_cast<int>(pt.car_id);
        
        // Compute Forward Direction Feature and labels
        vector<RoadPt> fwd_center_line;
        for (int s = 0; s <= this_point_idx; ++s) {
            int prev_pt_idx = way_point_idxs[s];
            PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
            
            fwd_center_line.push_back(RoadPt(prev_pt.x, prev_pt.y, prev_pt.head,                                             osmMap_->ways()[way_id].isOneway()));
        }
    
        if (fwd_center_line.size() > 2) {
            query_q_sample_type fwd_feature;
            bool has_feature = computeQueryQFeatureAt(radius,
                                                      trajectories_,
                                                      fwd_feature,
                                                      fwd_center_line);
            if (!has_feature) {
                continue;
            }
            
            int fwd_label = R_GROW;
            computeLabelAt(radius,
                           pt,
                           fwd_label);
            features_.push_back(fwd_feature);
            labels_.push_back(fwd_label);
            feature_locs_.push_back(Vertex(pt.x, pt.y, pt.head)); // Notice of a hack here: (x, y, heading)
            feature_is_front_.push_back(true);
            
            feature_headings_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
            feature_headings_.push_back(SceneConst::getInstance().normalize(pt.x + ARROW_LENGTH_*cos(heading_in_radius), pt.y + ARROW_LENGTH_*sin(heading_in_radius), Z_FEATURES));
            feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x + ARROW_LENGTH_*cos(heading_in_radius), pt.y + ARROW_LENGTH_*sin(heading_in_radius), Z_FEATURES));
            
            if (fwd_label == R_GROW) {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
            }
            else {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
            }
        }
        
        // Backward direction
        vector<RoadPt> bwd_center_line;
        for (int s = this_point_idx; s < way_point_idxs.size(); ++s) {
            int prev_pt_idx = way_point_idxs[s];
            PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
            
            bwd_center_line.push_back(RoadPt(prev_pt.x, prev_pt.y, prev_pt.head, osmMap_->ways()[way_id].isOneway()));
        }
        
        if (bwd_center_line.size() > 2) {
            query_q_sample_type bwd_feature;
            bool has_feature = computeQueryQFeatureAt(radius,
                                                      trajectories_,
                                                      bwd_feature,
                                                      bwd_center_line,
                                                      true);
            if (!has_feature) {
                continue;
            }
            
            int bwd_label = R_GROW;
            computeLabelAt(radius,
                           pt,
                           bwd_label,
                           true);
            features_.push_back(bwd_feature);
            labels_.push_back(bwd_label);
            feature_locs_.push_back(Vertex(pt.x, pt.y, pt.head));
            feature_is_front_.push_back(false);
            
            feature_headings_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
            feature_headings_.push_back(SceneConst::getInstance().normalize(pt.x - ARROW_LENGTH_*cos(heading_in_radius), pt.y - ARROW_LENGTH_*sin(heading_in_radius), Z_FEATURES));
            
            if (bwd_label == R_GROW) {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
            }
            else {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
            }
        }
    }
}

void QueryQFeatureSelector::computeLabelAt(float radius,
                                           PclPoint& point,
                                           int& label,
                                           bool is_reverse_dir){
    if (osmMap_ == NULL) {
        return;
    }
    
    int way_id = point.id_trajectory;
    vector<int>& way_point_idxs = osmMap_->way_point_idxs()[way_id];
    int this_point_idx = static_cast<int>(point.car_id);
    
    label = R_GROW;
    
    if (!is_reverse_dir){
        // Look Forward
        float cum_dist = 0.0f;
        for (int i = this_point_idx; i < way_point_idxs.size(); ++i) {
            int map_pt_idx = way_point_idxs[i];
            PclPoint& this_pt = osmMap_->map_point_cloud()->at(map_pt_idx);
            int this_node_degree = this_pt.id_sample;
            
            if (i > this_point_idx) {
                int prev_map_pt_idx = way_point_idxs[i-1];
                PclPoint& prev_pt = osmMap_->map_point_cloud()->at(prev_map_pt_idx);
                float delta_x = this_pt.x - prev_pt.x;
                float delta_y = this_pt.y - prev_pt.y;
                cum_dist += sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            
            if (this_node_degree > 2) {
                label = R_BRANCH;
                return;
            }
            
            if(cum_dist > 0.2f * radius){
                break;
            }
        }
    }
    else{
        // Look Backward
        float cum_dist = 0.0f;
        for (int i = this_point_idx; i >= 0; --i) {
            int map_pt_idx = way_point_idxs[i];
            PclPoint& this_pt = osmMap_->map_point_cloud()->at(map_pt_idx);
            
            int this_node_degree = this_pt.id_sample;
            
            if (i < this_point_idx) {
                int prev_map_pt_idx = way_point_idxs[i+1];
                PclPoint& prev_pt = osmMap_->map_point_cloud()->at(prev_map_pt_idx);
                float delta_x = this_pt.x - prev_pt.x;
                float delta_y = this_pt.y - prev_pt.y;
                cum_dist += sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            
            if (this_node_degree > 2) {
                label = R_BRANCH;
                return;
            }
            
            if(cum_dist > 0.2f * radius){
                break;
            }
        }
    }
}

bool QueryQFeatureSelector::saveTrainingSamples(const string& filename){
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    if (features_.size() != labels_.size()) {
        cout << "ERROR! Query Q feature and label doesn't match!" << endl;
        return false;
    }
    
    output << features_.size() << endl;
    int n_rows = features_[0].nr();
    for (size_t i = 0; i < features_.size(); ++i) {
        output << labels_[i] << endl;
        output << feature_locs_[i].x << ", " << feature_locs_[i].y << ", " << floor(feature_locs_[i].z) << ", " << feature_is_front_[i] << endl;
        for (int j = 0; j < n_rows-1; ++j) {
            output << features_[i](j) << ", ";
        }
        output << features_[i](n_rows-1) << endl;
    }
    output.close();
    return true;
}

bool QueryQFeatureSelector::loadTrainingSamples(const string& filename){
    features_.clear();
    feature_locs_.clear();
    feature_is_front_.clear();
    labels_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
    feature_headings_.clear();
    feature_heading_colors_.clear();
    
    ifstream input;
    input.open(filename);
    if (input.fail()) {
        return false;
    }
    
    string str;
    getline(input, str);
    int n_features = stoi(str);
    
    int n_grow_label = 0;
    int n_branch_label = 0;
    for(int i = 0; i < n_features; ++i){
        getline(input, str);
        int label = stoi(str);
        
        getline(input, str);
        QString str1(str.c_str());
        QStringList list = str1.split(", ");
        
        if(list.size() != 4){
            cout << "Error when loading query q samples!" << endl;
            return false;
        }
        
        float loc_x = stof(list.at(0).toStdString());
        float loc_y = stof(list.at(1).toStdString());
        int loc_head = stoi(list.at(2).toStdString());
        float heading_in_radius = loc_head * PI / 180.0f;
        int is_front = stoi(list.at(3).toStdString());
        
        if(is_front == 1){
            feature_is_front_.push_back(true);
        }
        else{
            feature_is_front_.push_back(false);
        }
        
        feature_locs_.push_back(Vertex(loc_x, loc_y, loc_head));
        
        getline(input, str);
        QString str2(str.c_str());
        list = str2.split(", ");
        
        query_q_sample_type new_feature;
        for (int j = 0; j < list.size(); ++j) {
            new_feature(j) = stof(list.at(j).toStdString());
        }
        
        // Add to features
        features_.push_back(new_feature);
        labels_.push_back(label);
        
        if (label == R_GROW) {
            n_grow_label++;
        }
        else{
            n_branch_label++;
        }
        
        // Visualization
        if (is_front == 1) {
            feature_headings_.push_back(SceneConst::getInstance().normalize(loc_x, loc_y, Z_FEATURES));
            feature_headings_.push_back(SceneConst::getInstance().normalize(loc_x + ARROW_LENGTH_*cos(heading_in_radius), loc_y + ARROW_LENGTH_*sin(heading_in_radius), Z_FEATURES));
            feature_vertices_.push_back(SceneConst::getInstance().normalize(loc_x + ARROW_LENGTH_*cos(heading_in_radius), loc_y + ARROW_LENGTH_*sin(heading_in_radius), Z_FEATURES));
            
            if (label == R_GROW) {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
            }
            else {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
            }
        }
        else{
            feature_headings_.push_back(SceneConst::getInstance().normalize(loc_x, loc_y, Z_FEATURES));
            feature_headings_.push_back(SceneConst::getInstance().normalize(loc_x - ARROW_LENGTH_*cos(heading_in_radius), loc_y - ARROW_LENGTH_*sin(heading_in_radius), Z_FEATURES));
            
            if (label == R_GROW) {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GROW_COLOR));
            }
            else {
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
                feature_heading_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BRANCH_COLOR));
            }
        }
    }
    
    input.close();
    
    cout << "Query Q samples loaded." << endl;
    cout << "\tTotally " << n_branch_label + n_grow_label << " samples, " << n_branch_label << " branch labels, " << n_grow_label << " grow labels." << endl;
    
    return true;
}

bool QueryQFeatureSelector::trainClassifier(){
    if(features_.size() == 0){
        cout << "Error! Empty training samples." << endl;
        return false;
    }
    if(features_.size() != labels_.size()){
        cout << "Error! Feature and label sizes do not match." << endl;
        return false;
    }
    
    trainQueryQClassifier(features_,
                          labels_,
                          df_);
    
    decision_function_valid_ = true;
    
    return true;
}

bool QueryQFeatureSelector::saveClassifier(const string& filename){
    if(!decision_function_valid_){
        cout << "Invalid decision function." << endl;
        return false;
    }
    
    dlib::serialize(filename.c_str()) << df_;
    
    return true;
}

void QueryQFeatureSelector::clearVisibleFeatures(){
    feature_vertices_.clear();
    feature_colors_.clear();
    feature_headings_.clear();
    feature_heading_colors_.clear();
}

void QueryQFeatureSelector::clear(){
    trajectories_ = NULL;
    osmMap_ = NULL;
    
    decision_function_valid_ = false;
    
    features_.clear();
    labels_.clear();
    feature_locs_.clear();
    feature_is_front_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
    feature_headings_.clear();
    feature_heading_colors_.clear();
}
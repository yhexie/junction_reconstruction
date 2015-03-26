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

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

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

void computeConsistentPointSet(float radius,
                               Trajectories* trajectories,
                               vector<Vertex>& center_line,
                               set<int>& candidate_point_set,
                               bool is_oneway,
                               bool grow_backward
                               ){
    candidate_point_set.clear();
    
    // (R, theta)distribution of consistent trajectories points + intersection distribution for points that have bigger than 7.5 degree heading diff.
    if (center_line.size() < 2) {
        cout << "size less than 2" << endl;
        return;
    }
    
    PclPointCloud::Ptr& data = trajectories->data();
    PclSearchTree::Ptr& tree = trajectories->tree();
    
    PclPoint orig_pt;
    if (grow_backward) {
        orig_pt.setCoordinate(center_line.front().x,
                              center_line.front().y,
                              0.0f);
        orig_pt.head = floor(center_line.front().z);
        
    }
    else{
        orig_pt.setCoordinate(center_line.back().x,
                              center_line.back().y,
                              0.0f);
        orig_pt.head = floor(center_line.back().z);
    }
    
    float orig_heading_in_radius = orig_pt.head * PI / 180.0f;
    Eigen::Vector3d orig_dir(cos(orig_heading_in_radius),
                             sin(orig_heading_in_radius),
                             0.0f);
    
    // Feature parameters
    float HEADING_THRESHOLD         = 15.0f;
    float MAX_DIST_TO_ROAD_CENTER   = 0.5f * radius; // in meters
    float DELTA_T_EXTENSION         = 30.0f; // in seconds. Include points that extend current point by at most this value.
    float MAX_EXTENSION             = 3.0f * radius; // we will trace the centerline for up to this distance to find trajectory fall on this road
    
    // Find trajectories that falls on this road, and compute angle distribution
    PclPoint pt;
    if (grow_backward) {
        // backward direction
        float cum_length = 0.0f; // the maximum extension length we will look at
        
        set<int> visited_traj_idxs;
        for (int i = 0; i < center_line.size(); ++i) {
            if (i != 0) {
                float delta_x = center_line[i].x - center_line[i-1].x;
                float delta_y = center_line[i].y - center_line[i-1].y;
                cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            
            pt.setCoordinate(center_line[i].x, center_line[i].y, 0.0f);
            int heading = floor(center_line[i].z);
            float heading_in_radius = heading * PI / 180.0f;
            Eigen::Vector3d dir(cos(heading_in_radius),
                                sin(heading_in_radius),
                                0.0f);
            
            // Look nearby
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            tree->radiusSearch(pt,
                               radius,
                               k_indices,
                               k_dist_sqrs);
            for (size_t s = 0; s < k_indices.size(); ++s) {
                int nearby_pt_idx = k_indices[s];
                PclPoint& nearby_pt = data->at(nearby_pt_idx);
                int nearby_traj_idx = nearby_pt.id_trajectory;
                if (visited_traj_idxs.find(nearby_traj_idx) != visited_traj_idxs.end()) {
                    // This trajectory has already been visited
                    continue;
                }
                
                Eigen::Vector3d vec(nearby_pt.x - pt.x,
                                    nearby_pt.y - pt.y,
                                    0.0f);
                
                Eigen::Vector3d vec0(nearby_pt.x - orig_pt.x,
                                     nearby_pt.y - orig_pt.y,
                                     0.0f);
                
                // Check position
                if (vec0.dot(orig_dir) < 1.0f) {
                    continue;
                }
                
                // Check heading
                float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, heading));
                float corrected_heading = delta_heading;
                if (!is_oneway) {
                    if (delta_heading > 90.0f) {
                        corrected_heading = 180.0f - delta_heading;
                    }
                }
                if (corrected_heading > HEADING_THRESHOLD) {
                    continue;
                }
                
                // Check perpendicular dist
                float perp_dist = dir.cross(vec)[2];
                if (abs(perp_dist) > MAX_DIST_TO_ROAD_CENTER) {
                    continue;
                }
                
                // Now, this trajectory is compatible with current road, compute its heading change cross the search region
                visited_traj_idxs.insert(nearby_traj_idx);
                
                bool look_backward = true;
                if (!is_oneway) {
                    if (delta_heading > 90.0f) {
                        look_backward = false;
                    }
                }
                
                const vector<int>& traj_pt_idxs = trajectories->trajectories()[nearby_traj_idx];
                if (traj_pt_idxs.size() == 0) {
                    continue;
                }
                
                int id_sample = nearby_pt.id_sample;
                float start_time = nearby_pt.t;
                
                if (!look_backward) {
                    bool passed_query_point = false;
                    for (int l = id_sample; l < traj_pt_idxs.size(); ++l) {
                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
                        
                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
                                                 this_pt.y - orig_pt.y,
                                                 0.0f);
                        
                        // Check position
                        if(!passed_query_point){
                            if (this_vec.dot(orig_dir) < -1.0f) {
                                passed_query_point = true;
                            }
                        }
                        
                        if (passed_query_point) {
                            float delta_t = abs(this_pt.t - start_time);
                            float delta_x = this_pt.x - orig_pt.x;
                            float delta_y = this_pt.y - orig_pt.y;
                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
                            
                            if (dist_to_orig < 4 * radius &&
                                delta_t < DELTA_T_EXTENSION) {
                                // Insert to candidate point set
                                candidate_point_set.insert(traj_pt_idxs[l]);
                            }
                            else{
                                break;
                            }
                        }
                    }
                }
                else{
                    bool passed_query_point = false;
                    for (int l = id_sample; l >= 0; --l) {
                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
                                                 this_pt.y - orig_pt.y,
                                                 0.0f);
                        
                        // Check position
                        if(!passed_query_point){
                            if (this_vec.dot(orig_dir) < -1.0f) {
                                passed_query_point = true;
                            }
                        }
                        if (passed_query_point) {
                            float delta_t = abs(this_pt.t - start_time);
                            float delta_x = this_pt.x - orig_pt.x;
                            float delta_y = this_pt.y - orig_pt.y;
                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
                            if (dist_to_orig < 4 * radius &&
                                delta_t < DELTA_T_EXTENSION) {
                                // Insert to candidate point set
                                candidate_point_set.insert(traj_pt_idxs[l]);
                            }
                            else{
                                break;
                            }
                        }
                    }
                }
            }
//            if (cum_length > MAX_EXTENSION) {
//                break;
//            }
        }
    }
    else{
        // forward direction
        float cum_length = 0.0f; // the extension length we will look at
        
        set<int> visited_traj_idxs;
        for (int i = center_line.size()-1; i >= 0; --i) {
            if (i != center_line.size() - 1) {
                float delta_x = center_line[i].x - center_line[i+1].x;
                float delta_y = center_line[i].y - center_line[i+1].y;
                cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            
            pt.setCoordinate(center_line[i].x, center_line[i].y, 0.0f);
            int heading = floor(center_line[i].z);
            float heading_in_radius = heading * PI / 180.0f;
            Eigen::Vector3d dir(cos(heading_in_radius),
                                sin(heading_in_radius),
                                0.0f);
            
            // Look nearby
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            tree->radiusSearch(pt,
                               radius,
                               k_indices,
                               k_dist_sqrs);
            for (size_t s = 0; s < k_indices.size(); ++s) {
                int nearby_pt_idx = k_indices[s];
                PclPoint& nearby_pt = data->at(nearby_pt_idx);
                int nearby_traj_idx = nearby_pt.id_trajectory;
                
                if (visited_traj_idxs.find(nearby_traj_idx) != visited_traj_idxs.end()) {
                    // This trajectory has already been visited
                    continue;
                }
                
                Eigen::Vector3d vec(nearby_pt.x - pt.x,
                                    nearby_pt.y - pt.y,
                                    0.0f);
                
                Eigen::Vector3d vec0(nearby_pt.x - orig_pt.x,
                                     nearby_pt.y - orig_pt.y,
                                     0.0f);
                
                // Check position
                if (vec0.dot(orig_dir) > -1.0f) {
                    continue;
                }
                
                // Check heading
                float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, heading));
                float corrected_heading = delta_heading;
                if (!is_oneway) {
                    if (delta_heading > 90.0f) {
                        corrected_heading = 180.0f - delta_heading;
                    }
                }
                if (corrected_heading > HEADING_THRESHOLD) {
                    continue;
                }
                
                // Check perpendicular dist
                float perp_dist = dir.cross(vec)[2];
                if (abs(perp_dist) > MAX_DIST_TO_ROAD_CENTER) {
                    continue;
                }
                
                // Now, this trajectory is compatible with current road, compute its heading change cross the search region
                // Find the first point that's outside the region
                visited_traj_idxs.insert(nearby_traj_idx);
                
                bool look_forward = true;
                if (!is_oneway) {
                    if (delta_heading > 90.0f) {
                        look_forward = false;
                    }
                }
                
                const vector<int>& traj_pt_idxs = trajectories->trajectories()[nearby_traj_idx];
                if (traj_pt_idxs.size() == 0) {
                    continue;
                }
                
                int id_sample = nearby_pt.id_sample;
                float start_time = nearby_pt.t;
                
                if (look_forward) {
                    bool passed_query_point = false;
                    for (int l = id_sample; l < traj_pt_idxs.size(); ++l) {
                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
                        
                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
                                                 this_pt.y - orig_pt.y,
                                                 0.0f);
                        
                        // Check position
                        if(!passed_query_point){
                            if (this_vec.dot(orig_dir) > 1.0f) {
                                passed_query_point = true;
                            }
                        }
                        
                        if (passed_query_point) {
                            float delta_t = abs(this_pt.t - start_time);
                            float delta_x = this_pt.x - orig_pt.x;
                            float delta_y = this_pt.y - orig_pt.y;
                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
                            if (dist_to_orig < 4 * radius &&
                                delta_t < DELTA_T_EXTENSION) {
                                // Insert to candidate point set
                                candidate_point_set.insert(traj_pt_idxs[l]);
                            }
                            else{
                                break;
                            }
                        }
                    }
                }
                else{
                    bool passed_query_point = false;
                    for (int l = id_sample; l >= 0; --l) {
                        PclPoint& this_pt = data->at(traj_pt_idxs[l]);
                        Eigen::Vector3d this_vec(this_pt.x - orig_pt.x,
                                                 this_pt.y - orig_pt.y,
                                                 0.0f);
                        
                        // Check position
                        if(!passed_query_point){
                            if (this_vec.dot(orig_dir) > 1.0f) {
                                passed_query_point = true;
                            }
                        }
                        if (passed_query_point) {
                            float delta_t = abs(this_pt.t - start_time);
                            float delta_x = this_pt.x - orig_pt.x;
                            float delta_y = this_pt.y - orig_pt.y;
                            float dist_to_orig = sqrt(delta_x*delta_x + delta_y*delta_y);
                            if (dist_to_orig < 4 * radius &&
                                delta_t < DELTA_T_EXTENSION) {
                                // Insert to candidate point set
                                candidate_point_set.insert(traj_pt_idxs[l]);
                            }
                            else{
                                break;
                            }
                        }
                    }
                }
            }
//            if (cum_length > MAX_EXTENSION) {
//                break;
//            }
        }
    }
}

bool computeQueryQFeatureAt(float radius,
                            Trajectories* trajectories,
                            query_q_sample_type& feature,
                            vector<Vertex>& center_line,
                            bool is_oneway,
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
        orig_pt.head = floor(center_line.front().z);
        
    }
    else{
        orig_pt.setCoordinate(center_line.back().x,
                              center_line.back().y,
                              0.0f);
        orig_pt.head = floor(center_line.back().z);
    }
    
    float orig_heading_in_radius = orig_pt.head * PI / 180.0f;
    Eigen::Vector3d orig_dir(cos(orig_heading_in_radius),
                             sin(orig_heading_in_radius),
                             0.0f);
    
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
    
    computeConsistentPointSet(radius,
                              trajectories,
                              center_line,
                              candidate_point_set,
                              is_oneway,
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
                                            vector<Vertex>& center_line,
                                            set<int>& candidate_point_set,
                                            bool is_oneway,
                                            bool grow_backward){
    candidate_point_set.clear();
    
    // (R, theta)distribution of consistent trajectories points + intersection distribution for points that have bigger than 7.5 degree heading diff.
    if (center_line.size() < 2) {
        cout << "size less than 2" << endl;
        return false;
    }
    
    computeConsistentPointSet(radius,
                              trajectories,
                              center_line,
                              candidate_point_set,
                              is_oneway,
                              grow_backward);
    
    return true;
}

bool branchPrediction(float               search_radius,
                      RoadPt&             start_point,
                      set<int>&           candidate_set,
                      Trajectories*       trajectories,
                      RoadPt&             junction_loc,
                      vector<vector<RoadPt> >&     branches,
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
    
    // Initialize point cloud
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    // Add points to point cloud
    for (set<int>::iterator it = candidate_set.begin(); it != candidate_set.end(); ++it) {
        PclPoint& pt = trajectories->data()->at(*it);
        points->push_back(pt);
    }
    
    search_tree->setInputCloud(points);
    
    vector<int> is_covered(points->size(), false);
    
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
    
    // Partition the points
    PclPointCloud::Ptr left_points(new PclPointCloud);
    PclPointCloud::Ptr right_points(new PclPointCloud);
    
    for (int i = 0; i < points->size(); ++i) {
        if (is_covered[i]) {
            continue;
        }
        
        PclPoint& pt = points->at(i);
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
        
        Eigen::Vector2d branch_pt(r_pt.x, r_pt.y);
        
        // Trace left branch
        bool has_left_branch = false;
        bool has_right_branch = false;
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
            Eigen::Vector2d nxt_loc = branch_pt + 10.0f * left_branch_dir;
            
            left_branch.push_back(RoadPt(nxt_loc.x(),
                                         nxt_loc.y(),
                                         search_pt.head,
                                         false));
            
            branches.push_back(left_branch);
            has_left_branch = true;
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
            Eigen::Vector2d nxt_loc = branch_pt + 10.0f * right_branch_dir;
            right_branch.push_back(RoadPt(nxt_loc.x(),
                                          nxt_loc.y(),
                                          search_pt.head,
                                          false));
            
            branches.push_back(right_branch);
            has_right_branch = true;
        }
        
        if (has_left_branch || has_right_branch) {
            return true;
        }
        else{
            return false;
        }
    }
    else{
        branches.push_back(first_branch);
        return false;
    }
}

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
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    // Add points to point cloud
    for (set<int>::iterator it = candidate_set.begin(); it != candidate_set.end(); ++it) {
        PclPoint& pt = trajectories->data()->at(*it);
        points->push_back(pt);
    }
    
    search_tree->setInputCloud(points);
    
    vector<int> is_covered(points->size(), false);
    
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
            vector<Vertex> fwd_center_line;
            for (int s = 0; s <= this_point_idx; ++s) {
                int prev_pt_idx = way_point_idxs[s];
                PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
                
                fwd_center_line.push_back(Vertex(prev_pt.x, prev_pt.y, prev_pt.head));
                
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
                                                        candidate_point_set,
                                                          osmMap_->ways()[way_id].isOneway());
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
            vector<Vertex> bwd_center_line;
            for (int s = this_point_idx; s < way_point_idxs.size(); ++s) {
                int prev_pt_idx = way_point_idxs[s];
                PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
                
                bwd_center_line.push_back(Vertex(prev_pt.x, prev_pt.y, prev_pt.head));
                
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
                    osmMap_->ways()[way_id].isOneway(),
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
        vector<Vertex> fwd_center_line;
        for (int s = 0; s <= this_point_idx; ++s) {
            int prev_pt_idx = way_point_idxs[s];
            PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
            
            fwd_center_line.push_back(Vertex(prev_pt.x, prev_pt.y, prev_pt.head));
        }
    
        if (fwd_center_line.size() > 2) {
            query_q_sample_type fwd_feature;
            bool has_feature = computeQueryQFeatureAt(radius,
                                                      trajectories_,
                                                      fwd_feature,
                                                      fwd_center_line,
                                                      osmMap_->ways()[way_id].isOneway());
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
        vector<Vertex> bwd_center_line;
        for (int s = this_point_idx; s < way_point_idxs.size(); ++s) {
            int prev_pt_idx = way_point_idxs[s];
            PclPoint& prev_pt = map_sample_points->at(prev_pt_idx);
            
            bwd_center_line.push_back(Vertex(prev_pt.x, prev_pt.y, prev_pt.head));
        }
        
        if (bwd_center_line.size() > 2) {
            query_q_sample_type bwd_feature;
            bool has_feature = computeQueryQFeatureAt(radius,
                                                      trajectories_,
                                                      bwd_feature,
                                                      bwd_center_line,
                                                      osmMap_->ways()[way_id].isOneway(),
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
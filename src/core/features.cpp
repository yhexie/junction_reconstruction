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

#include <algorithm>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/topological_sort.hpp>

using namespace boost;

float branchFitting(const vector<RoadPt>&          centerline,
                    PclPointCloud::Ptr&            points,
                    PclSearchTree::Ptr&            search_tree,
                    std::shared_ptr<Trajectories>& trajectories,
                    vector<vector<RoadPt> >&       branches,
                    bool                           grow_backward){
    if (points->size() == 0) {
        return 0.0f;
    }
    
    branches.clear();
    
    PclPointCloud::Ptr simplified_points(new PclPointCloud);
    PclSearchTree::Ptr simplified_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    /*
     *Sample the points to form simplified_points and simplified_search_tree
     */
    sampleGPSPoints(5.0f,
                    7.5f,
                    points,
                    search_tree,
                    simplified_points,
                    simplified_search_tree);
    
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*simplified_points, min_pt, max_pt);
    min_pt[0] -= 10.0f;
    max_pt[0] += 10.0f;
    min_pt[1] -= 10.0f;
    max_pt[1] += 10.0f;
    
    /*
     *Voting to form grid_points
     */
    PclPointCloud::Ptr grid_points(new PclPointCloud);
    PclSearchTree::Ptr grid_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    float delta            = Parameters::getInstance().roadVoteGridSize();
    int n_x                = floor((max_pt[0] - min_pt[0]) / delta + 0.5f) + 1;
    int n_y                = floor((max_pt[1] - min_pt[1]) / delta + 0.5f) + 1;
    float sigma_h          = Parameters::getInstance().roadSigmaH();
    float sigma_w          = Parameters::getInstance().roadSigmaW();
    
    int half_search_window = floor(sigma_h / delta + 0.5f);
    int N_ANGLE_BINS       = 24;
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

            vector<int> peak_idxs;
            peakDetector(votes,
                         4,
                         1.5f,
                         peak_idxs,
                         true);
            
            if (peak_idxs.size() >0) {
                for (const auto& idx : peak_idxs) { 
                    PclPoint pt;
                    int pt_i   = grid_pt_idx / n_y;
                    int pt_j   = grid_pt_idx % n_y;
                    float pt_x = (pt_i + 0.5f) * delta + min_pt[0];
                    float pt_y = (pt_j + 0.5f) * delta + min_pt[1];

                    float normalized_vote = votes[idx] / max_vote;
                    pt.setCoordinate(pt_x, pt_y, 0.0f);
                    pt.head = floor((idx + 0.5f) * 15.0f);
                    pt.head %= 360;
                    grid_points->push_back(pt);
                    grid_votes.push_back(normalized_vote);
                } 
            }
        }
    }
    
    if(grid_points->size() > 0){
        grid_point_search_tree->setInputCloud(grid_points);
    }
    
    // Generate paths
    float VOTE_THRESHOLD = Parameters::getInstance().roadVoteThreshold();
    vector<bool> is_local_maximum(grid_points->size(), false);
    for(size_t i = 0; i < centerline.size(); ++i){
        is_local_maximum[i] = true;
    }
    
    // Extract Local Maxima
    float MAX_DELTA_HEADING = 5.0f;
    for (size_t i = centerline.size(); i < grid_points->size(); ++i) {
        if(grid_votes[i] < VOTE_THRESHOLD){
            continue;
        }

        PclPoint& g_pt = grid_points->at(i);
        Eigen::Vector2d dir = headingTo2dVector(g_pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        grid_point_search_tree->radiusSearch(g_pt, 25.0f, k_indices, k_dist_sqrs);
        bool is_lateral_max = true;
        for (size_t j = 0; j < k_indices.size(); ++j) {
            if (k_indices[j] == i) {
                continue;
            }
            
            PclPoint& nb_g_pt = grid_points->at(k_indices[j]);
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, g_pt.head));
            
            if (delta_heading > MAX_DELTA_HEADING) {
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
        float search_radius = 20.0f;
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
            
            grid_pt_visited[grid_pt_idx] = true;
            
            if(grid_pt_idx < centerline.size()){
                continue;
            }
            
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
                    float this_fwd_distance = (dot_value + perp_dist) * grid_votes[k_indices[k]];
                    if(closest_fwd_distance > this_fwd_distance){
                        closest_fwd_distance = this_fwd_distance;
                        best_fwd_candidate = k_indices[k];
                    }
                }
                
                if(dot_value < 0){
                    float this_bwd_distance = (abs(dot_value) + perp_dist) * grid_votes[k_indices[k]];
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
    vector<float>           road_scores;
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
        float road_score = 0.0f;
        for(size_t j = 0; j < sorted_cluster.size(); ++j){
            PclPoint& pt = grid_points->at(sorted_cluster[j]);
            RoadPt r_pt(pt.x,
                        pt.y,
                        pt.head);
            if(sorted_cluster[j] >= centerline.size()){
                road_score += grid_votes[sorted_cluster[j]];
            } 
            
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
            roads.emplace_back(a_road);
            road_scores.emplace_back(road_score);
        }
    }

    float max_road_score = *std::max_element(road_scores.begin(), road_scores.end()); 
    if(max_road_score > 1e-3){
        // Normalize road_scores
        for(size_t j = 0; j < road_scores.size(); ++j){
            road_scores[j] /= max_road_score;
        } 

        for(size_t i = 0; i < roads.size(); ++i){
            roads[i][0].n_lanes = 100.0f * road_scores[i];
        } 

        // Visualize voting field of the grid_points
        /*
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
    }
    
    return 0.0f;
}

bool branchPrediction(const vector<RoadPt>&          road_centerline,
                      set<int>&                      candidate_set,
                      std::shared_ptr<Trajectories>& trajectories,
                      RoadPt&                        junction_loc,
                      vector<vector<RoadPt> >&       branches,
                      bool                           grow_backward){
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

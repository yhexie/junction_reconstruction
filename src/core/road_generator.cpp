//
//  road_generator.cpp
//  junction_reconstruction
//
//  Created by Chen Chen on 1/2/15.
//
//

#include "road_generator.h"
#include <fstream>
#include <ctime>
#include <algorithm>

#include <pcl/common/geometry.h>

#include <boost/graph/astar_search.hpp>

using namespace Eigen;
using namespace boost;

RoadGenerator::RoadGenerator(QObject *parent, std::shared_ptr<Trajectories> trajectories) : 
    Renderable(parent),
    trajectories_(trajectories),
    point_cloud_(new PclPointCloud),
    search_tree_(new pcl::search::FlannSearch<PclPoint>(false)),
    simplified_traj_points_(new PclPointCloud),
    simplified_traj_point_search_tree_(new pcl::search::FlannSearch<PclPoint>(false)),
    grid_points_(new PclPointCloud),
    grid_point_search_tree_(new pcl::search::FlannSearch<PclPoint>(false)),
    road_points_(new PclPointCloud),
    road_point_search_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    tmp_ = 0;
    max_road_label_ = 0;
    max_junc_label_ = 0;
    cur_num_clusters_ = 0;
    debug_mode_ = true;
}

RoadGenerator::~RoadGenerator(){
}

void RoadGenerator::applyRules(){
}

void RoadGenerator::pointBasedVoting(){
    simplified_traj_points_->clear();
   
    sampleGPSPoints(2.5f,
                    7.5f,
                    trajectories_->data(),
                    trajectories_->tree(),
                    simplified_traj_points_,
                    simplified_traj_point_search_tree_);

    float avg_speed = trajectories_->avgSpeed(); 

    // Road voting
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*simplified_traj_points_, min_pt, max_pt);
    min_pt[0] -= 10.0f;
    max_pt[0] += 10.0f;
    min_pt[1] -= 10.0f;
    max_pt[1] += 10.0f;
    
    float delta = Parameters::getInstance().roadVoteGridSize();
    float vote_threshold = Parameters::getInstance().roadVoteThreshold();
    
    int n_x = floor((max_pt[0] - min_pt[0]) / delta + 0.5f) + 1;
    int n_y = floor((max_pt[1] - min_pt[1]) / delta + 0.5f) + 1;
    
   
    float sigma_h = Parameters::getInstance().roadSigmaH();
    float sigma_w = Parameters::getInstance().roadSigmaW();
    
    cout << "Starting point based road voting: \n\tsigma_h= " << sigma_h <<
            "m, \tsigma_w= " << sigma_w <<
            "m, \tgrid_size= " << delta << "m" << endl;
    
    int half_search_window = floor(sigma_h / delta + 0.5f);
    
    int N_ANGLE_BINS = 24;
    float D_HEADING_BIN = 360.0f / N_ANGLE_BINS;
    map<int, vector<float> > grid_angle_votes;
    
    float max_vote = 0.0f;
    for (size_t i = 0; i < simplified_traj_points_->size(); ++i) {
        PclPoint& pt = simplified_traj_points_->at(i);
        Eigen::Vector2d pt_dir = headingTo2dVector(pt.head);

        int adjusted_pt_head = static_cast<int>(pt.head + 0.5f * D_HEADING_BIN);
        adjusted_pt_head %= 360;
        int heading_bin_idx = floor(adjusted_pt_head / D_HEADING_BIN);
        
        int pt_i = floor((pt.x - min_pt[0]) / delta);
        int pt_j = floor((pt.y - min_pt[1]) / delta);
        
        for(int pi = pt_i - half_search_window; pi <= pt_i + half_search_window; ++pi){
            if (pi < 0 || pi >= n_x) {
                continue;
            }

            for(int pj = pt_j - half_search_window; pj <= pt_j + half_search_window; ++pj){
                if (pj < 0 || pj >= n_y) {
                    continue;
                }
                
                int grid_pt_idx = pj + n_y * pi;
                
                float grid_pt_x = (pi + 0.5f) * delta + min_pt[0];
                float grid_pt_y = (pj + 0.5f) * delta + min_pt[1];
                
                Eigen::Vector2d vec(grid_pt_x - pt.x,
                                    grid_pt_y - pt.y);
                
                float dot_value_sqr = pow(vec.dot(pt_dir), 2.0f);
                float perp_value_sqr = vec.dot(vec) - dot_value_sqr;
                
                float adjusted_sigma_w = sigma_w;
                if(pt.speed < 1.5f * avg_speed && pt.speed > 1e-3){
                    adjusted_sigma_w = sigma_w * avg_speed / (pt.speed + 0.1f);
                }

                float vote = pt.id_sample *
                exp(-1.0f * dot_value_sqr / 2.0f / sigma_h / sigma_h) *
                exp(-1.0f * perp_value_sqr / 2.0f / adjusted_sigma_w / adjusted_sigma_w);
                
                if (vote > 0.0f) {
                    for (int s = heading_bin_idx - 1; s <= heading_bin_idx + 1; ++s) {
                        int corresponding_bin_idx = s;
                        if (s < 0) {
                            corresponding_bin_idx += N_ANGLE_BINS;
                        }
                        if(s >= N_ANGLE_BINS){
                            corresponding_bin_idx %= N_ANGLE_BINS;
                        }
                        
                        float bin_center = (corresponding_bin_idx) * D_HEADING_BIN;
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
    
    grid_votes_.clear();
    grid_points_->clear();
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
            
            if(peak_idxs.size() > 0){
                for (const auto& idx : peak_idxs) { 
                    PclPoint pt;
                    
                    int pt_i = grid_pt_idx / n_y;
                    int pt_j = grid_pt_idx % n_y;
                    float pt_x = (pt_i + 0.5f) * delta + min_pt[0];
                    float pt_y = (pt_j + 0.5f) * delta + min_pt[1];

                    float normalized_vote = votes[idx] / max_vote;
                    if(normalized_vote < vote_threshold){
                        continue;
                    }

                    pt.setCoordinate(pt_x, pt_y, 0.0f);
                    pt.head = floor((idx) * D_HEADING_BIN);
                    pt.head %= 360;
                    grid_points_->push_back(pt);
                    grid_votes_.push_back(normalized_vote);
                } 
            }
        }
    }
    
    if(grid_points_->size() > 0){
        grid_point_search_tree_->setInputCloud(grid_points_);
    }
    
    // Visualization
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
    
    for (size_t i = 0; i < grid_points_->size(); ++i) {
        if(grid_votes_[i] < 1e-3){
            continue;
        }
        PclPoint& g_pt = grid_points_->at(i);
        
        points_to_draw_.push_back(SceneConst::getInstance().normalize(g_pt.x, g_pt.y, Z_DEBUG));
        point_colors_.push_back(ColorMap::getInstance().getJetColor(grid_votes_[i]));
        
        Eigen::Vector2d e1 = headingTo2dVector(g_pt.head);
        
        lines_to_draw_.push_back(SceneConst::getInstance().normalize(g_pt.x, g_pt.y, Z_DEBUG));
        line_colors_.push_back(ColorMap::getInstance().getJetColor(grid_votes_[i]));
        lines_to_draw_.push_back(SceneConst::getInstance().normalize(g_pt.x+1.0*e1[0], g_pt.y+1.0*e1[1], Z_DEBUG));
        line_colors_.push_back(ColorMap::getInstance().getJetColor(grid_votes_[i]));
    }
}

void RoadGenerator::computeVoteLocalMaxima(){
    grid_is_local_maximum_.clear(); 
    if( grid_points_->size() == 0 )
       return; 

    if( grid_points_->size() != grid_votes_.size() ){
        cout << "ERROR: grid_points_ and grid_votes_ have different size!" << endl;
        exit(1); 
    } 

    grid_is_local_maximum_.resize(grid_points_->size(), false);

    float VOTE_THRESHOLD = Parameters::getInstance().roadVoteThreshold();
    float MAX_DELTA_HEADING = 7.5f; // in degree

    // Extract peaks
    for (size_t i = 0; i < grid_points_->size(); ++i) {
        if(grid_votes_[i] < VOTE_THRESHOLD) 
            continue;

        PclPoint& g_pt = grid_points_->at(i);
        Eigen::Vector2d dir = headingTo2dVector(g_pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        grid_point_search_tree_->radiusSearch(g_pt, 25.0f, k_indices, k_dist_sqrs);

        bool is_lateral_max = true;
        for (size_t j = 0; j < k_indices.size(); ++j) {
            if (k_indices[j] == i) {
                continue;
            }
            
            PclPoint& nb_g_pt = grid_points_->at(k_indices[j]);
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, g_pt.head));
            
            if (delta_heading > MAX_DELTA_HEADING) {
                continue;
            }
            
            Eigen::Vector2d vec(nb_g_pt.x - g_pt.x,
                                nb_g_pt.y - g_pt.y);
            
            if (abs(vec.dot(dir)) < 1.5f) {
                if (grid_votes_[k_indices[j]] > grid_votes_[i]) {
                    is_lateral_max = false;
                    break;
                }
            }
        }
        
        if (is_lateral_max) {
            grid_is_local_maximum_[i] = true;
        }
    }
} 

void RoadGenerator::pointBasedVotingVisualization(){
    // Visualization
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();

    // Extract peaks
    computeVoteLocalMaxima();
    
    for (size_t i = 0; i < grid_points_->size(); ++i) {
        PclPoint& g_pt = grid_points_->at(i);
                
        if (grid_is_local_maximum_[i]) {
            points_to_draw_.push_back(SceneConst::getInstance().normalize(g_pt.x, g_pt.y, Z_DEBUG+0.05f));
            point_colors_.push_back(ColorMap::getInstance().getJetColor(grid_votes_[i]));
        }
    }
}

bool RoadGenerator::computeInitialRoadGuess(){
    road_pieces_.clear();
   
    // Extract peaks
    computeVoteLocalMaxima(); 

    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();

    // Sort grid_votes_ with index
    vector<pair<int, float> > grid_votes;
    for(size_t i = 0; i < grid_votes_.size(); ++i){
        grid_votes.push_back(pair<int, float>(i, grid_votes_[i]));
    }
    
    sort(grid_votes.begin(), grid_votes.end(), pairCompareDescend);
    
    float STOPPING_THRESHOLD = Parameters::getInstance().roadVoteThreshold();
    
    typedef adjacency_list<vecS, vecS, undirectedS>    graph_t;
    typedef graph_traits<graph_t>::vertex_descriptor    vertex_descriptor;
    typedef graph_traits<graph_t>::edge_descriptor      edge_descriptor;
    
    graph_t G(grid_votes.size());
   
    {
        using namespace boost;
        float search_radius = 10.0f;
        vector<bool> grid_pt_visited(grid_votes.size(), false);
        for (int i = 0; i < grid_votes.size(); ++i) {
            int grid_pt_idx = grid_votes[i].first;
            if(!grid_is_local_maximum_[grid_pt_idx]){
                continue;
            }

            if(grid_votes_[grid_pt_idx] < STOPPING_THRESHOLD) 
                break;
            
            grid_pt_visited[grid_pt_idx] = true;
            float grid_pt_vote = grid_votes[i].second;
            //if(grid_pt_vote < stopping_threshold){
                //break;
            //}
            
            PclPoint& g_pt = grid_points_->at(grid_pt_idx);
            Eigen::Vector2d g_pt_dir = headingTo2dVector(g_pt.head);
            
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            grid_point_search_tree_->radiusSearch(g_pt,
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
                
                PclPoint& nb_g_pt = grid_points_->at(k_indices[k]);
                Eigen::Vector2d nb_g_pt_dir = headingTo2dVector(nb_g_pt.head);
                
                float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, g_pt.head));
                if(delta_heading > 15.0f){
                    continue;
                }
                
                Eigen::Vector2d vec(nb_g_pt.x - g_pt.x,
                                    nb_g_pt.y - g_pt.y);
                
                float dot_value = vec.dot(g_pt_dir);
                float perp_dist = sqrt(vec.dot(vec) - dot_value*dot_value);
                float vec_length = vec.norm();
                dot_value /= vec_length;

                if (dot_value > 0 && perp_dist < 7.0f) {
                    float this_fwd_distance = vec_length * (2.0f - g_pt_dir.dot(nb_g_pt_dir)) * (2.0f - dot_value);
                    int n_degree = out_degree(k_indices[k], G);

                    if(n_degree <= 1){
                        this_fwd_distance /= (1.0f + n_degree);
                        if(closest_fwd_distance > this_fwd_distance){
                            closest_fwd_distance = this_fwd_distance;
                            best_fwd_candidate = k_indices[k];
                        }
                    }
                }
               
                if(dot_value < 0 && perp_dist < 7.0f){
                    float this_bwd_distance = vec_length * (2.0f - g_pt_dir.dot(nb_g_pt_dir)) * (2.0f - dot_value);
                    int n_degree = out_degree(k_indices[k], G);
                    if(n_degree <= 1){
                        this_bwd_distance /= (1.0f + n_degree);
                        if (closest_bwd_distance > this_bwd_distance) {
                            closest_bwd_distance = this_bwd_distance;
                            best_bwd_candidate = k_indices[k];
                        }
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
                    PclPoint& target_g_pt = grid_points_->at(target_idx);
                    PclPoint& source_g_pt = grid_points_->at(best_fwd_candidate);
                    Eigen::Vector2d edge_dir(target_g_pt.x - source_g_pt.x,
                                             target_g_pt.y - source_g_pt.y);
                    edge_dir.normalize();
                    if (edge_dir.dot(g_pt_dir) < 0.1f) {
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
                    PclPoint& source_g_pt = grid_points_->at(source_idx);
                    PclPoint& target_g_pt = grid_points_->at(best_bwd_candidate);
                    Eigen::Vector2d edge_dir(target_g_pt.x - source_g_pt.x,
                                             target_g_pt.y - source_g_pt.y);
                    edge_dir.normalize();
                    if (edge_dir.dot(g_pt_dir) < 0.1f) {
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

    // Visualize graph
    //auto es = edges(G);
    //for(auto eit = es.first; eit != es.second; ++eit){
    //    int source_idx = source(*eit, G);
    //    int target_idx = target(*eit, G);
    //    PclPoint& source_g_pt = grid_points_->at(source_idx);
    //    PclPoint& target_g_pt = grid_points_->at(target_idx);
    //    lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(source_g_pt.x, source_g_pt.y, Z_DEBUG - 0.01f));
    //    line_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
    //    lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(target_g_pt.x, target_g_pt.y, Z_DEBUG - 0.01f));
    //    line_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
    //}
    
    //return true;

    // Compute connected component
    vector<int> component(num_vertices(G));
    int         num = connected_components(G, &component[0]);
    
    vector<vector<int> > clusters(num, vector<int>());
    vector<pair<int, float> > cluster_scores(num, pair<int, float>(0, 0.0f));
    for(int i = 0; i < num; ++i){
        cluster_scores[i].first = i; 
    }

    for (int i = 0; i != component.size(); ++i){
        clusters[component[i]].emplace_back(i);
        cluster_scores[component[i]].second += grid_votes_[i];
    }

    // Sort cluster according to their scores
    sort(cluster_scores.begin(), cluster_scores.end(), pairCompareDescend);
    
    // Trace roads
    vector<vector<int> > sorted_clusters;
    vector<vector<RoadPt> > roads;
    for (size_t i = 0; i < cluster_scores.size(); ++i) {
        vector<int>& cluster = clusters[cluster_scores[i].first];
        float cluster_score = cluster_scores[i].second;
        if(cluster.size() < 10){
            continue;
        }
        
        vector<int> sorted_cluster;
        
        // Find source
        int source_idx = -1;
        for (size_t j = 0; j < cluster.size(); ++j) {
            source_idx = cluster[j];
            PclPoint& source_pt = grid_points_->at(source_idx);
            if(out_degree(source_idx, G) == 1){
                // Check edge direction
                auto es = out_edges(source_idx, G);
                bool is_head = true;
                for (auto eit = es.first; eit != es.second; ++eit){
                    int target_idx = target(*eit, G);
                    PclPoint& target_pt = grid_points_->at(target_idx);
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
            PclPoint& pt = grid_points_->at(sorted_cluster[j]);
            RoadPt r_pt(pt.x,
                        pt.y,
                        pt.head);
            
            adjustRoadCenterAt(r_pt,
                               simplified_traj_points_,
                               simplified_traj_point_search_tree_,
                               trajectories_->avgSpeed(),
                               25.0f,
                               7.5f,
                               2.5f,
                               5.0f,
                               true);
          
            a_road.push_back(r_pt);
        }

        smoothCurve(a_road, false);
        // Check road length
        float cum_length = 0.0f;
        for(size_t j = 1; j < a_road.size(); ++j){
            float delta_x = a_road[j].x - a_road[j-1].x;
            float delta_y = a_road[j].y - a_road[j-1].y;
            
            cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
        }
        
        if(cum_length >= 120.0f){
            roads.push_back(a_road);
        }
    }
    
    cout << "There are " << roads.size() << " roads." << endl;
    for(size_t i = 0; i < roads.size(); ++i){
        // Smooth road width
        vector<RoadPt> a_road = roads[i];
        
        int cum_n_lanes = 0;
        for (size_t j = 0; j < a_road.size(); ++j) {
            cum_n_lanes += a_road[j].n_lanes;
        }
        
        cum_n_lanes /= a_road.size();
        
        vector<float> xs;
        vector<float> ys;
        vector<float> color_value;
        for (size_t j = 0; j < a_road.size(); ++j) {
            RoadPt& r_pt = a_road[j];
            r_pt.n_lanes = cum_n_lanes;
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5f * cum_n_lanes * LANE_WIDTH * Eigen::Vector2f(-1*direction[1], direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            xs.push_back(v1.x());
            ys.push_back(v1.y());
            color_value.emplace_back(static_cast<float>(j) / a_road.size()); 
        }
        
        for (int j = a_road.size() - 1; j >= 0; --j) {
            RoadPt& r_pt = a_road[j];
            
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5 * cum_n_lanes * LANE_WIDTH * Eigen::Vector2f(direction[1], -1.0f * direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            xs.push_back(v1.x());
            ys.push_back(v1.y());

            color_value.emplace_back(static_cast<float>(j) / a_road.size()); 
        }
        
        xs.push_back(xs[0]);
        ys.push_back(ys[0]);
        color_value.emplace_back(0.0f); 
       
        float cv = 1.0f - static_cast<float>(i) / roads.size(); 
        Color c = ColorMap::getInstance().getJetColor(cv);   

        for (size_t j = 0; j < xs.size() - 1; ++j) {
            float color_ratio = 1.0f - 0.5f * color_value[j]; 

            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j], ys[j], Z_ROAD));
            line_colors_.push_back(Color(color_ratio * c.r, color_ratio * c.g, color_ratio * c.b, 1.0f));
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j+1], ys[j+1], Z_ROAD));
            line_colors_.push_back(Color(color_ratio * c.r, color_ratio * c.g, color_ratio * c.b, 1.0f));
        }
        road_pieces_.push_back(a_road);
    }

    if(road_pieces_.size() == 0){
        cout << "WARNING from addInitialRoad: generate initial roads first!" << endl;
        return false;
    }

    //Break roads_ into small pieces as a directional graph.
        // Initialize the point cloud 
    indexed_roads_.clear(); 
    max_road_label_ = 0;
    for (size_t i = 0; i < road_pieces_.size(); ++i) { 
        vector<RoadPt> i_road = road_pieces_[i];

        vector<road_graph_vertex_descriptor> a_road_idx;
        int road_label = max_road_label_;
        road_graph_vertex_descriptor prev_vertex;
        for (size_t j = 0; j < i_road.size(); ++j) { 
            // Add vertex
            road_graph_vertex_descriptor v = boost::add_vertex(road_graph_); 
            road_graph_[v].road_label = road_label;
            road_graph_[v].pt = i_road[j]; 
            road_graph_[v].type = RoadGraphNodeType::road; 
            road_graph_[v].idx_in_road = j;

            // Add edge if j > 0
            if (j > 0) { 
                auto e = boost::add_edge(prev_vertex, v, road_graph_);  
                if(e.second){
                    float dx = i_road[j].x - i_road[j-1].x;
                    float dy = i_road[j].y - i_road[j-1].y;
                    road_graph_[e.first].length = sqrt(dx*dx + dy*dy);
                } 
            } 

            a_road_idx.emplace_back(v); 
            prev_vertex = v;
        } 
        indexed_roads_.emplace_back(a_road_idx);
        max_road_label_++;
    } 

    return true;
}

void RoadGenerator::tmpFunc(){
    if(debug_mode_){
        connectRoads();
        tmp_++;
        return;
    }
    
    recomputeRoads();

    vector<int> unexplained_pts;
    for (int i = 0; i < trajectories_->trajectories().size(); ++i){ 
        vector<int> projection;
        partialMapMatching(i, projection); 

        if(projection.size() < 2) 
            continue;

        const vector<int>& traj = trajectories_->trajectories()[i];
        if(projection[0] == -1) 
            unexplained_pts.emplace_back(traj[0]); 

        for (size_t j = 1; j < projection.size(); ++j) { 
            if(projection[j] == -1){
                unexplained_pts.emplace_back(traj[j]); 
                continue; 
            }
        }
    }

    for (size_t i = 0; i < unexplained_pts.size(); ++i) { 
        PclPoint& pt = trajectories_->data()->at(unexplained_pts[i]);
        points_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_TRAJECTORIES + 0.01f));
        point_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE));
    } 

    float explained_pt_percent = 100 - static_cast<float>(unexplained_pts.size()) / trajectories_->data()->size() * 100.0f;
    cout << explained_pt_percent << "%% data point explained." << endl;
}

bool RoadGenerator::updateRoadPointCloud(){
    road_points_->clear();
    auto vs = vertices(road_graph_);
    for(auto vit = vs.first; vit != vs.second; ++vit){
        RoadPt& r_pt = road_graph_[*vit].pt;
        PclPoint pt;
        pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
        pt.head = r_pt.head;
        pt.id_trajectory = road_graph_[*vit].road_label;
        pt.t             = r_pt.n_lanes;
        if(road_graph_[*vit].type == RoadGraphNodeType::road){
            pt.id_sample = road_graph_[*vit].idx_in_road;
        }
        else{
            pt.id_sample = -1;
        }
        road_points_->push_back(pt);
    }

    if(road_points_->size() > 0){
        road_point_search_tree_->setInputCloud(road_points_); 
        cout << "num of points: " << road_points_->size() << endl;
    } 
    else{
        cout << "WARNING from addInitialRoad: road_points is an empty point cloud!" << endl;
        return false;
    }

    return true;
}

bool RoadGenerator::addInitialRoad(){
    if(!updateRoadPointCloud()){
        cout << "Failed to update road_points_." << endl;
        return false;
    }

    // Visualize trajectory projection
    //int traj_idx = tmp_;
    //vector<int> projection;
    //partialMapMatching(traj_idx, projection);

    //// Visualize trajectory
    //feature_vertices_.clear();
    //feature_colors_.clear();
    //points_to_draw_.clear();
    //point_colors_.clear();
    //lines_to_draw_.clear();
    //line_colors_.clear();

    //const vector<int> traj = trajectories_->trajectories()[traj_idx]; 
    //for(size_t i = 0; i < traj.size(); ++i){
    //    PclPoint& pt = trajectories_->data()->at(traj[i]); 
    //    points_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
    //    point_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
    //    if(i > 0){
    //        PclPoint& prev_pt = trajectories_->data()->at(traj[i-1]); 
    //        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(prev_pt.x, prev_pt.y, Z_DEBUG));
    //        line_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
    //        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
    //        line_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
    //    } 
    //}

    //// Draw projection
    //for(size_t i = 0; i < projection.size(); ++i){
    //    PclPoint& pt = trajectories_->data()->at(traj[i]); 
    //    if(projection[i] != -1){
    //        PclPoint& r_pt = road_points_->at(projection[i]); 
    //        feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
    //        feature_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(r_pt.id_trajectory));
    //    } 
    //    else{
    //        feature_vertices_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
    //        feature_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
    //    }
    //}

    //tmp_++;

    //return false;

    //vector<bool> point_explained(trajectories_->data()->size(), false);
    
    map<pair<int, int>, pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>> additional_edges;
    map<pair<int, int>, int>   additional_edge_support;
    // Partial map matching each trajectory
    vector<int> unexplained_pts;
    int max_extension = 40; // look back and look forward
    for (size_t i = 0; i < trajectories_->trajectories().size(); ++i){ 
        vector<int> projection;
        partialMapMatching(i, projection); 

        if(projection.size() < 2) 
            continue;

        const vector<int>& traj = trajectories_->trajectories()[i]; 
        
        // Add edges
        int prev_road_pt_idx = projection[0];
        if(projection[0] == -1) 
            unexplained_pts.emplace_back(traj[0]); 

        float prev_road_pt_t   = trajectories_->data()->at(traj[0]).t;
        for (size_t j = 1; j < projection.size(); ++j) { 
            if(projection[j] == -1){
                unexplained_pts.emplace_back(traj[j]); 
                continue; 
            }

            PclPoint& traj_pt = trajectories_->data()->at(traj[j]);
            float delta_t = traj_pt.t - prev_road_pt_t;
            if(prev_road_pt_idx != -1 && delta_t < 30.0f){
                // Add an edge from prev_road_pt_idx to traj_road_projection[j]
                PclPoint& first_pt = road_points_->at(prev_road_pt_idx);
                PclPoint& second_pt = road_points_->at(projection[j]);

                if(first_pt.id_trajectory != second_pt.id_trajectory) {
                    int first_road_idx = first_pt.id_trajectory;
                    int second_road_idx = second_pt.id_trajectory;

                    if(additional_edges.find(pair<int, int>(first_road_idx, second_road_idx)) == additional_edges.end()){
                        vector<road_graph_vertex_descriptor>& first_road = indexed_roads_[first_road_idx];
                        vector<road_graph_vertex_descriptor>& second_road = indexed_roads_[second_road_idx];

                        float min_dist = 1e9; 
                        road_graph_vertex_descriptor min_first_idx, min_second_idx;
                        for(int s = first_pt.id_sample - max_extension; s <= first_pt.id_sample + max_extension; ++s){
                            if(s < 0 || s >= first_road.size())
                                continue;
                            RoadPt& new_first_pt = road_graph_[first_road[s]].pt;
                            for(int t = second_pt.id_sample - max_extension; t < second_pt.id_sample + max_extension; ++t){
                                if(t < 0 || t >= second_road.size())
                                    continue;
                                RoadPt& new_second_pt = road_graph_[second_road[t]].pt;
                                Eigen::Vector2d vec(new_second_pt.x - new_first_pt.x,
                                                    new_second_pt.y - new_first_pt.y);

                                float length = vec.norm();
                                if(length < min_dist){
                                    min_first_idx = first_road[s];
                                    min_second_idx = second_road[t];
                                    min_dist = length; 
                                }
                            }
                        }

                        if(min_dist < 25.0f){
                            additional_edges[pair<int, int>(first_road_idx, second_road_idx)] = pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>(min_first_idx, min_second_idx);
                            additional_edge_support[pair<int, int>(first_road_idx, second_road_idx)] = 1;
                        }
                    }
                    else{
                        additional_edge_support[pair<int, int>(first_road_idx, second_road_idx)] += 1;
                    }
                }
            } 
             
            prev_road_pt_idx = projection[j];
            prev_road_pt_t = traj_pt.t;
        } 
    } 

    cout << additional_edges.size() << " edges." << endl; 

    // add auxiliary edges
    for (const auto& edge : additional_edges) { 
        //if(additional_edge_support[edge.first] < 2)
        //    continue;

        road_graph_vertex_descriptor source_pt_idx = edge.second.first;
        road_graph_vertex_descriptor target_pt_idx = edge.second.second;

        auto e = add_edge(source_pt_idx, target_pt_idx, road_graph_);

        if(e.second){
            road_graph_[e.first].type = RoadGraphEdgeType::auxiliary;
        }
    }

    //Visualization
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();

    //for (size_t i = 0; i < unexplained_pts.size(); ++i) { 
    //    PclPoint& pt = trajectories_->data()->at(unexplained_pts[i]);
    //    points_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
    //    point_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
    //} 

    //return true;
    float explained_pt_percent = 100 - static_cast<float>(unexplained_pts.size()) / trajectories_->data()->size() * 100.0f;
    cout << "First time projection, " << explained_pt_percent << "% points explained." << endl;

    // Junction cluster
    float search_radius = Parameters::getInstance().searchRadius();

    PclPointCloud::Ptr edge_points(new PclPointCloud);
    PclSearchTree::Ptr edge_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));

    {
        using namespace boost;
        typedef adjacency_list<vecS, vecS, undirectedS >    graph_t;
        typedef graph_traits<graph_t>::vertex_descriptor    vertex_descriptor;
        typedef graph_traits<graph_t>::edge_descriptor      edge_descriptor;
        
        graph_t G;

        for (const auto& edge : additional_edges) { 
            //if(additional_edge_support[edge.first] < 2)
            //    continue;

            PclPoint first_r_pt, second_r_pt;
            
            RoadGraphNode& first_node = road_graph_[edge.second.first];
            RoadGraphNode& second_node = road_graph_[edge.second.second];

            first_r_pt.setCoordinate(first_node.pt.x, first_node.pt.y, 0.0f);
            first_r_pt.head = first_node.pt.head;
            first_r_pt.t = first_node.pt.n_lanes;
            first_r_pt.id_trajectory = first_node.road_label;
            first_r_pt.id_sample = edge.second.first;

            second_r_pt.setCoordinate(second_node.pt.x, second_node.pt.y, 0.0f);
            second_r_pt.head = second_node.pt.head;
            second_r_pt.t = second_node.pt.n_lanes;
            second_r_pt.id_trajectory = second_node.road_label;
            second_r_pt.id_sample = edge.second.second;

            int start_v_idx = edge_points->size();
            edge_points->push_back(first_r_pt);
            edge_points->push_back(second_r_pt);

            vertex_descriptor source_v = boost::add_vertex(G); 
            vertex_descriptor target_v = boost::add_vertex(G); 
            
            add_edge(source_v, target_v, G);
        }

        if(edge_points->size() > 0)
            edge_point_search_tree->setInputCloud(edge_points); 
        else{
            cout << "No edge to add." << endl;
            return true;
        }

        // Add edge for closeby same road points
        for (size_t i = 0; i < edge_points->size(); ++i) { 
            PclPoint& e_pt = edge_points->at(i);
            // Search nearby
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            edge_point_search_tree->radiusSearch(e_pt, search_radius, k_indices, k_dist_sqrs);

            for (const auto& idx : k_indices) { 
                if(idx == i){
                    continue;
                }

                PclPoint& nb_pt = edge_points->at(idx);

                if(nb_pt.id_trajectory == e_pt.id_trajectory){
                    // add edge
                    add_edge(i, idx, G);
                }
            } 
        } 

        // Compute connected component
        vector<int> component(num_vertices(G));
        int         num = connected_components(G, &component[0]);
        cur_num_clusters_ = num;
        
        vector<vector<int> > clusters(num, vector<int>());
        for (int i = 0; i != component.size(); ++i){
            clusters[component[i]].emplace_back(i);
        }

        for(size_t i = 0; i < clusters.size(); ++i){
            vector<int>& cluster = clusters[i];
            for(size_t j = 0; j < cluster.size(); ++j){
                PclPoint& e_pt = edge_points->at(cluster[j]);
                road_graph_[e_pt.id_sample].cluster_id = i;
            }
        }
    }

    // visualization
    for (size_t i = 0; i < road_points_->size(); ++i) { 
        PclPoint& r_pt = road_points_->at(i); 
        points_to_draw_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
        point_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(r_pt.id_trajectory));
    }

    //auto vs = vertices(road_graph_);
    //for(auto vit = vs.first; vit != vs.second; ++vit){
    //    if(road_graph_[*vit].cluster_id != -1){
    //        RoadPt& r_pt = road_graph_[*vit].pt;
    //        feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
    //        feature_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(road_graph_[*vit].cluster_id));
    //    }
    //}

        // auxiliary edges
    auto es = edges(road_graph_);
    for(auto eit = es.first; eit != es.second; ++eit){
        if(road_graph_[*eit].type == RoadGraphEdgeType::auxiliary){
            road_graph_vertex_descriptor source_v = source(*eit, road_graph_);
            road_graph_vertex_descriptor target_v = target(*eit, road_graph_);

            RoadPt& source_pt = road_graph_[source_v].pt;
            RoadPt& target_pt = road_graph_[target_v].pt;

            Color c1 = ColorMap::getInstance().getNamedColor(ColorMap::RED);
            Color c2 = Color(c1.r * 0.1f, c1.g * 0.1f, c1.b * 0.1f, 1.0f);
            
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(source_pt.x, source_pt.y, Z_DEBUG));
            line_colors_.push_back(c1);
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(target_pt.x, target_pt.y, Z_DEBUG));
            line_colors_.push_back(c2);
        }
    }

    if(!debug_mode_){
        for(int i = 0; i < cur_num_clusters_; ++i){
            tmp_ = i;
            connectRoads();
        }
    }

    //recomputeRoads();

    return true;
}

void RoadGenerator::recomputeRoads(){
    indexed_roads_.clear();
    max_road_label_ = 0;
    cur_num_clusters_ = 0;
    road_points_->clear();

    vector<bool> vertex_visited(num_vertices(road_graph_), false);
    auto vs = vertices(road_graph_);
    vector<road_graph_vertex_descriptor> junctions;
    for(auto vit = vs.first; vit != vs.second; ++vit){
        if(degree(*vit, road_graph_) > 2){
            junctions.emplace_back(*vit);
        }
        if(vertex_visited[*vit])
            continue;
        vertex_visited[*vit] = true;

        vector<road_graph_vertex_descriptor> a_road;
        int old_road_label = road_graph_[*vit].road_label;
        int new_road_label = max_road_label_;
        road_graph_[*vit].road_label = new_road_label;
        a_road.emplace_back(*vit);

        // backward
        road_graph_vertex_descriptor cur_vertex = *vit;
        while(true){
            auto es = in_edges(cur_vertex, road_graph_);
            bool has_prev_node = false;
            road_graph_vertex_descriptor source_vertex;
            for(auto eit = es.first; eit != es.second; ++eit){
                source_vertex = source(*eit, road_graph_);
                if(road_graph_[source_vertex].road_label == old_road_label){
                    has_prev_node = true;
                    break;
                }
            }

            if(!has_prev_node){
                break;
            }

            road_graph_[source_vertex].road_label = new_road_label;
            a_road.emplace(a_road.begin(), source_vertex);
            vertex_visited[source_vertex] = true;

            cur_vertex = source_vertex;
        }

        // forward
        cur_vertex = *vit;
        while(true){
            auto es = out_edges(cur_vertex, road_graph_);
            bool has_nxt_node = false;
            road_graph_vertex_descriptor target_vertex;
            for(auto eit = es.first; eit != es.second; ++eit){
                target_vertex = target(*eit, road_graph_);
                if(road_graph_[target_vertex].road_label == old_road_label){
                    has_nxt_node = true;
                    break;
                }
            }

            if(!has_nxt_node){
                break;
            }

            road_graph_[target_vertex].road_label = new_road_label;
            a_road.emplace_back(target_vertex);
            vertex_visited[target_vertex] = true;

            cur_vertex = target_vertex;
        }

        if(a_road.size() >= 10){
            for (int j = 0; j < a_road.size(); ++j) { 
                road_graph_[a_road[j]].idx_in_road = j;
            } 
            indexed_roads_.emplace_back(a_road);
            max_road_label_++;
        }
    }

    // Add to point cloud
    for (size_t i = 0; i < indexed_roads_.size(); ++i) { 
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        for (size_t j = 0; j < a_road.size(); ++j) { 
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            PclPoint pt;
            pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
            pt.head = r_pt.head;
            pt.id_trajectory = road_graph_[a_road[j]].road_label;
            pt.t             = r_pt.n_lanes;
            pt.id_sample = road_graph_[a_road[j]].idx_in_road;
            road_points_->push_back(pt);
        } 
    }

    if(road_points_->size() > 0)
        road_point_search_tree_->setInputCloud(road_points_);

    // Visualization
    cout << "There are " << indexed_roads_.size() << " roads." << endl;
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
 
    for(size_t i = 0; i < indexed_roads_.size(); ++i){
        // Smooth road width
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        
        int cum_n_lanes = 0;
        for (size_t j = 0; j < a_road.size(); ++j) {
            cum_n_lanes += road_graph_[a_road[j]].pt.n_lanes;
        }
        
        cum_n_lanes /= a_road.size();
        
        vector<float> xs;
        vector<float> ys;
        vector<float> color_value;
        for (size_t j = 0; j < a_road.size(); ++j) {
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5f * cum_n_lanes * LANE_WIDTH * Eigen::Vector2f(-1*direction[1], direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            xs.push_back(v1.x());
            ys.push_back(v1.y());
            color_value.emplace_back(static_cast<float>(j) / a_road.size());
        }
        
        for (int j = a_road.size() - 1; j >= 0; --j) {
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5 * cum_n_lanes * LANE_WIDTH * Eigen::Vector2f(direction[1], -1.0f * direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            xs.push_back(v1.x());
            ys.push_back(v1.y());
            color_value.emplace_back(static_cast<float>(j) / a_road.size());
        }
        
        xs.push_back(xs[0]);
        ys.push_back(ys[0]);
        color_value.emplace_back(0.0f); 
       
        Color c = ColorMap::getInstance().getDiscreteColor(i);   

        for (size_t j = 0; j < xs.size() - 1; ++j) {
            float color_ratio = 1.0f - 0.5f * color_value[j];
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j], ys[j], Z_ROAD));
            Color this_c(c.r * color_ratio, c.g * color_ratio, c.b * color_ratio, 1.0f);
            line_colors_.push_back(this_c);
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j+1], ys[j+1], Z_ROAD));
            line_colors_.push_back(this_c);
        }
    }

    // add junctions
    for(size_t i = 0; i < junctions.size(); ++i){
        RoadPt& r_pt = road_graph_[junctions[i]].pt;
        feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
        feature_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::YELLOW));
    }
}

enum class RoadConnectionType{
    CONNECT,
    SMOOTH_JOIN,
    NON_SMOOTH_JOIN,
    OPPOSITE_ROAD,
    UNDETERMINED
};

void RoadGenerator::connectRoads(){
    /*
     *This is THE FUNCTION that modify road_graph_ structures to include junctions!!! 
     *      This function deal with one linking cluster at a time.
     */

    if(cur_num_clusters_ == 0){
        cout << "WARNING: no junction cluster. Cannot connect roads." << endl;
    }

    int cluster_id = tmp_ % cur_num_clusters_;;

    // Get all vertices in the cluster
    auto vs = vertices(road_graph_);
    vector<road_graph_vertex_descriptor> cluster_vertices;
    for(auto vit = vs.first; vit != vs.second; ++vit){
        if(road_graph_[*vit].cluster_id == cluster_id){
            cluster_vertices.emplace_back(*vit);
        }
    }

    // Trace road segments of the related roads from road_graph_ 
    set<road_graph_vertex_descriptor> visited_vertices;
    vector<vector<road_graph_vertex_descriptor>> roads;
    int max_extension = 20;
    map<road_graph_vertex_descriptor, int> vertex_to_road_map;
    for (size_t i = 0; i < cluster_vertices.size(); ++i) { 
        road_graph_vertex_descriptor start_v = cluster_vertices[i];
        if(visited_vertices.find(start_v) != visited_vertices.end())
            continue;

        vector<road_graph_vertex_descriptor> a_road;
        a_road.emplace_back(start_v);

        // backward
        road_graph_vertex_descriptor cur_v = start_v;
        int road_label = road_graph_[cur_v].road_label;
        for(int j = 0; j < max_extension; ++j){
            auto es = in_edges(cur_v, road_graph_);
            
            bool has_prev_node = false;
            for (auto eit = es.first; eit != es.second; ++eit){
                road_graph_vertex_descriptor source_v = source(*eit, road_graph_);
                if(road_graph_[source_v].road_label == road_label){
                    a_road.emplace(a_road.begin(), source_v);
                    has_prev_node = true;
                    cur_v = source_v;
                    break;
                }
            }

            if(!has_prev_node)
                break;
        }

        // forward 
        cur_v = start_v;
        for(int j = 0; j < max_extension; ++j){
            auto es = out_edges(cur_v, road_graph_);
            
            bool has_nxt_node = false;
            for (auto eit = es.first; eit != es.second; ++eit){
                road_graph_vertex_descriptor target_v = target(*eit, road_graph_);
                if(road_graph_[target_v].road_label == road_label){
                    a_road.emplace_back(target_v);
                    has_nxt_node = true;
                    cur_v = target_v;
                    break;
                }
            }

            if(!has_nxt_node)
                break;
        }

        for (size_t j = 0; j < a_road.size(); ++j) { 
            visited_vertices.emplace(a_road[j]);
            vertex_to_road_map[a_road[j]] = roads.size();
        } 
        roads.emplace_back(a_road);
    } 

    // Detect existance of prior structures, such as roads running in opposite direction, near + crossing, etc. 
        // Get heading distribution
    int N_HEADING_BINS = 180;
    float delta_heading_bin = 360 / N_HEADING_BINS;
    vector<float> heading_hist(N_HEADING_BINS, 0.0f);
    PclPointCloud::Ptr junction_points(new PclPointCloud);
    PclSearchTree::Ptr junction_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    map<int, int> road_majority_heading;
    for (size_t i = 0; i < roads.size(); ++i) { 
        vector<road_graph_vertex_descriptor>& a_road = roads[i];
        vector<float> tmp_heading_hist(N_HEADING_BINS, 0.0f);
        // Each road only vote its majority direction
        for (size_t j = 0; j < a_road.size(); ++j) { 
            PclPoint pt;
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
            pt.head = r_pt.head;
            pt.id_trajectory = i;
            pt.id_sample = a_road[j];
            junction_points->push_back(pt);
            
            int shifted_head = floor(r_pt.head + delta_heading_bin / 2.0);
            shifted_head %= 360;
            int heading_bin_idx = floor(shifted_head / delta_heading_bin);
            for(int k = heading_bin_idx - 8; k <= heading_bin_idx + 8; ++k){
                int mk = (k + N_HEADING_BINS) % N_HEADING_BINS;
                float bin_center = mk * delta_heading_bin;
                float delta_h = deltaHeading1MinusHeading2(bin_center, r_pt.head);

                float vote = exp(-1.0f * delta_h * delta_h / 2.0f / 5.0f / 5.0f);
                if(vote < 0.1f)
                    vote = 0.0f;
                tmp_heading_hist[mk] += vote;
            }
        } 

        vector<int> tmp_peak_idxs;
        peakDetector(tmp_heading_hist,
                     15,
                     1.5f,
                     tmp_peak_idxs,
                     true);

        if(tmp_peak_idxs.size() > 0){
            int max_peak_idx = -1;
            float max_vote = 0.0f;
            for (const auto& p : tmp_peak_idxs) { 
                if(tmp_heading_hist[p] > max_vote){
                    max_vote = tmp_heading_hist[p];
                    max_peak_idx = p;
                }
            } 

            if(max_peak_idx != -1){
                road_majority_heading[i] = floor(max_peak_idx * delta_heading_bin);
                for(int s = max_peak_idx - 10; s <= max_peak_idx + 10; ++s){
                    int ms = (s + N_HEADING_BINS) % N_HEADING_BINS;
                    heading_hist[ms] += tmp_heading_hist[ms];
                }
            }
        }
    } 

    vector<int> peak_idxs;
    peakDetector(heading_hist,
                 15,
                 1.5f,
                 peak_idxs,
                 true);

    cout << "Peaks: " << endl;
    cout << "\t";
    for (const auto& idx : peak_idxs) { 
        cout << "(" << floor(idx * delta_heading_bin) << ", " << heading_hist[idx] << "), ";
    } 
    cout << endl;

    int ANGLE_PERP_THRESHOLD = 30;
    int ANGLE_OPPO_THRESHOLD = 10;
    if(peak_idxs.size() >= 2){
        vector<pair<int, int>> near_perp_peaks;
        vector<pair<int, int>> near_oppo_peaks;
        for (size_t i = 0; i < peak_idxs.size(); ++i) { 
            int h1 = floor(peak_idxs[i] * delta_heading_bin);
            for (size_t j = i+1; j < peak_idxs.size(); ++j) { 
                int h2 = floor(peak_idxs[j] * delta_heading_bin);
                float dh = abs(deltaHeading1MinusHeading2(h1, h2));
                if(dh < 90 + ANGLE_PERP_THRESHOLD && dh > 90 - ANGLE_PERP_THRESHOLD){
                    near_perp_peaks.emplace_back(pair<int, int>(h1, h2));
                }
                if(dh > 180 - ANGLE_OPPO_THRESHOLD){
                    near_oppo_peaks.emplace_back(pair<int, int>(h1, h2));
                }
            } 
        } 

        if(near_perp_peaks.size() > 0){
            cout << "\tPerpendicular peaks: " << endl;
            cout << "\t\t";
            for (const auto& p : near_perp_peaks) { 
                cout << "(" << p.first << ", " << p.second << "), ";
            } 
            cout << endl;
        }

        if(near_oppo_peaks.size() > 0){
            cout << "\tOpposite peaks: " << endl;
            cout << "\t\t";
            for (const auto& p : near_oppo_peaks) { 
                cout << "(" << p.first << ", " << p.second << "), ";
            } 
            cout << endl;
        }

        // Adjust opposite roads
        for (const auto& p : near_oppo_peaks) { 
            continue;
            int h1 = p.first;
            int h2 = (p.first + 180) % 360;
            cout << "\t\th1 = " << h1 << endl;

            vector<int> h1_consistent_road;
            vector<int> h2_consistent_road;
            for (const auto& r : road_majority_heading) { 
                float dh1 = abs(deltaHeading1MinusHeading2(h1, r.second));
                if(dh1 < 7.5f){
                    h1_consistent_road.emplace_back(r.first);
                }
                float dh2 = abs(deltaHeading1MinusHeading2(h2, r.second));
                if(dh2 < 7.5f){
                    h2_consistent_road.emplace_back(r.first);
                }
            } 

            if(h1_consistent_road.size() == 0 || h2_consistent_road.size() == 0)
                continue;

            map<int, float> h1_r_center_x;
            map<int, float> h1_r_center_y;
            // Per road adjustment
            int h1_n_lanes = 0;
            for (const auto& r_idx : h1_consistent_road) { 
                float cum_x = 0.0f;
                float cum_y = 0.0f;
                int n_pt = 0;
                vector<road_graph_vertex_descriptor>& a_road = roads[r_idx];
                for (size_t j = 0; j < a_road.size(); ++j) { 
                    RoadPt& r_pt = road_graph_[a_road[j]].pt;
                    float dh = abs(deltaHeading1MinusHeading2(r_pt.head, h1));
                    if(dh < 7.5f){
                        cum_x += r_pt.x;
                        cum_y += r_pt.y;
                        n_pt++;
                    }
                } 
                if(n_pt == 0){
                    continue;
                }
                cum_x /= n_pt;
                cum_y /= n_pt;
                h1_r_center_x[r_idx] = cum_x;
                h1_r_center_y[r_idx] = cum_y;
                h1_n_lanes += road_graph_[a_road[0]].pt.n_lanes;
            } 
            h1_n_lanes /= h1_consistent_road.size();

            if(h1_r_center_x.size() == 0)
                continue;

            float h1_avg_x = 0.0f; 
            float h1_avg_y = 0.0f; 
            if(h1_r_center_x.size() > 0){
                for (const auto& kv : h1_r_center_x) { 
                    h1_avg_x += kv.second;
                    h1_avg_y += h1_r_center_y[kv.first];
                } 

                h1_avg_x /= h1_r_center_x.size();
                h1_avg_y /= h1_r_center_y.size();
            }

            map<int, float> h2_r_center_x;
            map<int, float> h2_r_center_y;
            // Per road adjustment
            int h2_n_lanes = 0;
            for (const auto& r_idx : h2_consistent_road) { 
                float cum_x = 0.0f;
                float cum_y = 0.0f;
                int n_pt = 0;
                vector<road_graph_vertex_descriptor>& a_road = roads[r_idx];
                for (size_t j = 0; j < a_road.size(); ++j) { 
                    RoadPt& r_pt = road_graph_[a_road[j]].pt;
                    float dh = abs(deltaHeading1MinusHeading2(r_pt.head, h2));
                    if(dh < 7.5f){
                        cum_x += r_pt.x;
                        cum_y += r_pt.y;
                        n_pt++;
                    }
                } 
                if(n_pt == 0){
                    continue;
                }
                cum_x /= n_pt;
                cum_y /= n_pt;
                h2_r_center_x[r_idx] = cum_x;
                h2_r_center_y[r_idx] = cum_y;
                h2_n_lanes += road_graph_[a_road[0]].pt.n_lanes;
            } 
            h2_n_lanes /= h2_consistent_road.size();

            if(h2_r_center_x.size() == 0)
                continue;

            float h2_avg_x = 0.0f; 
            float h2_avg_y = 0.0f; 
            if(h2_r_center_x.size() > 0){
                for (const auto& kv : h2_r_center_x) { 
                    h2_avg_x += kv.second;
                    h2_avg_y += h2_r_center_y[kv.first];
                } 

                h2_avg_x /= h2_r_center_x.size();
                h2_avg_y /= h2_r_center_y.size();
            }

            Eigen::Vector3d vec(h2_avg_x - h1_avg_x,
                                h2_avg_y - h1_avg_y,
                                0.0f);

            Eigen::Vector3d h1_dir = headingTo3dVector(h1); 
            Eigen::Vector3d h2_dir = headingTo3dVector(h2); 
            Eigen::Vector3d center(0.5f*(h1_avg_x + h2_avg_x),
                                   0.5f*(h1_avg_y + h2_avg_y),
                                   0.0f);
            Eigen::Vector3d h1_perp_dir(-h1_dir.y(), h1_dir.x(), 0.0f);

            float perp_dist = vec.dot(h1_perp_dir);
            cout << "perp dist: " << perp_dist << endl;
            if(perp_dist < -10.0f || perp_dist > 20.0f)
                continue;
            
            float width_ratio = 0.5f;
            if(perp_dist < 10.0f){
                width_ratio = 0.25f;
            }

            Eigen::Vector3d new_h1_center = center - width_ratio * h1_n_lanes * LANE_WIDTH * h1_perp_dir;
            Eigen::Vector3d new_h2_center = center + width_ratio * h2_n_lanes * LANE_WIDTH * h1_perp_dir;

            // Actual adjustment
            float max_angle_tolerance = 10.0f;
            float R = 100.0f;
            for (const auto& r_idx : h1_consistent_road) { 
                vector<road_graph_vertex_descriptor>& a_road = roads[r_idx];
                vector<RoadPt> new_road;
                for (size_t j = 0; j < a_road.size(); ++j) { 
                    RoadPt& r_pt = road_graph_[a_road[j]].pt;
                    RoadPt new_r_pt = r_pt;
                    float dh = abs(deltaHeading1MinusHeading2(r_pt.head, h1));
                    if(dh < max_angle_tolerance){
                        Eigen::Vector3d v(new_r_pt.x - new_h1_center.x(),
                                          new_r_pt.y - new_h1_center.y(),
                                          0.0f);
                        float dot_value = v.dot(h1_dir);
                        v = new_h1_center + dot_value * h1_dir;
                        float ratio = 1.5f - min(abs(dot_value) / R, 1.0f);
                        if(ratio < 0){
                            ratio = 0.0f;
                        }
                        if(ratio > 0.8f){
                            ratio = 0.8f;
                        }
                        int new_head = ratio * h1 + (1.0f - ratio) * new_r_pt.head;
                        new_r_pt.head = new_head;
                        float new_x = ratio * v.x() + (1.0f - ratio) * new_r_pt.x;
                        float new_y = ratio * v.y() + (1.0f - ratio) * new_r_pt.y;
                        new_r_pt.x = new_x;
                        new_r_pt.y = new_y;
                    }
                    new_road.emplace_back(new_r_pt);
                } 
                
                for(size_t j = 0; j < a_road.size(); ++j){
                    road_graph_[a_road[j]].pt = new_road[j];
                }
            }

            for (const auto& r_idx : h2_consistent_road) { 
                vector<road_graph_vertex_descriptor>& a_road = roads[r_idx];
                vector<RoadPt> new_road;
                for (size_t j = 0; j < a_road.size(); ++j) { 
                    RoadPt& r_pt = road_graph_[a_road[j]].pt;
                    RoadPt new_r_pt = r_pt;
                    float dh = abs(deltaHeading1MinusHeading2(r_pt.head, h2));
                    if(dh < max_angle_tolerance){
                        Eigen::Vector3d v(new_r_pt.x - new_h2_center.x(),
                                          new_r_pt.y - new_h2_center.y(),
                                          0.0f);
                        float dot_value = v.dot(h2_dir);
                        v = new_h2_center + dot_value * h2_dir;
                        float ratio = 1.5f - min(abs(dot_value) / R, 1.0f);
                        if(ratio < 0){
                            ratio = 0.0f;
                        }
                        if(ratio >= 0.8f){
                            ratio = 0.8f;
                        }
                        int new_head = ratio * h2 + (1.0f - ratio) * new_r_pt.head;
                        new_r_pt.head = new_head;
                        float new_x = ratio * v.x() + (1.0f - ratio) * new_r_pt.x;
                        float new_y = ratio * v.y() + (1.0f - ratio) * new_r_pt.y;
                        new_r_pt.x = new_x;
                        new_r_pt.y = new_y;
                    }
                    new_road.emplace_back(new_r_pt);
                } 
                
                for(size_t j = 0; j < a_road.size(); ++j){
                    RoadPt& r_pt = road_graph_[a_road[j]].pt;
                    r_pt = new_road[j];
                }
            } 
        } 
         
        // Adjust perp roads

        if(junction_points->size() > 0){
            junction_point_search_tree->setInputCloud(junction_points);
        }
    }

    // Visualize the traced roads
    if(debug_mode_){
        feature_vertices_.clear();
        feature_colors_.clear();
        for (size_t i = 0; i < roads.size(); ++i) { 
            vector<road_graph_vertex_descriptor>& a_road = roads[i];
            for (size_t j = 0; j < a_road.size(); ++j) { 
                RoadPt& r_pt = road_graph_[a_road[j]].pt;
                feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
                feature_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(i));
            } 
        } 
    }

    float SIGMA_H = 7.5f;
    cout << "\tDealing method:" << endl;
    for (size_t i = 0; i < cluster_vertices.size(); ++i) { 
        road_graph_vertex_descriptor cur_vertex = cluster_vertices[i];
        int cur_road_idx = vertex_to_road_map[cur_vertex];
        vector<road_graph_vertex_descriptor>& cur_road = roads[cur_road_idx];
    
        while(true){
            bool has_auxiliary_edge = false;
            auto es = out_edges(cur_vertex, road_graph_);
            road_graph_edge_descriptor aug_edge;
            for(auto eit = es.first; eit != es.second; ++eit){
                if(road_graph_[*eit].type == RoadGraphEdgeType::auxiliary){
                    aug_edge = *eit;
                    has_auxiliary_edge = true;
                    break;
                }
            }
            if(!has_auxiliary_edge){
                break;
            }

            road_graph_vertex_descriptor to_vertex = target(aug_edge, road_graph_);

            int to_road_idx = vertex_to_road_map[to_vertex];
            vector<road_graph_vertex_descriptor>& to_road = roads[to_road_idx];

            // Recompute closest points
            float min_dist = 1e6;
            int min_cur_road_idx = -1;
            int min_to_road_idx = -1;
            for(int k = 0; k < cur_road.size(); ++k){
                for(int l = 0; l < to_road.size(); ++l){
                    float delta_x = road_graph_[cur_road[k]].pt.x - road_graph_[to_road[l]].pt.x;
                    float delta_y = road_graph_[cur_road[k]].pt.y - road_graph_[to_road[l]].pt.y;
                    float d = sqrt(delta_x*delta_x + delta_y*delta_y);
                    if(min_dist > d){
                        min_dist = d;
                        min_cur_road_idx = k;
                        min_to_road_idx = l;
                    }
                }
            }

            remove_edge(aug_edge, road_graph_);

            // Remove reverse edge if any
            auto e = edge(to_vertex, cur_vertex, road_graph_);

            cur_vertex = cur_road[min_cur_road_idx];
            to_vertex = to_road[min_to_road_idx];
            
            if(e.second){
                if(road_graph_[e.first].type == RoadGraphEdgeType::auxiliary)
                    remove_edge(e.first, road_graph_);
            }

            RoadConnectionType connect_type = RoadConnectionType::UNDETERMINED;

            // Check relationship
            RoadPt& from_r_pt = road_graph_[cur_vertex].pt;
            RoadPt& to_r_pt   = road_graph_[to_vertex].pt;

            float turning_angle = abs(deltaHeading1MinusHeading2(to_r_pt.head, from_r_pt.head));
            Eigen::Vector2d cur_vertex_dir = headingTo2dVector(from_r_pt.head);
            Eigen::Vector2d to_vertex_dir = headingTo2dVector(to_r_pt.head);
            Eigen::Vector2d edge_vec(to_r_pt.x - from_r_pt.x,
                                     to_r_pt.y - from_r_pt.y);
            float edge_vec_length = edge_vec.norm();
            float dot_value = edge_vec.dot(cur_vertex_dir);
            float perp_dist = sqrt(abs(edge_vec_length * edge_vec_length - dot_value * dot_value));

            if(turning_angle < SIGMA_H && perp_dist < 10.0f){
                if(cur_road.back() == cur_vertex && to_road.front() == to_vertex)
                    connect_type = RoadConnectionType::CONNECT;
            }

            if(connect_type == RoadConnectionType::UNDETERMINED){
                if(edge_vec_length < 5.0f){
                    if(turning_angle < 30.0f){
                        connect_type = RoadConnectionType::SMOOTH_JOIN;
                    }
                    else if(turning_angle < 180.0f - SIGMA_H){
                        connect_type = RoadConnectionType::NON_SMOOTH_JOIN;
                    }
                    else{
                        connect_type = RoadConnectionType::OPPOSITE_ROAD;
                    }
                }
            }

            if(connect_type == RoadConnectionType::UNDETERMINED){
                if(turning_angle < 30.0f){
                    connect_type = RoadConnectionType::SMOOTH_JOIN;
                }
                else if(turning_angle < 180.0f - SIGMA_H){
                    connect_type = RoadConnectionType::NON_SMOOTH_JOIN;
                }
                else{
                    connect_type = RoadConnectionType::OPPOSITE_ROAD;
                }
            }
            
            // Deal with each possibility
            switch (connect_type) { 
                case RoadConnectionType::CONNECT: { 
                    cout << "\t\tCONNECT" << endl;
                    int n_v_to_add = floor(edge_vec_length / 5.0f);
                    
                    float d = edge_vec_length / (n_v_to_add + 1);
                    road_graph_vertex_descriptor prev_v = cur_vertex;
                    for(int k = 0; k < n_v_to_add; ++k){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        float new_v_x = from_r_pt.x + edge_vec.x() * d * (k+1) / edge_vec_length;
                        float new_v_y = from_r_pt.y + edge_vec.y() * d * (k+1) / edge_vec_length;
                        Eigen::Vector2d head_dir = (k+1) * cur_vertex_dir + (n_v_to_add - k) * to_vertex_dir;
                        road_graph_[new_v].pt = from_r_pt;
                        road_graph_[new_v].pt.x = new_v_x;
                        road_graph_[new_v].pt.y = new_v_y;
                        road_graph_[new_v].pt.head = vector2dToHeading(head_dir);
                        road_graph_[new_v].road_label = road_graph_[cur_vertex].road_label;

                        auto es = add_edge(prev_v, new_v, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }

                        prev_v = new_v;
                    }

                    auto es = add_edge(prev_v, to_vertex, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                    }

                    // Update road label
                    int correct_road_label = road_graph_[cur_vertex].road_label;
                    int road_label_to_correct = road_graph_[to_vertex].road_label;
                    road_graph_vertex_descriptor tmp_vertex = to_vertex;
                    while(true){
                        if(road_graph_[tmp_vertex].road_label == road_label_to_correct){
                            road_graph_[tmp_vertex].road_label = correct_road_label;
                        }

                        bool has_next_vertex = false;
                        auto out_es = out_edges(to_vertex, road_graph_);
                        for(auto out_eit = out_es.first; out_eit != out_es.second; ++out_eit){
                            road_graph_vertex_descriptor nxt_vertex = target(*out_eit, road_graph_);
                            if(road_graph_[nxt_vertex].road_label == road_label_to_correct){
                                has_next_vertex = true;
                                tmp_vertex = nxt_vertex;
                                break;
                            }
                        }

                        if(!has_next_vertex)
                            break;
                    }
                } 
                break; 
                case RoadConnectionType::SMOOTH_JOIN: {
                    cout << "\t\tSMOOTH_JOIN" << endl;
                    // Find best connection
                    road_graph_vertex_descriptor best_s, best_t; 
                    float cur_min_score = 1e9;
                    for(int s = 0; s < cur_road.size(); ++s){
                        RoadPt& s_pt = road_graph_[cur_road[s]].pt;
                        Eigen::Vector2d vec(to_r_pt.x - s_pt.x,
                                            to_r_pt.y - s_pt.y);

                        float vec_length = vec.norm();
                        int vec_head = vector2dToHeading(vec);
                        float delta_h1 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                        float delta_h2 = abs(deltaHeading1MinusHeading2(to_r_pt.head, vec_head));
                        
                        float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                        if(score < cur_min_score){
                            best_s = cur_road[s];
                            best_t = to_vertex;
                            cur_min_score = score;
                        }
                    }

                    for(int s = 0; s < to_road.size(); ++s){
                        RoadPt& t_pt = road_graph_[to_road[s]].pt;
                        Eigen::Vector2d vec(t_pt.x - from_r_pt.x,
                                            t_pt.y - from_r_pt.y);

                        float vec_length = vec.norm();
                        int vec_head = vector2dToHeading(vec);
                        float delta_h1 = abs(deltaHeading1MinusHeading2(from_r_pt.head, vec_head));
                        float delta_h2 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                        
                        float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                        if(score < cur_min_score){
                            best_s = cur_vertex;
                            best_t = to_road[s];
                            cur_min_score = score;
                        }
                    }

                    //auto e = edge(to_vertex, cur_vertex, road_graph_);
                    //if(e.second){
                    //    if(road_graph_[e.first].type == RoadGraphEdgeType::auxiliary){
                    //        for(int s = 0; s < to_road.size(); ++s){
                    //            RoadPt& s_pt = road_graph_[to_road[s]].pt;

                    //            Eigen::Vector2d vec(from_r_pt.x - s_pt.x,
                    //                                from_r_pt.y - s_pt.y);

                    //            float vec_length = vec.norm();

                    //            int vec_head = vector2dToHeading(vec);
                    //            float delta_h1 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                    //            float delta_h2 = abs(deltaHeading1MinusHeading2(from_r_pt.head, vec_head));
                                
                    //            float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);

                    //            if(score < cur_min_score){
                    //                best_s = to_road[s];
                    //                best_t = cur_vertex;
                    //                cur_min_score = score;
                    //            }
                    //        }

                    //        for(int s = 0; s < cur_road.size(); ++s){
                    //            RoadPt& t_pt = road_graph_[cur_road[s]].pt;

                    //            Eigen::Vector2d vec(t_pt.x - to_r_pt.x,
                    //                                t_pt.y - to_r_pt.y);

                    //            float vec_length = vec.norm();

                    //            int vec_head = vector2dToHeading(vec);
                    //            float delta_h1 = abs(deltaHeading1MinusHeading2(to_r_pt.head, vec_head));
                    //            float delta_h2 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                                
                    //            float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);

                    //            if(score < cur_min_score){
                    //                best_s = to_vertex;
                    //                best_t = cur_road[s];
                    //                cur_min_score = score;
                    //            }
                    //        }

                    //        remove_edge(e.first, road_graph_);
                    //    }
                    //}

                    // Connect from best_s to best_t
                    RoadPt& new_from_r_pt = road_graph_[best_s].pt;
                    RoadPt& new_to_r_pt = road_graph_[best_t].pt;
                    Eigen::Vector2d new_from_r_pt_dir = headingTo2dVector(new_from_r_pt.head);
                    Eigen::Vector2d new_to_r_pt_dir = headingTo2dVector(new_to_r_pt.head);

                    int new_from_r_pt_degree = out_degree(best_s, road_graph_);
                    int new_to_r_pt_degree = in_degree(best_t, road_graph_);

                    int new_v_road_label;
                    if(new_from_r_pt_degree >= 1){
                        if(new_to_r_pt_degree == 0){
                            new_v_road_label = road_graph_[best_t].road_label;
                        }
                        else{
                            new_v_road_label = max_road_label_;
                            max_road_label_++;
                        }
                    }
                    else{
                        new_v_road_label = road_graph_[best_s].road_label;
                    }

                    road_graph_vertex_descriptor prev_v;
                    if(new_v_road_label != road_graph_[best_t].road_label){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        road_graph_[new_v].pt = new_from_r_pt;
                        road_graph_[new_v].road_label = new_v_road_label;
                        prev_v = new_v;
                    }
                    else{
                        prev_v = best_s;
                    }

                    Eigen::Vector2d new_edge_vec(new_to_r_pt.x - new_from_r_pt.x,
                                                 new_to_r_pt.y - new_from_r_pt.y);
                    float new_edge_vec_length = new_edge_vec.norm();
                    
                    int n_v_to_add = floor(new_edge_vec_length / 5.0f);
                    
                    float d = new_edge_vec_length / (n_v_to_add + 1);
                    for(int k = 0; k < n_v_to_add; ++k){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        float new_v_x = new_from_r_pt.x + new_edge_vec.x() * (k+1) * d / new_edge_vec_length;
                        float new_v_y = new_from_r_pt.y + new_edge_vec.y() * (k+1) * d / new_edge_vec_length;
                        Eigen::Vector2d head_dir = (k+1) * new_from_r_pt_dir + (n_v_to_add - k) * new_to_r_pt_dir;

                        road_graph_[new_v].pt = new_from_r_pt;
                        road_graph_[new_v].pt.x = new_v_x;
                        road_graph_[new_v].pt.y = new_v_y;
                        road_graph_[new_v].pt.head = vector2dToHeading(head_dir);
                        road_graph_[new_v].road_label = new_v_road_label;

                        auto es = add_edge(prev_v, new_v, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }

                        prev_v = new_v;
                    }
                    
                    if(new_v_road_label != road_graph_[best_t].road_label){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        auto es = add_edge(prev_v, new_v, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }

                        visited_vertices.emplace(new_v);

                        road_graph_[new_v].pt = new_to_r_pt;
                        road_graph_[new_v].road_label = new_v_road_label;

                        es = add_edge(new_v, best_t, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        }
                    }
                    else{
                        auto es = add_edge(prev_v, best_t, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }
                    }
                }
                break;
                case RoadConnectionType::NON_SMOOTH_JOIN: {
                    cout << "\t\tNON_SMOOTH_JOIN" << endl;
                    // find the best connection point
                    road_graph_vertex_descriptor best_s, best_t; 
                    float cur_min_score = 1e9;
                    for(int s = 0; s < cur_road.size(); ++s){
                        RoadPt& s_pt = road_graph_[cur_road[s]].pt;

                        Eigen::Vector2d vec(to_r_pt.x - s_pt.x,
                                            to_r_pt.y - s_pt.y);

                        int vec_head = vector2dToHeading(vec);
                        float vec_length = vec.norm();
                        float dot_value2 = headingTo2dVector(to_r_pt.head).dot(vec) / vec_length;
                        float score = (2.0f - dot_value2) * vec_length;
                        if(score < cur_min_score){
                            best_s = cur_road[s];
                            best_t = to_vertex;
                            cur_min_score = score;
                        }
                    }

                    for(int s = 0; s < to_road.size(); ++s){
                        RoadPt& t_pt = road_graph_[to_road[s]].pt;

                        Eigen::Vector2d vec(t_pt.x - from_r_pt.x,
                                            t_pt.y - from_r_pt.y);

                        int vec_head = vector2dToHeading(vec);
                        float vec_length = vec.norm();
                        float dot_value1 = headingTo2dVector(from_r_pt.head).dot(vec) / vec_length;
                        float score = (2.0f - dot_value1) * vec_length;
                        if(score < cur_min_score){
                            best_s = cur_vertex;
                            best_t = to_road[s];
                            cur_min_score = score;
                        }
                    }

                    // Connect from best_s to best_t
                    RoadPt& new_from_r_pt = road_graph_[best_s].pt;
                    RoadPt& new_to_r_pt = road_graph_[best_t].pt;
                    Eigen::Vector2d new_from_r_pt_dir = headingTo2dVector(new_from_r_pt.head);
                    Eigen::Vector2d new_to_r_pt_dir = headingTo2dVector(new_to_r_pt.head);

                    int new_from_r_pt_degree = out_degree(best_s, road_graph_);
                    int new_to_r_pt_degree = in_degree(best_t, road_graph_);

                    int new_v_road_label;
                    if(new_from_r_pt_degree >= 1){
                        if(new_to_r_pt_degree == 0){
                            new_v_road_label = road_graph_[best_t].road_label;
                        }
                        else{
                            new_v_road_label = max_road_label_;
                            max_road_label_++;
                        }
                    }
                    else{
                        new_v_road_label = road_graph_[best_s].road_label;
                    }

                    road_graph_vertex_descriptor prev_v;
                    if(new_v_road_label != road_graph_[best_t].road_label){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        road_graph_[new_v].pt = new_from_r_pt;
                        road_graph_[new_v].road_label = new_v_road_label;
                        prev_v = new_v;
                    }
                    else{
                        prev_v = best_s;
                    }

                    Eigen::Vector2d new_edge_vec(new_to_r_pt.x - new_from_r_pt.x,
                                                 new_to_r_pt.y - new_from_r_pt.y);
                    float new_edge_vec_length = new_edge_vec.norm();
                    
                    int n_v_to_add = floor(new_edge_vec_length / 5.0f);
                    
                    float d = new_edge_vec_length / (n_v_to_add + 1);
                    for(int k = 0; k < n_v_to_add; ++k){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        float new_v_x = new_from_r_pt.x + new_edge_vec.x() * (k+1) * d / new_edge_vec_length;
                        float new_v_y = new_from_r_pt.y + new_edge_vec.y() * (k+1) * d / new_edge_vec_length;
                        Eigen::Vector2d head_dir = (k+1) * new_from_r_pt_dir + (n_v_to_add - k) * new_to_r_pt_dir;

                        road_graph_[new_v].pt = new_from_r_pt;
                        road_graph_[new_v].pt.x = new_v_x;
                        road_graph_[new_v].pt.y = new_v_y;
                        road_graph_[new_v].pt.head = vector2dToHeading(head_dir);
                        road_graph_[new_v].road_label = new_v_road_label;

                        auto es = add_edge(prev_v, new_v, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }

                        prev_v = new_v;
                    }
                    
                    if(new_v_road_label != road_graph_[best_t].road_label){
                        road_graph_vertex_descriptor new_v = add_vertex(road_graph_);

                        visited_vertices.emplace(new_v);

                        auto es = add_edge(prev_v, new_v, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }

                        visited_vertices.emplace(new_v);

                        road_graph_[new_v].pt = new_to_r_pt;
                        road_graph_[new_v].road_label = new_v_road_label;

                        es = add_edge(new_v, best_t, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        }
                    }
                    else{
                        auto es = add_edge(prev_v, best_t, road_graph_);
                        if(es.second){
                            road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        }
                    }

                    auto e = edge(to_vertex, cur_vertex, road_graph_);
                    if(e.second){
                        if(road_graph_[e.first].type == RoadGraphEdgeType::auxiliary){
                            remove_edge(e.first, road_graph_);
                        }
                    }
                }
                break;
                case RoadConnectionType::OPPOSITE_ROAD: {
                }
                break;
            
                default: { 
                } 
                break; 
            }
        }
    } 

    // Visualize edges and junctions
    if(debug_mode_){
        line_colors_.clear();
        lines_to_draw_.clear();
        for (const auto& vertex : visited_vertices) { 
        auto es = out_edges(vertex, road_graph_);

        Color c = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE);
        for(auto eit = es.first; eit != es.second; ++eit){
            road_graph_vertex_descriptor s_vertex, t_vertex;
            s_vertex = source(*eit, road_graph_);
            t_vertex = target(*eit, road_graph_);
            RoadPt& s_pt = road_graph_[s_vertex].pt;
            RoadPt& t_pt = road_graph_[t_vertex].pt;
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(s_pt.x, s_pt.y, Z_DEBUG + 0.01f));
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(t_pt.x, t_pt.y, Z_DEBUG + 0.01f));
            line_colors_.emplace_back(c);
            line_colors_.emplace_back(Color(0.1f*c.r, 0.1f*c.g, 0.1f*c.b, 1.0f));
        }
    }
    }
}

// A-star visitor
// euclidean distance heuristic
template <class Graph, class CostType>
class distance_heuristic : public astar_heuristic<Graph, CostType>
{
public:
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    distance_heuristic(Vertex goal, Graph& g)
    : m_goal(goal), m_g(g) {}
    CostType operator()(Vertex u)
    {
        CostType dx = m_g[m_goal].pt.x - m_g[u].pt.x;
        CostType dy = m_g[m_goal].pt.y - m_g[u].pt.y;
        return sqrt(dx * dx + dy * dy);
    }
private:
    Vertex m_goal;
    Graph& m_g;
};

struct found_goal {};
template <class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor{
    public:
        astar_goal_visitor(Vertex goal) : m_goal(goal){} 
        template <class Graph>
        void examine_vertex(Vertex u, Graph& g){
            if(u == m_goal) 
                throw found_goal();
        }
    private:
        Vertex m_goal;
};

float RoadGenerator::shortestPath(int source, int target, vector<int>& path){
    path.clear();
    
    vector<int> p(num_vertices(road_graph_));
    vector<float> d(num_vertices(road_graph_));

    try{
        astar_search_tree(road_graph_, 
                          source, 
                          distance_heuristic<road_graph_t, float>(target, road_graph_), 
                          predecessor_map(make_iterator_property_map(p.begin(), get(boost::vertex_index, road_graph_))).
                          distance_map(make_iterator_property_map(d.begin(), get(boost::vertex_index, road_graph_))).
                          weight_map(get(&RoadGraphEdge::length, road_graph_)).
                         visitor(astar_goal_visitor<road_graph_vertex_descriptor>(target)));

    }catch (found_goal fg){
        return d[target];
    }

    return POSITIVE_INFINITY;
} 

void RoadGenerator::mapMatching(size_t traj_idx, vector<int>& projection){
    projection.clear();

    if(traj_idx > trajectories_->getNumTraj() - 1){
        cout << "Warning from RoadGenerator::partialMapMatching: traj_idx greater than the actual number of trajectories." << endl;
        return;
    } 

    float SEARCH_RADIUS = Parameters::getInstance().searchRadius();  
    float CUT_OFF_PROBABILITY = 0.01f; // this is only for projection
    float SIGMA_W = Parameters::getInstance().roadSigmaW();
    float SIGMA_L = Parameters::getInstance().roadSigmaH();
    float SIGMA_H = 7.5f;

    // Project each GPS points to nearby road_points_
    const vector<int>& traj = trajectories_->trajectories()[traj_idx];

    if(traj.size() < 2){
        cout << "\tTrajectory size less than 2." << endl;
        return;
    } 

    vector<vector<pair<int, float>>> candidate_projections(traj.size(), vector<pair<int, float>>()); // -1 means NR projection

    for (size_t i = 0; i < traj.size(); ++i) { 
        PclPoint& pt = trajectories_->data()->at(traj[i]);  
        vector<pair<int, float>>& candidate_projection = candidate_projections[i];

        // Search nearby road_points_
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        road_point_search_tree_->radiusSearch(pt,
                                              SEARCH_RADIUS,
                                              k_indices,
                                              k_dist_sqrs); 

        float cur_max_projection = 0.0f;

        // id_road, pair<id_road_pt, probability>
        map<int, pair<int, float>> road_max_prob;
        
        for (size_t j = 0; j < k_indices.size() ; ++j) { 
            PclPoint& r_pt = road_points_->at(k_indices[j]); 
            float delta_h = abs(deltaHeading1MinusHeading2(pt.head, r_pt.head));
            if(delta_h > 3.0f * SIGMA_H) 
                continue;

            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            Eigen::Vector2d vec(pt.x - r_pt.x,
                                pt.y - r_pt.y);
            float delta_l = vec.dot(r_pt_dir);
            float delta_w = sqrt(abs(vec.dot(vec) - delta_l*delta_l));

            delta_w -= 0.25f * r_pt.t * LANE_WIDTH;
            if(delta_w < 0.0f) 
                delta_w = 0.0f;

            float prob = exp(-1.0f * delta_w * delta_w / 2.0f / SIGMA_W / SIGMA_W) 
                       * exp(-1.0f * delta_l * delta_l / 2.0f / SIGMA_L / SIGMA_L) 
                       * exp(-1.0f * delta_h * delta_h / 2.0f / SIGMA_H / SIGMA_H);
            
            if(prob < CUT_OFF_PROBABILITY) 
                continue;

            if(prob > cur_max_projection){
                cur_max_projection = prob;
            }

            if(road_max_prob.find(r_pt.id_trajectory) != road_max_prob.end()){
                if(prob > road_max_prob[r_pt.id_trajectory].second){
                    road_max_prob[r_pt.id_trajectory].first = k_indices[j];
                    road_max_prob[r_pt.id_trajectory].second = prob;
                } 
            }
            else{
                road_max_prob[r_pt.id_trajectory] = pair<int, float>(k_indices[j], prob);
            }
        }

        for(map<int, pair<int, float>>::iterator it = road_max_prob.begin(); it != road_max_prob.end(); ++it){
            candidate_projection.emplace_back(pair<int, float>(it->second.first, it->second.second)); 
        } 
    } 

    //cout << "projection results:" << endl;
    //for (size_t i = 0; i < candidate_projections.size() ; ++i) { 
    //    cout << "\t " << i << ": ";
    //    vector<pair<int, float>>& candidate_projection = candidate_projections[i];
    //    for (size_t j = 0; j < candidate_projection.size() ; ++j) { 
    //        if(candidate_projection[j].first != -1) 
    //            cout << "(" << road_points_->at(candidate_projection[j].first).id_trajectory << "," << candidate_projection[j].second <<"), ";
    //        else{
    //            cout << "(-1,"<< candidate_projection[j].second << "), ";
    //        }
    //    } 
    //    cout << endl;
    //} 
    //cout << endl;

    //HMM map matching
    float MIN_TRANSISTION_SCORE = 0.01f; // Everything is connected to everything with this minimum probability
    float MIN_LOG_TRANSISTION_SCORE = log10(MIN_TRANSISTION_SCORE);

    vector<vector<int>>   pre(traj.size(), vector<int>());
    vector<vector<float>> scores(traj.size(), vector<float>());

    // Initialize scores
    for (size_t i = 0; i < traj.size(); ++i) { 
        vector<float>& score = scores[i];
        vector<int>&   prei  = pre[i];

        vector<pair<int, float>>& candidate_projection = candidate_projections[i];

        score.resize(candidate_projection.size(), MIN_LOG_TRANSISTION_SCORE); 
        prei.resize(candidate_projection.size(), -1);
        if(i == 0){
            for (size_t j = 0; j < candidate_projection.size(); ++j) { 
                score[j] = log10(candidate_projection[j].second); 
            }  
        } 
    } 

    for (size_t i = 1; i < traj.size() ; ++i) { 
        vector<pair<int, float>>& R = candidate_projections[i-1];
        vector<pair<int, float>>& L = candidate_projections[i];
        vector<int>&           prei = pre[i];
    
        vector<float>& scorer = scores[i-1];
        vector<float>& scorel = scores[i];

        for (size_t j = 0; j < L.size(); ++j) { 
            int cur_max_idx = -1;
            float cur_max   = -1e9;
            for (size_t k = 0; k < R.size(); ++k) { 
                // Compute prob(L[j].first -> R[k].first) based on road_graph_
                float p_r_l = MIN_TRANSISTION_SCORE; 

                if(L[j].first != -1 && R[k].first != -1){
                    // Can compute shortest path
                    if(road_points_->at(L[j].first).id_trajectory == road_points_->at(R[k].first).id_trajectory) 
                        p_r_l = 1.0f;
                    else{
                        vector<int> path;
                        road_graph_vertex_descriptor source_idx, target_idx;
                        PclPoint& R_pt = road_points_->at(R[k].first);
                        PclPoint& L_pt = road_points_->at(L[j].first);
                        source_idx = indexed_roads_[R_pt.id_trajectory][R_pt.id_sample];
                        target_idx = indexed_roads_[L_pt.id_trajectory][L_pt.id_sample];
                        float dist = shortestPath(source_idx, target_idx, path);
                        
                        float delta_x = L_pt.x - R_pt.x;
                        float delta_y = L_pt.y - R_pt.y;
                        float D = sqrt(delta_x*delta_x + delta_y*delta_y);

                        if(D > 1e-3){
                            float tmp = abs(dist - D) / D;
                            tmp = 1.0f - tmp;

                            if(tmp > MIN_TRANSISTION_SCORE) 
                                p_r_l = tmp;
                        }
                    }
                }
                // make -1 -> -1 high probability
                //else if(L[j].first == -1 && R[k].first == -1) 
                //    p_r_l = 0.5f;

                float s_r_p_r_l = scorer[k] + log10(p_r_l);
                if(cur_max < s_r_p_r_l){
                    cur_max = s_r_p_r_l;
                    cur_max_idx = k;
                } 
            } 
            
            scorel[j] = log10(L[j].second) + cur_max;   
            prei[j]   = cur_max_idx;
        } 
    } 

    // Trace projection results
    // find max idx
    float last_max_score = -1e9;
    float last_max_idx = -1;
    vector<float>& last_score = scores.back(); 
    for (size_t i = 0; i < last_score.size(); ++i) { 
        if(last_max_score < last_score[i]) {
            last_max_score = last_score[i];
            last_max_idx = i;
        }
    } 

    if(last_max_idx != -1){
        projection.resize(traj.size(), -1); 

        int last_idx = last_max_idx;
        projection[traj.size()-1] = candidate_projections[traj.size()-1][last_max_idx].first;

        for(int i = traj.size()-1; i >= 1; --i){
            projection[i-1] = candidate_projections[i-1][pre[i][last_idx]].first;
            last_idx = pre[i][last_idx];
        }
    } 
} 

void RoadGenerator::partialMapMatching(size_t traj_idx, vector<int>& projection){
    projection.clear();

    if(traj_idx > trajectories_->getNumTraj() - 1){
        cout << "Warning from RoadGenerator::partialMapMatching: traj_idx greater than the actual number of trajectories." << endl;
        return;
    } 

    float SEARCH_RADIUS = Parameters::getInstance().searchRadius();  
    float CUT_OFF_PROBABILITY = 0.3f; // this is only for projection
    float SIGMA_W = Parameters::getInstance().roadSigmaW();
    float SIGMA_L = Parameters::getInstance().roadSigmaH();
    float SIGMA_H = 7.5f;

    // Project each GPS points to nearby road_points_
    const vector<int>& traj = trajectories_->trajectories()[traj_idx];

    if(traj.size() < 2){
        return;
    } 

    vector<vector<pair<int, float>>> candidate_projections(traj.size(), vector<pair<int, float>>()); // -1 means NR projection

    for (size_t i = 0; i < traj.size(); ++i) { 
        PclPoint& pt = trajectories_->data()->at(traj[i]);  
        vector<pair<int, float>>& candidate_projection = candidate_projections[i];

        // Search nearby road_points_
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        road_point_search_tree_->radiusSearch(pt,
                                              SEARCH_RADIUS,
                                              k_indices,
                                              k_dist_sqrs); 

        float cur_max_projection = 0.0f;

        // id_road, pair<id_road_pt, probability>
        map<int, pair<int, float>> road_max_prob;
        
        for (size_t j = 0; j < k_indices.size() ; ++j) { 
            PclPoint& r_pt = road_points_->at(k_indices[j]); 
            float delta_h = abs(deltaHeading1MinusHeading2(pt.head, r_pt.head));
            if(delta_h > 3.0f * SIGMA_H) 
                continue;

            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            Eigen::Vector2d vec(pt.x - r_pt.x,
                                pt.y - r_pt.y);
            float delta_l = vec.dot(r_pt_dir);
            float delta_w = sqrt(abs(vec.dot(vec) - delta_l*delta_l));

            delta_w -= 0.25f * r_pt.t * LANE_WIDTH;
            if(delta_w < 0.0f) 
                delta_w = 0.0f;

            float prob = exp(-1.0f * delta_w * delta_w / 2.0f / SIGMA_W / SIGMA_W) 
                       * exp(-1.0f * delta_l * delta_l / 2.0f / SIGMA_L / SIGMA_L) 
                       * exp(-1.0f * delta_h * delta_h / 2.0f / SIGMA_H / SIGMA_H);
            
            if(prob < CUT_OFF_PROBABILITY) 
                continue;

            if(prob > cur_max_projection){
                cur_max_projection = prob;
            }

            if(road_max_prob.find(r_pt.id_trajectory) != road_max_prob.end()){
                if(prob > road_max_prob[r_pt.id_trajectory].second){
                    road_max_prob[r_pt.id_trajectory].first = k_indices[j];
                    road_max_prob[r_pt.id_trajectory].second = prob;
                } 
            }
            else{
                road_max_prob[r_pt.id_trajectory] = pair<int, float>(k_indices[j], prob);
            }
        }

        for(map<int, pair<int, float>>::iterator it = road_max_prob.begin(); it != road_max_prob.end(); ++it){
            candidate_projection.emplace_back(pair<int, float>(it->second.first, it->second.second)); 
        } 

        float NR_prob = 1.0f - cur_max_projection;
        candidate_projection.emplace_back(pair<int, float>(-1, NR_prob)); 
    } 

    //cout << "projection results:" << endl;
    //for (size_t i = 0; i < candidate_projections.size() ; ++i) { 
    //    cout << "\t " << i << ": ";
    //    vector<pair<int, float>>& candidate_projection = candidate_projections[i];
    //    for (size_t j = 0; j < candidate_projection.size() ; ++j) { 
    //        if(candidate_projection[j].first != -1) 
    //            cout << "(" << road_points_->at(candidate_projection[j].first).id_trajectory << "," << candidate_projection[j].second <<"), ";
    //        else{
    //            cout << "(-1,"<< candidate_projection[j].second << "), ";
    //        }
    //    } 
    //    cout << endl;
    //} 
    //cout << endl;

    //HMM map matching
    float MIN_TRANSISTION_SCORE = 0.1f; // Everything is connected to everything with this minimum probability
    float MIN_LOG_TRANSISTION_SCORE = log10(MIN_TRANSISTION_SCORE);

    vector<vector<int>>   pre(traj.size(), vector<int>());
    vector<vector<float>> scores(traj.size(), vector<float>());

    // Initialie scores
    for (size_t i = 0; i < traj.size(); ++i) { 
        vector<float>& score = scores[i];
        vector<int>&   prei  = pre[i];

        vector<pair<int, float>>& candidate_projection = candidate_projections[i];

        score.resize(candidate_projection.size(), MIN_LOG_TRANSISTION_SCORE); 
        prei.resize(candidate_projection.size(), -1);
        if(i == 0){
            for (size_t j = 0; j < candidate_projection.size(); ++j) { 
                score[j] = log10(candidate_projection[j].second); 
            }  
        } 
    } 

    for (size_t i = 1; i < traj.size() ; ++i) { 
        vector<pair<int, float>>& R = candidate_projections[i-1];
        vector<pair<int, float>>& L = candidate_projections[i];
        vector<int>&           prei = pre[i];
    
        vector<float>& scorer = scores[i-1];
        vector<float>& scorel = scores[i];

        for (size_t j = 0; j < L.size(); ++j) { 
            int cur_max_idx = -1;
            float cur_max   = -1e9;
            for (size_t k = 0; k < R.size(); ++k) { 
                // Compute prob(L[j].first -> R[k].first) based on road_graph_
                float p_r_l = MIN_TRANSISTION_SCORE; 

                if(L[j].first != -1 && R[k].first != -1){
                    // Can compute shortest path
                    if(road_points_->at(L[j].first).id_trajectory == road_points_->at(R[k].first).id_trajectory) 
                        p_r_l = 1.0f;
                    else{
                        vector<int> path;
                        road_graph_vertex_descriptor source_idx, target_idx;
                        PclPoint& R_pt = road_points_->at(R[k].first);
                        PclPoint& L_pt = road_points_->at(L[j].first);
                        source_idx = indexed_roads_[R_pt.id_trajectory][R_pt.id_sample];
                        target_idx = indexed_roads_[L_pt.id_trajectory][L_pt.id_sample];
                        float dist = shortestPath(source_idx, target_idx, path);
                        
                        float delta_x = L_pt.x - R_pt.x;
                        float delta_y = L_pt.y - R_pt.y;
                        float D = sqrt(delta_x*delta_x + delta_y*delta_y);

                        if(D > 1e-3){
                            float tmp = abs(dist - D) / D;
                            tmp = 1.0f - tmp;

                            if(tmp > MIN_TRANSISTION_SCORE) 
                                p_r_l = tmp;
                        }
                    }
                }
                // make -1 -> -1 high probability
                //else if(L[j].first == -1 && R[k].first == -1) 
                //    p_r_l = 0.5f;

                float s_r_p_r_l = scorer[k] + log10(p_r_l);
                if(cur_max < s_r_p_r_l){
                    cur_max = s_r_p_r_l;
                    cur_max_idx = k;
                } 
            } 
            
            scorel[j] = log10(L[j].second) + cur_max;   
            prei[j]   = cur_max_idx;
        } 
    } 

    // Trace projection results
    // find max idx
    float last_max_score = -1e9;
    float last_max_idx = -1;
    vector<float>& last_score = scores.back(); 
    for (size_t i = 0; i < last_score.size(); ++i) { 
        if(last_max_score < last_score[i]) {
            last_max_score = last_score[i];
            last_max_idx = i;
        }
    } 

    if(last_max_idx != -1){
        projection.resize(traj.size(), -1); 

        int last_idx = last_max_idx;
        projection[traj.size()-1] = candidate_projections[traj.size()-1][last_max_idx].first;

        for(int i = traj.size()-1; i >= 1; --i){
            projection[i-1] = candidate_projections[i-1][pre[i][last_idx]].first;
            last_idx = pre[i][last_idx];
        }
    } 
} 

void RoadGenerator::computeGPSPointsOnRoad(const vector<RoadPt>& road,
                                           set<int>& results){
    if(road.size()  == 0 || trajectories_ == NULL){
        return;
    }
    
    if (road.size() < 2) {
        cout << "WARNING from computePointsOnRoad: center_line size less than 2." << endl;
        return;
    }
    
    float heading_threshold = 10.0f; // in degrees
    float gps_sigma = Parameters::getInstance().gpsSigma();
    
    float search_radius = Parameters::getInstance().searchRadius();
    
    for (size_t i = 0; i < road.size(); ++i) {
        PclPoint pt;
        pt.setCoordinate(road[i].x, road[i].y, 0.0f);
        
        Eigen::Vector3d dir = headingTo3dVector(road[i].head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        trajectories_->tree()->radiusSearch(pt,
                                            search_radius,
                                            k_indices,
                                            k_dist_sqrs);
        
        for (size_t s = 0; s < k_indices.size(); ++s) {
            int nb_pt_idx = k_indices[s];
            if(results.find(nb_pt_idx) != results.end())
                continue;
            
            PclPoint& nb_pt = trajectories_->data()->at(nb_pt_idx);
            
            Eigen::Vector3d vec(nb_pt.x - road[i].x,
                                nb_pt.y - road[i].y,
                                0.0f);
            float parallel_dist = abs(dir.cross(vec)[2]);
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, road[i].head));
            
            if (delta_heading > heading_threshold) {
                continue;
            }
            
            parallel_dist -= 0.5f * road[i].n_lanes * LANE_WIDTH;
            if(parallel_dist < 0){
                parallel_dist = 0.0f;
            }
            
            float probability = exp(-1.0f * parallel_dist * parallel_dist / 2.0f / gps_sigma / gps_sigma);
            if(probability > 0.5f) 
                results.emplace(k_indices[s]); 
        }
    }
} 

void RoadGenerator::production(){
}

void RoadGenerator::resolveQuery(){
}

void RoadGenerator::localAdjustment(){
}

void RoadGenerator::updatePointCloud(){
}

void RoadGenerator::refit(){
}

void RoadGenerator::draw(){
    // Draw features
    if (feature_vertices_.size() != 0) {
        QOpenGLBuffer position_buffer;
        position_buffer.create();
        position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        position_buffer.bind();
        position_buffer.allocate(&feature_vertices_[0], 3*feature_vertices_.size()*sizeof(float));
        shader_program_->setupPositionAttributes();
        
        QOpenGLBuffer color_buffer;
        color_buffer.create();
        color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        color_buffer.bind();
        color_buffer.allocate(&feature_colors_[0], 4*feature_colors_.size()*sizeof(float));
        shader_program_->setupColorAttributes();
        glPointSize(20);
        //    glLineWidth(5);
        //    glDrawArrays(GL_LINES, 0, feature_vertices_.size());
        glDrawArrays(GL_POINTS, 0, feature_vertices_.size());
    }
    
    // DEBUGGING
    if(lines_to_draw_.size() != 0){
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&lines_to_draw_[0], 3*lines_to_draw_.size()*sizeof(float));
        shader_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&line_colors_[0], 4*line_colors_.size()*sizeof(float));
        shader_program_->setupColorAttributes();
        glLineWidth(8.0);
        glDrawArrays(GL_LINES, 0, lines_to_draw_.size());
    }
    
    if (points_to_draw_.size() != 0) {
        QOpenGLBuffer position_buffer;
        position_buffer.create();
        position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        position_buffer.bind();
        position_buffer.allocate(&points_to_draw_[0], 3*points_to_draw_.size()*sizeof(float));
        shader_program_->setupPositionAttributes();
        
        QOpenGLBuffer color_buffer;
        color_buffer.create();
        color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        color_buffer.bind();
        color_buffer.allocate(&point_colors_[0], 4*point_colors_.size()*sizeof(float));
        shader_program_->setupColorAttributes();
        glPointSize(10.0f);
        glDrawArrays(GL_POINTS, 0, points_to_draw_.size());
    }
}

void RoadGenerator::clear(){
    trajectories_.reset();
    
    point_cloud_->clear();
    simplified_traj_points_->clear();
    grid_points_->clear();
    grid_votes_.clear();

    road_pieces_.clear(); 
    indexed_roads_.clear();
    max_road_label_ = 0;
    max_junc_label_ = 0;
    cur_num_clusters_ = 0;

    road_graph_.clear(); 
    road_points_->clear(); 
    
    feature_vertices_.clear(); 
    feature_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();

    tmp_ = 0;
}

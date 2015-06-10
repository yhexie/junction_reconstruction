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
    debug_mode_ = false;

    show_generated_map_ = true;
    generated_map_render_mode_ = GeneratedMapRenderingMode::realistic;
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

    cout << additional_edges.size() << " additional edges." << endl; 

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
    //for (size_t i = 0; i < road_points_->size(); ++i) { 
    //    PclPoint& r_pt = road_points_->at(i); 
    //    points_to_draw_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
    //    point_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(r_pt.id_trajectory));
    //}

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
        recomputeRoads();
    }

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
        if(!road_graph_[*vit].is_valid){
            continue;
        }
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

    // Smooth road
    for (size_t i = 0; i < indexed_roads_.size(); ++i) { 
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        vector<RoadPt> centerline;
        for(size_t j = 0; j < a_road.size(); ++j){
            RoadPt r_pt = road_graph_[a_road[j]].pt;
            centerline.emplace_back(r_pt);
        }
        smoothCurve(centerline, false);
        if(centerline.size() > a_road.size()){
            int n_pt_to_remove = centerline.size() - a_road.size();
            int n_removed = 0;
            while(n_removed < n_pt_to_remove){
                for(int s = 1; s < centerline.size() - 1; ++s) {
                    int dh1 = abs(deltaHeading1MinusHeading2(centerline[s].head, centerline[s-1].head));
                    int dh2 = abs(deltaHeading1MinusHeading2(centerline[s].head, centerline[s+1].head));
                    if(dh1 < 7.5f && dh2 < 7.5f){
                        centerline[s-1].x = 0.5f * (centerline[s-1].x + centerline[s].x);
                        centerline[s-1].y = 0.5f * (centerline[s-1].y + centerline[s].y);
                        centerline[s+1].x = 0.5f * (centerline[s+1].x + centerline[s].x);
                        centerline[s+1].y = 0.5f * (centerline[s+1].y + centerline[s].y);
                        centerline.erase(centerline.begin() + s);
                        break;
                    }
                }
                n_removed++;
            }
        }
        else if(a_road.size() > centerline.size()){
            int n_pt_to_add = a_road.size() - centerline.size();
            int n_added = 0;
            while(n_added < n_pt_to_add){
                for(int s = 1; s < centerline.size() - 1; ++s) {
                    int dh1 = abs(deltaHeading1MinusHeading2(centerline[s].head, centerline[s+1].head));
                    if(dh1 < 7.5f){
                        RoadPt new_pt = centerline[s];
                        new_pt.x = 0.5f * (centerline[s].x + centerline[s+1].x);
                        new_pt.y = 0.5f * (centerline[s].y + centerline[s+1].y);
                        centerline.insert(centerline.begin()+s, new_pt);
                        break;
                    }
                }
                n_added++;
            }
        }

        if(a_road.size() != centerline.size()){
            cout << "WARNING: size not equal!" << endl;
        }
        else{
            for(size_t j = 0; j < a_road.size(); ++j){
                road_graph_[a_road[j]].pt = centerline[j];
            }
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

    cout << "There are " << indexed_roads_.size() << " roads." << endl;
    // Visualization
    prepareGeneratedMap();
}

void RoadGenerator::prepareGeneratedMap(){
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
 
    Color skeleton_color = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE);
    Color junc_color = ColorMap::getInstance().getNamedColor(ColorMap::YELLOW);
    Color link_color = ColorMap::getInstance().getNamedColor(ColorMap::RED);
    cout << "indexed_road_size: " << indexed_roads_.size() << endl;
    for(size_t i = 0; i < indexed_roads_.size(); ++i){
        // Smooth road width
        vector<Vertex> road_line_loop;
        vector<Color>  road_line_loop_color;
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        Color road_color = ColorMap::getInstance().getDiscreteColor(i);
        // forward
        for (size_t j = 0; j < a_road.size(); ++j) { 
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            Eigen::Vector2d r_pt_perp_dir(-1.0f * r_pt_dir.y(), r_pt_dir.x());
            float half_width = 0.5f * r_pt.n_lanes * LANE_WIDTH;
            Eigen::Vector2d pt_loc(r_pt.x, r_pt.y);
            pt_loc += r_pt_perp_dir * half_width;
            road_line_loop.emplace_back(SceneConst::getInstance().normalize(pt_loc.x(), pt_loc.y(), Z_ROAD));
            float color_ratio = 1.0f - 0.5f * static_cast<float>(j) / a_road.size();
            road_line_loop_color.emplace_back(Color(color_ratio*road_color.r, color_ratio*road_color.g, color_ratio*road_color.b, 1.0f));
            if(j > 0){
                RoadPt& prev_r_pt = road_graph_[a_road[j-1]].pt;
                generated_map_lines_.emplace_back(SceneConst::getInstance().normalize(prev_r_pt.x, prev_r_pt.y, Z_ROAD));
                generated_map_lines_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_ROAD));
                generated_map_line_colors_.emplace_back(skeleton_color);
                generated_map_line_colors_.emplace_back(Color(0.5f* skeleton_color.r, 0.5f * skeleton_color.g, 0.5f * skeleton_color.b, 1.0f));
            }
        } 
        // backward
        for (int j = a_road.size()-1; j >= 0; --j) { 
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            Eigen::Vector2d r_pt_perp_dir(r_pt_dir.y(), -1.0f * r_pt_dir.x());
            float half_width = 0.5f * r_pt.n_lanes * LANE_WIDTH;
            Eigen::Vector2d pt_loc(r_pt.x, r_pt.y);
            pt_loc += r_pt_perp_dir * half_width;
            road_line_loop.emplace_back(SceneConst::getInstance().normalize(pt_loc.x(), pt_loc.y(), Z_ROAD));
            float color_ratio = 1.0f - 0.5f * static_cast<float>(j) / a_road.size();
            road_line_loop_color.emplace_back(Color(color_ratio*road_color.r, color_ratio*road_color.g, color_ratio*road_color.b, 1.0f));
        }
        generated_map_line_loops_.emplace_back(road_line_loop);
        generated_map_line_loop_colors_.emplace_back(road_line_loop_color);
    }

    // Draw junctions
    auto vs = vertices(road_graph_); 
    for(auto vit = vs.first; vit != vs.second; ++vit){
        if(road_graph_[*vit].is_valid){
            int n_degree = degree(*vit, road_graph_);
            if(n_degree > 2){
                RoadPt& r_pt = road_graph_[*vit].pt;
                generated_map_points_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_ROAD + 0.01f));
                generated_map_point_colors_.emplace_back(junc_color);
            }
        }
    }

    // Draw linking edges
    auto es = edges(road_graph_);
    for(auto eit = es.first; eit != es.second; ++eit){
        if(road_graph_[*eit].type == RoadGraphEdgeType::linking){
            road_graph_vertex_descriptor source_v = source(*eit, road_graph_);
            road_graph_vertex_descriptor target_v = target(*eit, road_graph_);
            RoadPt& source_r_pt = road_graph_[source_v].pt; 
            RoadPt& target_r_pt = road_graph_[target_v].pt; 
            generated_map_lines_.emplace_back(SceneConst::getInstance().normalize(source_r_pt.x, source_r_pt.y, Z_ROAD + 0.01f));
            generated_map_lines_.emplace_back(SceneConst::getInstance().normalize(target_r_pt.x, target_r_pt.y, Z_ROAD + 0.01f));
            generated_map_line_colors_.emplace_back(link_color);
            generated_map_line_colors_.emplace_back(Color(0.5f* link_color.r, 0.5f * link_color.g, 0.5f * link_color.b, 1.0f));
        }
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
    //lines_to_draw_.clear();
    //line_colors_.clear();

    if(cur_num_clusters_ == 0){
        cout << "WARNING: no junction cluster. Cannot connect roads." << endl;
    }

    int cluster_id = tmp_;
    if(cluster_id >= cur_num_clusters_){
        cout << "All links clusters are processed" << endl;
        return;
    }

    // Get all vertices in the cluster
    float raw_junction_center_x = 0.0f;
    float raw_junction_center_y = 0.0f;
    auto vs = vertices(road_graph_);
    vector<road_graph_vertex_descriptor> cluster_vertices;
    for(auto vit = vs.first; vit != vs.second; ++vit){
        if(road_graph_[*vit].cluster_id == cluster_id){
            cluster_vertices.emplace_back(*vit);
            raw_junction_center_x += road_graph_[*vit].pt.x;
            raw_junction_center_y += road_graph_[*vit].pt.y;
        }
    }

    if(cluster_vertices.size() > 0){
        raw_junction_center_x /= cluster_vertices.size();
        raw_junction_center_y /= cluster_vertices.size();
    }
    else{
        return;
    }

    cout << "raw junction center: (" << raw_junction_center_x << ", " << raw_junction_center_y << endl;

    // Trace road segments of the related roads from road_graph_ 
    struct Link{
        road_graph_vertex_descriptor source_vertex;
        road_graph_vertex_descriptor target_vertex;
        int source_related_road_idx;
        int source_idx_in_related_roads;
        int target_related_road_idx;
        int target_idx_in_related_roads;
    };

    vector<vector<road_graph_vertex_descriptor>> related_roads;
    vector<int>                                  related_road_focus_idx; // stores the idx of one of the link points in related_road

    vector<Link> links;
    set<road_graph_vertex_descriptor> visited_vertices;
    map<road_graph_vertex_descriptor, pair<int, int>> vertex_to_road_map;
    for (size_t i = 0; i < cluster_vertices.size(); ++i) { 
        road_graph_vertex_descriptor start_v = cluster_vertices[i];
        if(!visited_vertices.emplace(start_v).second)
            continue;

        vector<road_graph_vertex_descriptor> a_road;

        int idx_in_road = 0;
        // backward
        road_graph_vertex_descriptor cur_v = start_v;
        a_road.emplace_back(start_v);
        int road_label = road_graph_[cur_v].road_label;
        while(true){
            auto es = in_edges(cur_v, road_graph_);
            
            bool has_prev_node = false;
            for (auto eit = es.first; eit != es.second; ++eit){
                road_graph_vertex_descriptor source_v = source(*eit, road_graph_);
                if(road_graph_[source_v].road_label == road_label){
                    a_road.emplace(a_road.begin(), source_v);
                    idx_in_road++;
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
        while(true){
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

        int cur_road_idx = related_roads.size();
        for (size_t j = 0; j < a_road.size(); ++j) { 
            visited_vertices.emplace(a_road[j]);
            vertex_to_road_map[a_road[j]] = pair<int, int>(cur_road_idx, j);
        } 
        related_roads.emplace_back(a_road);
        related_road_focus_idx.emplace_back(idx_in_road);
    }

    // Update Link structures
    for (size_t i = 0; i < cluster_vertices.size(); ++i) { 
        road_graph_vertex_descriptor cur_vertex = cluster_vertices[i];
        pair<int, int>& cur_vertex_map = vertex_to_road_map[cur_vertex];
    
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
            pair<int, int>& to_vertex_map = vertex_to_road_map[to_vertex];
            
            Link new_link;
            new_link.source_vertex = cur_vertex;
            new_link.target_vertex = to_vertex;
            new_link.source_related_road_idx = cur_vertex_map.first;
            new_link.source_idx_in_related_roads = cur_vertex_map.second;
            new_link.target_related_road_idx = to_vertex_map.first;
            new_link.target_idx_in_related_roads = to_vertex_map.second;

            links.emplace_back(new_link);
            remove_edge(aug_edge, road_graph_);
        }
    }

    cout << links.size() << " links" << endl;
    for(size_t i = 0; i < links.size(); ++i){
        cout << "\t" << links[i].source_vertex << " to " << links[i].target_vertex << endl;
    }

    // Create a point cloud search tree to determine relations among roads
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    vector<vector<int>> indexed_related_roads;
    for (size_t i = 0; i < related_roads.size(); ++i) { 
        vector<road_graph_vertex_descriptor>& a_road = related_roads[i];
        vector<int> indexed_road;
        for (size_t j = 0; j < a_road.size(); ++j) { 
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            PclPoint pt;
            pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
            pt.id_trajectory = i;
            pt.id_sample = j;
            pt.t = r_pt.n_lanes;
            pt.head = r_pt.head;
            indexed_road.emplace_back(points->size());
            points->push_back(pt);
        } 
        indexed_related_roads.emplace_back(indexed_road);
    } 
    if(points->size() > 0){
        point_search_tree->setInputCloud(points);
    }
    else{
        return;
    }

    /*
     *Determine road relationships. Relationships include: OPPOSITE, CROSSING
     */
    int MIN_N_PT_TO_DETERMINE_OPPOSITE_RELATION = 5;
    float MAX_PERP_DIST = 20.0f;
    float MIN_PERP_DIST = -10.0f;
    struct OppositeRelation{
        set<int> h1_road_idxs;
        set<int> h2_road_idxs;
    };

    float SEARCH_RADIUS = 25.0f;  // in meters
    int ANGLE_OPPO_THRESHOLD = 15;
    vector<OppositeRelation> oppo_relations;
    for(size_t i = 0; i < indexed_related_roads.size(); ++i){
        map<int, int> candidate_opposite_road_support;
        map<int, int> candidate_same_direction_road_support;
        vector<int>& a_indexed_road = indexed_related_roads[i];
        for (size_t j = 0; j < a_indexed_road.size(); ++j) { 
            PclPoint& search_pt = points->at(a_indexed_road[j]);
            Eigen::Vector3d search_pt_dir = headingTo3dVector(search_pt.head);
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            
            point_search_tree->radiusSearch(search_pt, SEARCH_RADIUS, k_indices, k_dist_sqrs);
            for(size_t k = 0; k < k_indices.size(); ++k){
                int nearby_pt_idx = k_indices[k];
                PclPoint& nb_pt = points->at(nearby_pt_idx);
                if(nb_pt.id_trajectory == search_pt.id_trajectory)
                    continue;

                int delta_h = abs(deltaHeading1MinusHeading2(nb_pt.head, search_pt.head));
                if(delta_h < 7.5f){
                    // Same direction
                    Eigen::Vector3d vec(nb_pt.x - search_pt.x,
                                        nb_pt.y - search_pt.y,
                                        0.0f);
                    float parallel_dist = abs(vec.dot(search_pt_dir));
                    float perp_dist = sqrt(abs(k_dist_sqrs[k] - parallel_dist*parallel_dist));
                    if(parallel_dist < 5.0f && perp_dist < 10.0f){
                        if(candidate_same_direction_road_support.find(nb_pt.id_trajectory) != 
                                candidate_same_direction_road_support.end()){
                            candidate_same_direction_road_support[nb_pt.id_trajectory]++;
                        }
                        else{
                            candidate_same_direction_road_support[nb_pt.id_trajectory] = 1;
                        }
                    }
                }
                else if(delta_h > 180 - ANGLE_OPPO_THRESHOLD){
                    // Opposite direction
                    Eigen::Vector3d vec(nb_pt.x - search_pt.x,
                                        nb_pt.y - search_pt.y,
                                        0.0f);
                    float perp_dist = -1.0f * search_pt_dir.cross(vec)[2];
                    float parallel_dist = sqrt(abs(k_dist_sqrs[k] - perp_dist*perp_dist));
                    if(perp_dist < MAX_PERP_DIST && perp_dist > MIN_PERP_DIST && parallel_dist < 5.0f){
                        if(candidate_opposite_road_support.find(nb_pt.id_trajectory) !=
                                candidate_opposite_road_support.end()){
                            candidate_opposite_road_support[nb_pt.id_trajectory]++;
                        }
                        else{
                            candidate_opposite_road_support[nb_pt.id_trajectory] = 1;
                        }
                    }
                }
                else{
                    continue;
                }
            }
        } 

        set<int> h1_road_idxs;
        h1_road_idxs.emplace(i);
        set<int> h2_road_idxs;

        for (const auto& p : candidate_same_direction_road_support) { 
            if(p.second > MIN_N_PT_TO_DETERMINE_OPPOSITE_RELATION){
                h1_road_idxs.emplace(p.first);
            }
        } 

        for (const auto& p : candidate_opposite_road_support) { 
            if(p.second > MIN_N_PT_TO_DETERMINE_OPPOSITE_RELATION){
                h2_road_idxs.emplace(p.first);
            }
        } 

        if(h2_road_idxs.size() == 0)
            continue;

        // Check existing opposite relations
        bool absorbed = false;
        for (auto& oppo_relation : oppo_relations) { 
            bool consistent_with_h1 = false;
            for (const auto& a : h1_road_idxs) { 
                if(oppo_relation.h1_road_idxs.find(a) != oppo_relation.h1_road_idxs.end()){
                    consistent_with_h1 = true;
                    break;
                }
            } 

            for (const auto& b : h2_road_idxs) { 
                if(oppo_relation.h2_road_idxs.find(b) != oppo_relation.h2_road_idxs.end()){
                    consistent_with_h1 = true;
                    break;
                }
            } 

            if(consistent_with_h1){
                absorbed = true;
                for (auto& a : h1_road_idxs) { 
                    oppo_relation.h1_road_idxs.emplace(a);
                } 

                for (auto& a : h2_road_idxs) { 
                    oppo_relation.h2_road_idxs.emplace(a);
                } 
            }
            
            bool consistent_with_h2 = false;
            for (const auto& a : h1_road_idxs) { 
                if(oppo_relation.h2_road_idxs.find(a) != oppo_relation.h2_road_idxs.end()){
                    consistent_with_h2 = true;
                    break;
                }
            } 
            for (const auto& b : h2_road_idxs) { 
                if(oppo_relation.h1_road_idxs.find(b) != oppo_relation.h1_road_idxs.end()){
                    consistent_with_h2 = true;
                    break;
                }
            } 

            if(consistent_with_h2){
                absorbed = true;
                for (auto& a : h1_road_idxs) { 
                    oppo_relation.h2_road_idxs.emplace(a);
                } 

                for (auto& a : h2_road_idxs) { 
                    oppo_relation.h1_road_idxs.emplace(a);
                } 
            }

            if(absorbed)
                break;
        } 

        if(!absorbed){
            OppositeRelation new_oppo_relation;
            new_oppo_relation.h1_road_idxs = std::move(h1_road_idxs);
            new_oppo_relation.h2_road_idxs = std::move(h2_road_idxs);
            oppo_relations.emplace_back(new_oppo_relation);
        }
    }

    // Compute Perpendicular Relation
    int N_HEADING_BINS = 72;
    float delta_heading_bin = 360 / N_HEADING_BINS;
    int max_extension = 20;
    int ANGLE_PERP_THRESHOLD = 30;
    set<pair<int, int>> perp_relations;
    for(size_t i = 0; i < links.size(); ++i){
        Link link = links[i];
        int source_road_idx = link.source_related_road_idx;
        int target_road_idx = link.target_related_road_idx;

        // Source road vote
        vector<road_graph_vertex_descriptor>& source_road = related_roads[link.source_related_road_idx];
        vector<float> source_heading_hist(N_HEADING_BINS, 0.0f);
        for (int j = link.source_idx_in_related_roads - max_extension; j <= link.source_idx_in_related_roads + max_extension; ++j) { 
            if(j < 0)
                continue;
            if(j >= source_road.size())
                break;

            RoadPt& r_pt = road_graph_[source_road[j]].pt;
            
            float shifted_head = r_pt.head + delta_heading_bin / 2.0;
            if(shifted_head > 360.0f)
                shifted_head -= 360.0f;

            int heading_bin_idx = floor(shifted_head / delta_heading_bin);
            for(int k = heading_bin_idx - 3; k <= heading_bin_idx + 3; ++k){
                int mk = (k + N_HEADING_BINS) % N_HEADING_BINS;
                float bin_center = mk * delta_heading_bin;
                float delta_h = deltaHeading1MinusHeading2(bin_center, r_pt.head);

                float vote = exp(-1.0f * delta_h * delta_h / 2.0f / 7.5f / 7.5f);
                if(vote < 0.1f)
                    vote = 0.0f;
                source_heading_hist[mk] += vote;
            }
        }

        vector<int> source_peak_idxs;
        peakDetector(source_heading_hist,
                     6,
                     1.5f,
                     source_peak_idxs,
                     true);
        int source_heading = -1;
        if(source_peak_idxs.size() > 0){
            int max_peak_idx = -1;
            float max_vote = 0.0f;
            for (const auto& p : source_peak_idxs) { 
                if(source_heading_hist[p] > max_vote){
                    max_vote = source_heading_hist[p];
                    max_peak_idx = p;
                }
            } 

            if(max_peak_idx != -1){
                source_heading = floor(max_peak_idx * delta_heading_bin);
            }
        }

        // Target road vote
        vector<road_graph_vertex_descriptor>& target_road = related_roads[link.target_related_road_idx];
        vector<float> target_heading_hist(N_HEADING_BINS, 0.0f);
        pair<int, int> existing_perp_relations;
        for (int j = link.target_idx_in_related_roads - max_extension; j <= link.target_idx_in_related_roads + max_extension; ++j) { 
            if(j < 0)
                continue;
            if(j >= target_road.size())
                break;

            RoadPt& r_pt = road_graph_[target_road[j]].pt;
            
            float shifted_head = r_pt.head + delta_heading_bin / 2.0;
            if(shifted_head > 360.0f)
                shifted_head -= 360.0f;

            int heading_bin_idx = floor(shifted_head / delta_heading_bin);
            for(int k = heading_bin_idx - 3; k <= heading_bin_idx + 3; ++k){
                int mk = (k + N_HEADING_BINS) % N_HEADING_BINS;
                float bin_center = mk * delta_heading_bin;
                float delta_h = deltaHeading1MinusHeading2(bin_center, r_pt.head);

                float vote = exp(-1.0f * delta_h * delta_h / 2.0f / 7.5f / 7.5f);
                if(vote < 0.1f)
                    vote = 0.0f;
                target_heading_hist[mk] += vote;
            }
        }

        vector<int> target_peak_idxs;
        peakDetector(target_heading_hist,
                     6,
                     1.5f,
                     target_peak_idxs,
                     true);
        int target_heading = -1;
        if(target_peak_idxs.size() > 0){
            int max_peak_idx = -1;
            float max_vote = 0.0f;
            for (const auto& p : target_peak_idxs) { 
                if(target_heading_hist[p] > max_vote){
                    max_vote = target_heading_hist[p];
                    max_peak_idx = p;
                }
            } 

            if(max_peak_idx != -1){
                target_heading = floor(max_peak_idx * delta_heading_bin);
            }
        }

        if(source_heading != -1 && target_heading != -1){
            float dh = abs(deltaHeading1MinusHeading2(source_heading, target_heading));
            if(dh < 90 + ANGLE_PERP_THRESHOLD && dh > 90 - ANGLE_PERP_THRESHOLD){
                bool absorbed = false;
                if(perp_relations.find(pair<int, int>(source_road_idx, target_road_idx)) ==
                    perp_relations.end()){
                    perp_relations.emplace(pair<int, int>(source_road_idx, target_road_idx));
                }
            }
        }
    }

    struct RoadGroup{
        set<int> road_idxs;
    };

    vector<RoadGroup> raw_road_groups;
    if(oppo_relations.size() > 0){
        for (const auto& oppo_relation : oppo_relations) {
            if(oppo_relation.h1_road_idxs.size() > 0){
                RoadGroup new_group;
                for (const auto& a : oppo_relation.h1_road_idxs) { 
                    new_group.road_idxs.emplace(a);
                } 
                raw_road_groups.emplace_back(new_group);
            }
            if(oppo_relation.h2_road_idxs.size() > 0){
                RoadGroup new_group;
                for (const auto& a : oppo_relation.h2_road_idxs) { 
                    new_group.road_idxs.emplace(a);
                } 
                raw_road_groups.emplace_back(new_group);
            }
        }
    }

    if(perp_relations.size() > 0){
        for(size_t i = 0; i < links.size(); ++i){
            Link link = links[i];
            int source_road_idx = link.source_related_road_idx;
            int target_road_idx = link.target_related_road_idx;
            // If they have link, and perp to the same set of roads, then they should be in one group
            set<int> source_road_perp_to;
            for (const auto& perp_relation : perp_relations) { 
                if(perp_relation.first == source_road_idx){
                    source_road_perp_to.emplace(perp_relation.second);
                }
                if(perp_relation.second == source_road_idx){
                    source_road_perp_to.emplace(perp_relation.first);
                }
            } 
            
            set<int> target_road_perp_to;
            for (const auto& perp_relation : perp_relations) { 
                if(perp_relation.first == target_road_idx){
                    target_road_perp_to.emplace(perp_relation.second);
                }
                if(perp_relation.second == target_road_idx){
                    target_road_perp_to.emplace(perp_relation.first);
                }
            }
            bool perp_to_same_road = false;
            for (const auto& a : source_road_perp_to) { 
                if(target_road_perp_to.find(a) != target_road_perp_to.end()){
                    perp_to_same_road = true;
                    break;
                }
            } 
            if(perp_to_same_road){
                // Check heading
                vector<road_graph_vertex_descriptor>& source_road = related_roads[source_road_idx];
                vector<road_graph_vertex_descriptor>& target_road = related_roads[target_road_idx];
                Eigen::Vector2d s_pt_dir;
                for(int k = link.source_idx_in_related_roads - 5; k <= link.source_idx_in_related_roads + 5; ++k){
                    if(k < 0)
                        continue;
                    if(k >= source_road.size())
                        break;
                    s_pt_dir += headingTo2dVector(road_graph_[source_road[k]].pt.head);
                }
                 
                Eigen::Vector2d t_pt_dir;
                for(int k = link.target_idx_in_related_roads - 5; k <= link.target_related_road_idx + 5; ++k){
                    if(k < 0)
                        continue;
                    if(k >= target_road.size())
                        break;
                    t_pt_dir += headingTo2dVector(road_graph_[target_road[k]].pt.head);
                }
                int s_pt_head = vector2dToHeading(s_pt_dir);
                int t_pt_head = vector2dToHeading(t_pt_dir);
                int dh = abs(deltaHeading1MinusHeading2(s_pt_head, t_pt_head));
                if(dh < 15.0f){
                    RoadPt& t_pt = road_graph_[link.target_vertex].pt;
                    RoadGroup new_group;
                    new_group.road_idxs.emplace(source_road_idx);
                    new_group.road_idxs.emplace(target_road_idx);
                    raw_road_groups.emplace_back(new_group);
                }
            }
        }
    }

    vector<RoadGroup> road_groups;
    {
        using namespace boost;
        typedef adjacency_list<vecS, vecS, undirectedS>    graph_t;
        typedef graph_traits<graph_t>::vertex_descriptor    vertex_descriptor;
        typedef graph_traits<graph_t>::edge_descriptor      edge_descriptor;
        graph_t G(related_roads.size());
        for (const auto& a_group : raw_road_groups) { 
            for (const auto& r1 : a_group.road_idxs) { 
                for (const auto& r2 : a_group.road_idxs) { 
                    if(r1 >= r2)
                        continue;
                    add_edge(r1, r2, G);
                } 
            } 
        } 
        vector<int> component(num_vertices(G));
        int         num = connected_components(G, &component[0]);
        road_groups.resize(num);
        for(int i = 0; i < num_vertices(G); ++i){
            road_groups[component[i]].road_idxs.emplace(i);
        }
    }

    // Readjust oppo_relations
    for (auto& oppo_relation : oppo_relations) { 
        int h1_group_idx = -1;
        for (const auto& a : oppo_relation.h1_road_idxs) { 
            for(int j = 0; j < road_groups.size(); ++j){
                RoadGroup& a_group = road_groups[j];
                if(a_group.road_idxs.find(a) != a_group.road_idxs.end()){
                    h1_group_idx = j;
                }
            } 
        } 
        for (auto& a : road_groups[h1_group_idx].road_idxs) { 
            oppo_relation.h1_road_idxs.emplace(a);
        } 

        int h2_group_idx = -1;
        for (const auto& a : oppo_relation.h2_road_idxs) { 
            for(int j = 0; j < road_groups.size(); ++j){
                RoadGroup& a_group = road_groups[j];
                if(a_group.road_idxs.find(a) != a_group.road_idxs.end()){
                    h2_group_idx = j;
                }
            } 
        } 
        for (auto& a : road_groups[h2_group_idx].road_idxs) { 
            oppo_relation.h2_road_idxs.emplace(a);
        }
    } 

    cout << "\tGroups:" << endl;
    for (const auto& a_group : road_groups) { 
        cout << "\t\t";
        for (const auto& idx : a_group.road_idxs) { 
            cout << idx << ", ";
        } 
        cout << endl;
    } 

    if(oppo_relations.size() > 0){
        cout << "\tOpposite relations:" << endl;
        for (const auto& oppo_relation : oppo_relations) { 
            cout << "\t\t{";
            for (const auto& p : oppo_relation.h1_road_idxs) { 
                cout << p << ", ";
            } 
            cout << "} ";

            cout << "is opposite to {";
            for (const auto& p : oppo_relation.h2_road_idxs) { 
                cout << p << ", ";
            } 
            cout << "} " << endl;
        }
    }
     
    if(perp_relations.size() > 0){
        cout << "\tPerpendicular relations:" << endl;
        for (const auto& perp_relation : perp_relations) { 
            cout << "\t\t" << perp_relation.first << " is perpendicular to " << perp_relation.second << endl; 
        }
    }

    // Visualize the traced roads
    if(debug_mode_){
        feature_vertices_.clear();
        feature_colors_.clear();
        for (size_t i = 0; i < related_roads.size(); ++i) { 
            vector<road_graph_vertex_descriptor>& a_road = related_roads[i];
            for (size_t j = 0; j < a_road.size(); ++j) { 
                RoadPt& r_pt = road_graph_[a_road[j]].pt;
                feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
                feature_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(i));
            } 
        } 
    }

    // 
    struct JuncSegment{
        vector<RoadPt> center_line;
        set<int>       road_idxs;
        int            junc_lower_idx = 1e6;
        int            junc_upper_idx = -1e6; // these are two rough window indicating junctions are inside this window 
    };

    struct OppositeWays{
        int junc_segment_idx1;
        int junc_segment_idx2;
    };

    struct CrossingWays{
        int junc_segment_idx1;
        int junc_segment_idx2;
    };

    vector<JuncSegment>  junc_segments;
    vector<OppositeWays> oppo_ways;
    vector<CrossingWays> cross_ways;

    float range = 50.0f;
    if(oppo_relations.size() > 0){
        for (const auto& oppo_relation : oppo_relations) { 
            vector<RoadPt> tmp_center_line;
            PclPointCloud::Ptr oppo_points(new PclPointCloud);
            PclSearchTree::Ptr oppo_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
            int h1_n_lanes = 0;
            // Get h1_points
            for (const auto& r_idx : oppo_relation.h1_road_idxs) { 
                vector<road_graph_vertex_descriptor>& a_road = related_roads[r_idx];
                for(int s = 0; s < a_road.size(); ++s){
                    RoadPt& r_pt = road_graph_[a_road[s]].pt;
                    float delta_x_to_center = r_pt.x - raw_junction_center_x;
                    float delta_y_to_center = r_pt.y - raw_junction_center_y;
                    float dist_to_center = sqrt(delta_x_to_center*delta_x_to_center + delta_y_to_center*delta_y_to_center);
                    if(dist_to_center < range){
                        PclPoint new_pt;
                        new_pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                        new_pt.head = r_pt.head;
                        new_pt.t = r_pt.n_lanes;
                        h1_n_lanes = r_pt.n_lanes;
                        new_pt.id_trajectory = 0; // 0 for h1 and 1 for h2
                        oppo_points->push_back(new_pt);
                    }
                }
            } 
            // Get h2_points
            int h2_n_lanes = 0;
            for (const auto& r_idx : oppo_relation.h2_road_idxs) { 
                vector<road_graph_vertex_descriptor>& a_road = related_roads[r_idx];
                for(int s = 0; s < a_road.size(); ++s){
                    RoadPt& r_pt = road_graph_[a_road[s]].pt;
                    float delta_x_to_center = r_pt.x - raw_junction_center_x;
                    float delta_y_to_center = r_pt.y - raw_junction_center_y;
                    float dist_to_center = sqrt(delta_x_to_center*delta_x_to_center + delta_y_to_center*delta_y_to_center);
                    if(dist_to_center < range){
                        PclPoint new_pt;
                        new_pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                        new_pt.head = r_pt.head;
                        new_pt.t = r_pt.n_lanes;
                        h2_n_lanes = r_pt.n_lanes;
                        new_pt.id_trajectory = 1; // 0 for h2 and 1 for h2
                        oppo_points->push_back(new_pt);
                    }
                }
            }
            if(oppo_points->size() > 0){
                oppo_point_search_tree->setInputCloud(oppo_points);
            }
            else{
                continue;
            }
            // Produce a joint center line for both roads
            vector<bool> marked_points(oppo_points->size(), false);
            float avg_oppo_dist = 0.0f;
            int n_avg_oppo_dist = 0;
            for (size_t k = 0; k < oppo_points->size(); ++k) { 
                if(marked_points[k])
                    continue;
                PclPoint& oppo_pt = oppo_points->at(k);
                Eigen::Vector3d oppo_pt_dir = headingTo3dVector(oppo_pt.head);
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                oppo_point_search_tree->radiusSearch(oppo_pt, 50.0f, k_indices, k_dist_sqrs);
                float avg_x = oppo_pt.x;
                float avg_y = oppo_pt.y;
                Eigen::Vector3d avg_dir;
                if(oppo_pt.id_trajectory == 0){
                    avg_dir = oppo_pt_dir; 
                }
                else{
                    avg_dir = -1.0f * oppo_pt_dir;
                }
                int   n_avg_pt = 1;
                bool valid = true;
                for(size_t l = 0; l < k_indices.size(); ++l){
                    int nb_pt_idx = k_indices[l];
                    if(nb_pt_idx == k)
                        continue;
                    if(marked_points[nb_pt_idx])
                        continue;
                    PclPoint& nb_pt = oppo_points->at(nb_pt_idx);
                    Eigen::Vector3d vec(nb_pt.x - oppo_pt.x,
                                        nb_pt.y - oppo_pt.y,
                                        0.0f);
                    float perp_dist = -1.0f * oppo_pt_dir.cross(vec)[2];
                    float parallel_dist = sqrt(abs(k_dist_sqrs[l] - perp_dist*perp_dist));
                    if(parallel_dist < 5.0f){
                        marked_points[nb_pt_idx] = true;
                        if(oppo_pt.id_trajectory == nb_pt.id_trajectory){
                            // Same direction, add to average
                            avg_x += nb_pt.x;
                            avg_y += nb_pt.y;
                            if(oppo_pt.id_trajectory == 0){
                                avg_dir += headingTo3dVector(nb_pt.head);
                            }
                            else{
                                avg_dir -= headingTo3dVector(nb_pt.head);
                            }
                            n_avg_pt++;
                        }
                        else{
                            // Opposite direction
                            if(perp_dist < MAX_PERP_DIST && perp_dist > MIN_PERP_DIST){
                                int dh = 180 - abs(deltaHeading1MinusHeading2(nb_pt.head, oppo_pt.head));
                                if(dh > 15)
                                    valid = false;
                                avg_x += nb_pt.x;
                                avg_y += nb_pt.y;
                                n_avg_pt++;
                                avg_oppo_dist += abs(perp_dist);
                                n_avg_oppo_dist++;
                                if(nb_pt.id_trajectory == 0){
                                    avg_dir += headingTo3dVector(nb_pt.head);
                                }
                                else{
                                    avg_dir -= headingTo3dVector(nb_pt.head);
                                }
                            }
                        }
                    }
                }
                if(valid){
                    avg_x /= n_avg_pt;
                    avg_y /= n_avg_pt;
                    RoadPt new_r_pt;
                    new_r_pt.x = avg_x;
                    new_r_pt.y = avg_y;
                    new_r_pt.head = vector3dToHeading(avg_dir);
                    tmp_center_line.emplace_back(new_r_pt);
                }
            } 
            if(tmp_center_line.size() == 0)
                continue;
            avg_oppo_dist /= n_avg_oppo_dist;
            // Sort tmp_center_line
            vector<pair<int, float>> proj_value;
            Eigen::Vector2d h_dir = headingTo2dVector(tmp_center_line[0].head);
            for (size_t k = 0; k < tmp_center_line.size(); ++k) { 
                Eigen::Vector2d vec(tmp_center_line[k].x - tmp_center_line[0].x,
                                    tmp_center_line[k].y - tmp_center_line[0].y);
                float dot_value = h_dir.dot(vec);
                proj_value.emplace_back(pair<int, float>(k, dot_value));
            } 
            sort(proj_value.begin(), proj_value.end(), pairCompare);
            vector<RoadPt> center_line;
            for(size_t k = 0; k < tmp_center_line.size(); ++k){
                center_line.emplace_back(tmp_center_line[proj_value[k].first]);
            }
            smoothCurve(center_line, false);
              
            JuncSegment new_h1_segment;
            JuncSegment new_h2_segment;
            float width_ratio = 0.5f;
            if(avg_oppo_dist < 10.0f)
                width_ratio *= 0.5f;
            for(int k = 0; k < center_line.size(); ++k){
                RoadPt& r_pt = center_line[k];
                Eigen::Vector2d h_dir = headingTo2dVector(r_pt.head);
                Eigen::Vector2d h_perp_dir(-h_dir.y(), h_dir.x());
                RoadPt new_pt;
                Eigen::Vector2d new_loc(r_pt.x, r_pt.y);
                new_loc -= width_ratio * h1_n_lanes * LANE_WIDTH * h_perp_dir;
                new_pt.x = new_loc.x();
                new_pt.y = new_loc.y();
                new_pt.head = r_pt.head;
                new_pt.n_lanes = h1_n_lanes;
                new_h1_segment.center_line.emplace_back(new_pt);
            }
            for (const auto& r_idx : oppo_relation.h1_road_idxs) { 
                new_h1_segment.road_idxs.emplace(r_idx);
            }
            for(int k = center_line.size() - 1; k >= 0; --k){
                RoadPt& r_pt = center_line[k];
                Eigen::Vector2d h_dir = headingTo2dVector(r_pt.head);
                Eigen::Vector2d h_perp_dir(-h_dir.y(), h_dir.x());
                RoadPt new_pt;
                Eigen::Vector2d new_loc(r_pt.x, r_pt.y);
                new_loc += width_ratio * h2_n_lanes * LANE_WIDTH * h_perp_dir;
                new_pt.x = new_loc.x();
                new_pt.y = new_loc.y();
                new_pt.head = (r_pt.head + 180) % 360;
                new_pt.n_lanes = h2_n_lanes;
                new_h2_segment.center_line.emplace_back(new_pt);
            }
            for (const auto& r_idx : oppo_relation.h2_road_idxs) { 
                new_h2_segment.road_idxs.emplace(r_idx);
            }
            junc_segments.emplace_back(new_h1_segment);
            junc_segments.emplace_back(new_h2_segment);
        }
    }

    if(perp_relations.size() > 0){
        for (const auto& perp_relation : perp_relations) { 
            int source_road_idx = perp_relation.first;
            bool source_is_covered = false;
            for (const auto& junc_seg : junc_segments) { 
                if(junc_seg.road_idxs.find(source_road_idx) != junc_seg.road_idxs.end()){
                    source_is_covered = true;
                    break;
                }
            } 
            if(!source_is_covered){
                int source_group_idx = -1;
                int source_n_lanes = 0;
                for(size_t l = 0; l < road_groups.size(); ++l){
                    if(road_groups[l].road_idxs.find(source_road_idx) != road_groups[l].road_idxs.end()){
                        source_group_idx = l;
                        break;
                    }
                }
                PclPointCloud::Ptr perp_points(new PclPointCloud);
                PclSearchTree::Ptr perp_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
                for (const auto& r_idx : road_groups[source_group_idx].road_idxs) { 
                    vector<road_graph_vertex_descriptor>& a_road = related_roads[r_idx];
                    for(int s = 0; s < a_road.size(); ++s){
                        RoadPt& r_pt = road_graph_[a_road[s]].pt;
                        float delta_x_to_center = r_pt.x - raw_junction_center_x;
                        float delta_y_to_center = r_pt.y - raw_junction_center_y;
                        float dist_to_center = sqrt(delta_x_to_center*delta_x_to_center + delta_y_to_center*delta_y_to_center);
                        if(dist_to_center < range){
                            PclPoint new_pt;
                            new_pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                            new_pt.head = r_pt.head;
                            new_pt.t = r_pt.n_lanes;
                            source_n_lanes = r_pt.n_lanes;
                            new_pt.id_trajectory = 0; // 0 for h1 and 1 for h2
                            perp_points->push_back(new_pt);
                        }
                    }
                }
                if(perp_points->size() > 0){
                    vector<RoadPt> tmp_center_line;
                    perp_point_search_tree->setInputCloud(perp_points);
                    vector<bool> marked_points(perp_points->size(), false);
                    for (size_t k = 0; k < perp_points->size(); ++k) { 
                        if(marked_points[k])
                            continue;
                        PclPoint& perp_pt = perp_points->at(k);
                        Eigen::Vector3d perp_pt_dir = headingTo3dVector(perp_pt.head);
                        vector<int> k_indices;
                        vector<float> k_dist_sqrs;
                        perp_point_search_tree->radiusSearch(perp_pt, 25.0f, k_indices, k_dist_sqrs);
                        float avg_x = perp_pt.x;
                        float avg_y = perp_pt.y;
                        Eigen::Vector3d avg_dir;
                        avg_dir = perp_pt_dir; 
                        int   n_avg_pt = 1;
                        for(size_t l = 0; l < k_indices.size(); ++l){
                            int nb_pt_idx = k_indices[l];
                            if(nb_pt_idx == k)
                                continue;
                            if(marked_points[nb_pt_idx])
                                continue;
                            PclPoint& nb_pt = perp_points->at(nb_pt_idx);
                            Eigen::Vector3d vec(nb_pt.x - perp_pt.x,
                                                nb_pt.y - perp_pt.y,
                                                0.0f);
                            float perp_dist = -1.0f * perp_pt_dir.cross(vec)[2];
                            float parallel_dist = sqrt(abs(k_dist_sqrs[l] - perp_dist*perp_dist));
                            if(parallel_dist < 5.0f){
                                marked_points[nb_pt_idx] = true;
                                avg_x += nb_pt.x;
                                avg_y += nb_pt.y;
                                avg_dir += headingTo3dVector(nb_pt.head);
                                n_avg_pt++;
                            }
                        }
                        avg_x /= n_avg_pt;
                        avg_y /= n_avg_pt;
                        RoadPt new_r_pt;
                        new_r_pt.x = avg_x;
                        new_r_pt.y = avg_y;
                        new_r_pt.head = vector3dToHeading(avg_dir);
                        tmp_center_line.emplace_back(new_r_pt);
                    } 
                    if(tmp_center_line.size() == 0)
                        continue;
                    // Sort tmp_center_line
                    vector<pair<int, float>> proj_value;
                    Eigen::Vector2d h_dir = headingTo2dVector(tmp_center_line[0].head);
                    for (size_t k = 0; k < tmp_center_line.size(); ++k) { 
                        Eigen::Vector2d vec(tmp_center_line[k].x - tmp_center_line[0].x,
                                            tmp_center_line[k].y - tmp_center_line[0].y);
                        float dot_value = h_dir.dot(vec);
                        proj_value.emplace_back(pair<int, float>(k, dot_value));
                    } 
                    sort(proj_value.begin(), proj_value.end(), pairCompare);
                    vector<RoadPt> center_line;
                    for(size_t k = 0; k < tmp_center_line.size(); ++k){
                        center_line.emplace_back(tmp_center_line[proj_value[k].first]);
                    }
                    smoothCurve(center_line, false);
                    JuncSegment new_segment;
                    for(int k = 0; k < center_line.size(); ++k){
                        RoadPt& r_pt = center_line[k];
                        RoadPt new_pt;
                        Eigen::Vector2d new_loc(r_pt.x, r_pt.y);
                        new_pt.x = new_loc.x();
                        new_pt.y = new_loc.y();
                        new_pt.head = r_pt.head;
                        new_pt.n_lanes = source_n_lanes;
                        new_segment.center_line.emplace_back(new_pt);
                    }
                    for (const auto& r_idx : road_groups[source_group_idx].road_idxs) { 
                        new_segment.road_idxs.emplace(r_idx);
                    }
                    junc_segments.emplace_back(new_segment);
                }
            }
            
            int target_road_idx = perp_relation.second;
            bool target_is_covered = false;
            for (const auto& junc_seg : junc_segments) { 
                if(junc_seg.road_idxs.find(target_road_idx) != junc_seg.road_idxs.end()){
                    target_is_covered = true;
                    break;
                }
            } 
            if(!target_is_covered){
                int target_group_idx = -1;
                int target_n_lanes = 0;
                for(size_t l = 0; l < road_groups.size(); ++l){
                    if(road_groups[l].road_idxs.find(target_road_idx) != road_groups[l].road_idxs.end()){
                        target_group_idx = l;
                        break;
                    }
                }
                PclPointCloud::Ptr perp_points(new PclPointCloud);
                PclSearchTree::Ptr perp_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
                for (const auto& r_idx : road_groups[target_group_idx].road_idxs) { 
                    vector<road_graph_vertex_descriptor>& a_road = related_roads[r_idx];
                    for(int s = 0; s < a_road.size(); ++s){
                        RoadPt& r_pt = road_graph_[a_road[s]].pt;
                        float delta_x_to_center = r_pt.x - raw_junction_center_x;
                        float delta_y_to_center = r_pt.y - raw_junction_center_y;
                        float dist_to_center = sqrt(delta_x_to_center*delta_x_to_center + delta_y_to_center*delta_y_to_center);
                        if(dist_to_center < range){
                            PclPoint new_pt;
                            new_pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                            new_pt.head = r_pt.head;
                            new_pt.t = r_pt.n_lanes;
                            target_n_lanes = r_pt.n_lanes;
                            new_pt.id_trajectory = 0; // 0 for h1 and 1 for h2
                            perp_points->push_back(new_pt);
                        }
                    }
                }
                if(perp_points->size() > 0){
                    vector<RoadPt> tmp_center_line;
                    perp_point_search_tree->setInputCloud(perp_points);
                    vector<bool> marked_points(perp_points->size(), false);
                    for (size_t k = 0; k < perp_points->size(); ++k) { 
                        if(marked_points[k])
                            continue;
                        PclPoint& perp_pt = perp_points->at(k);
                        Eigen::Vector3d perp_pt_dir = headingTo3dVector(perp_pt.head);
                        vector<int> k_indices;
                        vector<float> k_dist_sqrs;
                        perp_point_search_tree->radiusSearch(perp_pt, 25.0f, k_indices, k_dist_sqrs);
                        float avg_x = perp_pt.x;
                        float avg_y = perp_pt.y;
                        Eigen::Vector3d avg_dir;
                        avg_dir = perp_pt_dir; 
                        int   n_avg_pt = 1;
                        for(size_t l = 0; l < k_indices.size(); ++l){
                            int nb_pt_idx = k_indices[l];
                            if(nb_pt_idx == k)
                                continue;
                            if(marked_points[nb_pt_idx])
                                continue;
                            PclPoint& nb_pt = perp_points->at(nb_pt_idx);
                            Eigen::Vector3d vec(nb_pt.x - perp_pt.x,
                                                nb_pt.y - perp_pt.y,
                                                0.0f);
                            float perp_dist = -1.0f * perp_pt_dir.cross(vec)[2];
                            float parallel_dist = sqrt(abs(k_dist_sqrs[l] - perp_dist*perp_dist));
                            if(parallel_dist < 5.0f){
                                marked_points[nb_pt_idx] = true;
                                avg_x += nb_pt.x;
                                avg_y += nb_pt.y;
                                avg_dir += headingTo3dVector(nb_pt.head);
                                n_avg_pt++;
                            }
                        }
                        avg_x /= n_avg_pt;
                        avg_y /= n_avg_pt;
                        RoadPt new_r_pt;
                        new_r_pt.x = avg_x;
                        new_r_pt.y = avg_y;
                        new_r_pt.head = vector3dToHeading(avg_dir);
                        tmp_center_line.emplace_back(new_r_pt);
                    } 
                    if(tmp_center_line.size() == 0)
                        continue;
                    // Sort tmp_center_line
                    vector<pair<int, float>> proj_value;
                    Eigen::Vector2d h_dir = headingTo2dVector(tmp_center_line[0].head);
                    for (size_t k = 0; k < tmp_center_line.size(); ++k) { 
                        Eigen::Vector2d vec(tmp_center_line[k].x - tmp_center_line[0].x,
                                            tmp_center_line[k].y - tmp_center_line[0].y);
                        float dot_value = h_dir.dot(vec);
                        proj_value.emplace_back(pair<int, float>(k, dot_value));
                    } 
                    sort(proj_value.begin(), proj_value.end(), pairCompare);
                    vector<RoadPt> center_line;
                    for(size_t k = 0; k < tmp_center_line.size(); ++k){
                        center_line.emplace_back(tmp_center_line[proj_value[k].first]);
                    }
                    smoothCurve(center_line, false);
                    JuncSegment new_segment;
                    for(int k = 0; k < center_line.size(); ++k){
                        RoadPt& r_pt = center_line[k];
                        RoadPt new_pt;
                        Eigen::Vector2d new_loc(r_pt.x, r_pt.y);
                        new_pt.x = new_loc.x();
                        new_pt.y = new_loc.y();
                        new_pt.head = r_pt.head;
                        new_pt.n_lanes = target_n_lanes;
                        new_segment.center_line.emplace_back(new_pt);
                    }
                    for (const auto& r_idx : road_groups[target_group_idx].road_idxs) { 
                        new_segment.road_idxs.emplace(r_idx);
                    }
                    junc_segments.emplace_back(new_segment);
                }
            }
        } 
    }

    // Visualize
    for(int i = 0; i < junc_segments.size(); ++i){
        JuncSegment& a_seg = junc_segments[i];
        Color c = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
        for(int j = 1; j < a_seg.center_line.size(); ++j){
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(a_seg.center_line[j-1].x, a_seg.center_line[j-1].y, Z_DEBUG + 0.01f));
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(a_seg.center_line[j].x, a_seg.center_line[j].y, Z_DEBUG + 0.01f));
            line_colors_.emplace_back(c);
            line_colors_.emplace_back(Color(c.r * 0.3f, c.g * 0.3f, c.b * 0.3f, 1.0f));
        }
    }
    
    /*
     * Adjust graph
     *Start adjusting road_graph_!!! The most exciting moment finally comes!!!!
     */
    map<int, int> road_label_map;
    vector<vector<road_graph_vertex_descriptor>> junc_segment_vertices;
    for (size_t i = 0; i < junc_segments.size(); ++i) { 
        JuncSegment& a_seg = junc_segments[i];
        int road_label;
        if(a_seg.road_idxs.size() > 1){
            // Two roads need to be merged
            road_label = max_road_label_; 
            max_road_label_++;
        }
        else{
            int r_idx = *a_seg.road_idxs.begin();
            road_label = road_graph_[related_roads[r_idx][0]].road_label;
        }

        vector<road_graph_vertex_descriptor> new_vertices;
        for(int j = 0; j < a_seg.center_line.size(); ++j){
            road_graph_vertex_descriptor v = add_vertex(road_graph_);
            road_graph_[v].road_label = road_label;
            road_graph_[v].pt = a_seg.center_line[j]; 
            new_vertices.emplace_back(v);
            if(j > 0){
                auto e = boost::add_edge(new_vertices[j-1], new_vertices[j], road_graph_);  
                if(e.second){
                    float dx = a_seg.center_line[j].x - a_seg.center_line[j-1].x;
                    float dy = a_seg.center_line[j].y - a_seg.center_line[j-1].y;
                    road_graph_[e.first].length = sqrt(dx*dx + dy*dy);
                }
            }
        }
        junc_segment_vertices.emplace_back(new_vertices);

        for (const auto& r_idx : a_seg.road_idxs) { 
            vector<road_graph_vertex_descriptor>& a_road = related_roads[r_idx];
            vector<bool> marked(a_road.size(), false);
            for(size_t j = 0; j < a_seg.center_line.size(); ++j){
                RoadPt& seg_pt = a_seg.center_line[j];
                int min_dist_idx = -1;
                int min_dist = POSITIVE_INFINITY;
                for (size_t k = 0; k < a_road.size(); ++k) { 
                    RoadPt& r_pt = road_graph_[a_road[k]].pt;
                    float delta_x = r_pt.x - seg_pt.x;
                    float delta_y = r_pt.y - seg_pt.y;
                    float dd = sqrt(delta_x*delta_x + delta_y*delta_y);
                    if(dd < min_dist){
                        min_dist_idx = k;
                        min_dist = dd;
                    }
                }

                if(min_dist_idx != -1){
                    marked[min_dist_idx] = true;
                }
            }
            int start_idx = 0;
            while(true){
                if(marked[start_idx]){
                    break;
                }
                start_idx++;
                if(start_idx >= a_road.size())
                    break;
            }
            int end_idx = a_road.size() - 1;
            while(true){
                if(marked[end_idx]){
                    break;
                }
                end_idx--;
                if(end_idx < 0)
                    break;
            }

            // vertex between start_idx and end_idx will be removed
            if(start_idx - 1 >= 0){
                auto e = boost::add_edge(a_road[start_idx-1], new_vertices[0], road_graph_);  
                if(e.second){
                    float dx = road_graph_[a_road[start_idx-1]].pt.x - road_graph_[new_vertices[0]].pt.x;
                    float dy = road_graph_[a_road[start_idx-1]].pt.y - road_graph_[new_vertices[0]].pt.y;
                    road_graph_[e.first].length = sqrt(dx*dx + dy*dy);
                }
            }
            if(end_idx + 1 < a_road.size()){
                auto e = boost::add_edge(new_vertices.back(), a_road[end_idx+1], road_graph_);  
                if(e.second){
                    float dx = road_graph_[a_road[end_idx+1]].pt.x - road_graph_[new_vertices.back()].pt.x;
                    float dy = road_graph_[a_road[end_idx+1]].pt.y - road_graph_[new_vertices.back()].pt.y;
                    road_graph_[e.first].length = sqrt(dx*dx + dy*dy);
                }
            }
            for(int s = 0; s < a_road.size(); ++s){
                road_graph_[a_road[s]].road_label = road_label;
                if(s >= start_idx && s <= end_idx){
                    road_graph_[a_road[s]].road_label = -1;
                    road_graph_[a_road[s]].is_valid = false;
                    clear_vertex(a_road[s], road_graph_);
                }
            }
        } 
    } 

    // Link perpendicular roads
    map<pair<int, int>, road_graph_edge_descriptor> already_added_edges;
    for (const auto& perp_pair : perp_relations) { 
        int source_junc_seg_idx = -1;
        for(int i = 0; i < junc_segments.size(); ++i){
            JuncSegment& a_junc_seg = junc_segments[i];
            if(a_junc_seg.road_idxs.find(perp_pair.first) != a_junc_seg.road_idxs.end()){
                source_junc_seg_idx = i;
                break;
            }
        }

        int target_junc_seg_idx = -1;
        for(int i = 0; i < junc_segments.size(); ++i){
            JuncSegment& a_junc_seg = junc_segments[i];
            if(a_junc_seg.road_idxs.find(perp_pair.second) != a_junc_seg.road_idxs.end()){
                target_junc_seg_idx = i;
                break;
            }
        }

        if(source_junc_seg_idx != -1 && target_junc_seg_idx != -1){
            // Connect source_junc_seg_idx and target_junc_seg_idx
            float min_dist = POSITIVE_INFINITY;
            int min_source_idx = -1;
            int min_target_idx = -1;
            for(int j = 0; j < junc_segment_vertices[source_junc_seg_idx].size(); ++j){
                road_graph_vertex_descriptor source_v = junc_segment_vertices[source_junc_seg_idx][j];
                RoadPt& source_v_pt = road_graph_[source_v].pt;
                Eigen::Vector2d source_v_dir = headingTo2dVector(source_v_pt.head);
                for(int k = 0; k < junc_segment_vertices[target_junc_seg_idx].size(); ++k){
                    road_graph_vertex_descriptor target_v = junc_segment_vertices[target_junc_seg_idx][k];
                    RoadPt& target_v_pt = road_graph_[target_v].pt;
                    Eigen::Vector2d vec(target_v_pt.x - source_v_pt.x,
                                        target_v_pt.y - source_v_pt.y);
                    if(vec.dot(source_v_dir) > 0){
                        float dist = vec.norm();
                        if(min_dist > dist){
                            min_dist = dist;
                            min_source_idx = j;
                            min_target_idx = k;
                        }
                    }
                }
            }
            if(min_source_idx != -1 && min_target_idx != -1){
                // add link edge between min_source_idx and min_target_idx
                auto new_e = add_edge(junc_segment_vertices[source_junc_seg_idx][min_source_idx],
                                      junc_segment_vertices[target_junc_seg_idx][min_target_idx],
                                      road_graph_);
                if(new_e.second){
                    road_graph_[new_e.first].type = RoadGraphEdgeType::linking;
                    road_graph_[new_e.first].length = min_dist;
                    already_added_edges[pair<int, int>(source_junc_seg_idx, target_junc_seg_idx)] = new_e.first;
                }
            }
            // Connect target_junc_seg_idx and source_junc_seg_idx
            min_dist = POSITIVE_INFINITY;
            min_source_idx = -1;
            min_target_idx = -1;
            for(int j = 0; j < junc_segment_vertices[target_junc_seg_idx].size(); ++j){
                road_graph_vertex_descriptor target_v = junc_segment_vertices[target_junc_seg_idx][j];
                RoadPt& target_v_pt = road_graph_[target_v].pt;
                Eigen::Vector2d target_v_dir = headingTo2dVector(target_v_pt.head);
                for(int k = 0; k < junc_segment_vertices[source_junc_seg_idx].size(); ++k){
                    road_graph_vertex_descriptor source_v = junc_segment_vertices[source_junc_seg_idx][k];
                    RoadPt& source_v_pt = road_graph_[source_v].pt;
                    Eigen::Vector2d vec(source_v_pt.x - target_v_pt.x,
                                        source_v_pt.y - target_v_pt.y);
                    if(vec.dot(target_v_dir) > 0){
                        float dist = vec.norm();
                        if(min_dist > dist){
                            min_dist = dist;
                            min_target_idx = j;
                            min_source_idx = k;
                        }
                    }
                }
            }
            if(min_source_idx != -1 && min_target_idx != -1){
                // add link edge between min_source_idx and min_target_idx
                auto new_e1 = add_edge(junc_segment_vertices[target_junc_seg_idx][min_target_idx],
                                       junc_segment_vertices[source_junc_seg_idx][min_source_idx],
                                       road_graph_);
                if(new_e1.second){
                    road_graph_[new_e1.first].type = RoadGraphEdgeType::linking;
                    road_graph_[new_e1.first].length = min_dist;
                    already_added_edges[pair<int, int>(target_junc_seg_idx, source_junc_seg_idx)] = new_e1.first;
                    // Connect source_junc_seg_idx and target_junc_seg_idx
                    float min_dist = POSITIVE_INFINITY;
                    int min_source_idx = -1;
                    int min_target_idx = -1;
                    for(int j = 0; j < junc_segment_vertices[source_junc_seg_idx].size(); ++j){
                        road_graph_vertex_descriptor source_v = junc_segment_vertices[source_junc_seg_idx][j];
                        RoadPt& source_v_pt = road_graph_[source_v].pt;
                        Eigen::Vector2d source_v_dir = headingTo2dVector(source_v_pt.head);
                        for(int k = 0; k < junc_segment_vertices[target_junc_seg_idx].size(); ++k){
                            road_graph_vertex_descriptor target_v = junc_segment_vertices[target_junc_seg_idx][k];
                            RoadPt& target_v_pt = road_graph_[target_v].pt;
                            Eigen::Vector2d vec(target_v_pt.x - source_v_pt.x,
                                                target_v_pt.y - source_v_pt.y);
                            if(vec.dot(source_v_dir) > 0){
                                float dist = vec.norm();
                                if(min_dist > dist){
                                    min_dist = dist;
                                    min_source_idx = j;
                                    min_target_idx = k;
                                }
                            }
                        }
                    }
                    if(min_source_idx != -1 && min_target_idx != -1){
                        // add link edge between min_source_idx and min_target_idx
                        auto new_e = add_edge(junc_segment_vertices[source_junc_seg_idx][min_source_idx],
                                              junc_segment_vertices[target_junc_seg_idx][min_target_idx],
                                              road_graph_);
                        if(new_e.second){
                            road_graph_[new_e.first].type = RoadGraphEdgeType::linking;
                            road_graph_[new_e.first].length = min_dist;
                            already_added_edges[pair<int, int>(source_junc_seg_idx, target_junc_seg_idx)] = new_e.first;
                        }
                    }
                }
            }
        }
    }  

    // Process the remaining links
    vector<road_graph_edge_descriptor> more_edges;
    set<pair<int, int>> processed_link_pairs;
    for(size_t m = 0; m < links.size(); ++m){
        Link& a_link = links[m];
        
        float dh = abs(deltaHeading1MinusHeading2(road_graph_[a_link.source_vertex].pt.head,
                                                  road_graph_[a_link.target_vertex].pt.head));
        if(dh > 180 - ANGLE_OPPO_THRESHOLD)
            continue;

        // Ignore opposite links
        bool is_opposite_link = false;
        for (const auto& oppo_relation : oppo_relations) { 
            bool s_covered = false;
            if(oppo_relation.h1_road_idxs.find(a_link.source_related_road_idx) != oppo_relation.h1_road_idxs.end() ||
                    oppo_relation.h2_road_idxs.find(a_link.source_related_road_idx) != oppo_relation.h2_road_idxs.end()){
                s_covered = true;
            }
            bool t_covered = false;
            if(oppo_relation.h1_road_idxs.find(a_link.target_related_road_idx) != oppo_relation.h1_road_idxs.end() ||
                    oppo_relation.h2_road_idxs.find(a_link.target_related_road_idx) != oppo_relation.h2_road_idxs.end()){
                t_covered = true;
            }

            if(s_covered && t_covered){
                continue;
            }
        } 
        
        int source_junc_seg_idx = -1;
        for(int i = 0; i < junc_segments.size(); ++i){
            JuncSegment& a_junc_seg = junc_segments[i];
            if(a_junc_seg.road_idxs.find(a_link.source_related_road_idx) != a_junc_seg.road_idxs.end()){
                source_junc_seg_idx = i;
                break;
            }
        }
        int target_junc_seg_idx = -1;
        for(int i = 0; i < junc_segments.size(); ++i){
            JuncSegment& a_junc_seg = junc_segments[i];
            if(a_junc_seg.road_idxs.find(a_link.target_related_road_idx) != a_junc_seg.road_idxs.end()){
                target_junc_seg_idx = i;
                break;
            }
        }
        if(source_junc_seg_idx != -1 && target_junc_seg_idx != -1){
            if(already_added_edges.find(pair<int, int>(source_junc_seg_idx, target_junc_seg_idx)) == already_added_edges.end()){
                // New edge needed
                // Connect source_junc_seg_idx and target_junc_seg_idx
                float min_dist = POSITIVE_INFINITY;
                int min_source_idx = -1;
                int min_target_idx = -1;
                for(int j = 0; j < junc_segment_vertices[source_junc_seg_idx].size(); ++j){
                    road_graph_vertex_descriptor source_v = junc_segment_vertices[source_junc_seg_idx][j];
                    RoadPt& source_v_pt = road_graph_[source_v].pt;
                    Eigen::Vector2d source_v_dir = headingTo2dVector(source_v_pt.head);
                    for(int k = 0; k < junc_segment_vertices[target_junc_seg_idx].size(); ++k){
                        road_graph_vertex_descriptor target_v = junc_segment_vertices[target_junc_seg_idx][k];
                        RoadPt& target_v_pt = road_graph_[target_v].pt;
                        Eigen::Vector2d vec(target_v_pt.x - source_v_pt.x,
                                            target_v_pt.y - source_v_pt.y);
                        if(vec.dot(source_v_dir) > 0){
                            float dist = vec.norm();
                            if(min_dist > dist){
                                min_dist = dist;
                                min_source_idx = j;
                                min_target_idx = k;
                            }
                        }
                    }
                }
                if(min_source_idx != -1 && min_target_idx != -1){
                    // add link edge between min_source_idx and min_target_idx
                    auto new_e = add_edge(junc_segment_vertices[source_junc_seg_idx][min_source_idx],
                                          junc_segment_vertices[target_junc_seg_idx][min_target_idx],
                                          road_graph_);
                    if(new_e.second){
                        road_graph_[new_e.first].type = RoadGraphEdgeType::linking;
                        road_graph_[new_e.first].length = min_dist;
                        already_added_edges[pair<int, int>(source_junc_seg_idx, target_junc_seg_idx)] = new_e.first;
                    }
                }
            }
        }
        else{
            if(processed_link_pairs.find(pair<int, int>(a_link.source_related_road_idx, a_link.target_related_road_idx)) != processed_link_pairs.end()){
                continue;
            }
            processed_link_pairs.emplace(pair<int, int>(a_link.source_related_road_idx, a_link.target_related_road_idx));
            processed_link_pairs.emplace(pair<int, int>(a_link.target_related_road_idx, a_link.source_related_road_idx));
            int will_remove_under_n = 5;
            if(source_junc_seg_idx != -1){
                // Source junc seg is registered
                vector<road_graph_vertex_descriptor>& source_road = junc_segment_vertices[source_junc_seg_idx];
                vector<road_graph_vertex_descriptor>& target_road = related_roads[a_link.target_related_road_idx];
                RoadPt& to_r_pt = road_graph_[a_link.target_vertex].pt;
                // Find from_v
                float min_s_dist = POSITIVE_INFINITY;
                float min_s_idx = -1;
                for(int s = 0; s < source_road.size(); ++s){
                    float delta_x = road_graph_[source_road[s]].pt.x - road_graph_[a_link.source_vertex].pt.x;
                    float delta_y = road_graph_[source_road[s]].pt.y - road_graph_[a_link.source_vertex].pt.y;
                    float dist_sqr = delta_x*delta_x + delta_y*delta_y;
                    if(min_s_dist > dist_sqr){
                        min_s_dist = dist_sqr;
                        min_s_idx = s;
                    }
                }
                RoadPt& from_r_pt = road_graph_[source_road[min_s_idx]].pt;
                // Find best connection
                road_graph_vertex_descriptor best_s, best_t; 
                float cur_min_score = 1e9;
                for(int s = 0; s < source_road.size(); ++s){
                    RoadPt& s_pt = road_graph_[source_road[s]].pt;
                    Eigen::Vector2d vec(to_r_pt.x - s_pt.x,
                                        to_r_pt.y - s_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(to_r_pt.head, vec_head));
                    
                    float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                    if(score < cur_min_score){
                        best_s = source_road[s];
                        best_t = a_link.target_vertex;
                        cur_min_score = score;
                    }
                }
                for(int s = 0; s < target_road.size(); ++s){
                    RoadPt& t_pt = road_graph_[target_road[s]].pt;
                    Eigen::Vector2d vec(t_pt.x - from_r_pt.x,
                                        t_pt.y - from_r_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(from_r_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                    
                    float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                    if(score < cur_min_score){
                        best_s = source_road[min_s_idx];
                        best_t = target_road[s];
                        cur_min_score = score;
                    }
                }

                // For target road, if after best_t is close to one of its end, remove it
                int best_t_idx = 0;
                for(int l = 0; l < target_road.size(); ++l){
                    if(target_road[l] == best_t)
                        best_t_idx = l;
                }

                if(best_t_idx < will_remove_under_n){
                    for(int l = 0; l < best_t_idx; ++l){
                        road_graph_[target_road[l]].is_valid = false;
                        road_graph_[target_road[l]].road_label = -1;
                        road_graph_[target_road[l]].cluster_id = -1;
                        clear_vertex(target_road[l], road_graph_);
                    }
                }
                if(target_road.size() - best_t_idx < will_remove_under_n){
                    for(int l = best_t_idx; l < target_road.size(); ++l){
                        road_graph_[target_road[l]].is_valid = false;
                        road_graph_[target_road[l]].road_label = -1;
                        road_graph_[target_road[l]].cluster_id = -1;
                        clear_vertex(target_road[l], road_graph_);
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
                        more_edges.emplace_back(es.first);
                    }
                    prev_v = new_v;
                }
                
                if(new_v_road_label != road_graph_[best_t].road_label){
                    road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                    visited_vertices.emplace(new_v);
                    auto es = add_edge(prev_v, new_v, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        more_edges.emplace_back(es.first);
                    }
                    visited_vertices.emplace(new_v);
                    road_graph_[new_v].pt = new_to_r_pt;
                    road_graph_[new_v].road_label = new_v_road_label;
                    es = add_edge(new_v, best_t, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        more_edges.emplace_back(es.first);
                    }
                }
                else{
                    auto es = add_edge(prev_v, best_t, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        more_edges.emplace_back(es.first);
                    }
                }
            }
            else if(target_junc_seg_idx != -1){
                // Target junc seg is registered
                vector<road_graph_vertex_descriptor>& target_road = junc_segment_vertices[target_junc_seg_idx];
                vector<road_graph_vertex_descriptor>& source_road = related_roads[a_link.source_related_road_idx];
                RoadPt& from_r_pt = road_graph_[a_link.source_vertex].pt;
                // Find to_v 
                float min_t_dist = POSITIVE_INFINITY;
                float min_t_idx = -1;
                for(int s = 0; s < target_road.size(); ++s){
                    float delta_x = road_graph_[target_road[s]].pt.x - road_graph_[a_link.target_vertex].pt.x;
                    float delta_y = road_graph_[target_road[s]].pt.y - road_graph_[a_link.target_vertex].pt.y;
                    float dist_sqr = delta_x*delta_x + delta_y*delta_y;
                    if(min_t_dist > dist_sqr){
                        min_t_dist = dist_sqr;
                        min_t_idx = s;
                    }
                }
                RoadPt& to_r_pt = road_graph_[target_road[min_t_idx]].pt;
                // Find best connection
                road_graph_vertex_descriptor best_s, best_t; 
                float cur_min_score = POSITIVE_INFINITY;
                for(int s = 0; s < source_road.size(); ++s){
                    RoadPt& s_pt = road_graph_[source_road[s]].pt;
                    Eigen::Vector2d vec(to_r_pt.x - s_pt.x,
                                        to_r_pt.y - s_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(to_r_pt.head, vec_head));
                    
                    float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                    if(score < cur_min_score){
                        best_s = source_road[s];
                        best_t = a_link.target_vertex;
                        cur_min_score = score;
                    }
                }
                for(int s = 0; s < target_road.size(); ++s){
                    RoadPt& t_pt = road_graph_[target_road[s]].pt;
                    Eigen::Vector2d vec(t_pt.x - from_r_pt.x,
                                        t_pt.y - from_r_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(from_r_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                    
                    float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                    if(score < cur_min_score){
                        best_s = a_link.source_vertex;
                        best_t = target_road[s];
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
                        more_edges.emplace_back(es.first);
                    }
                    prev_v = new_v;
                }
                
                if(new_v_road_label != road_graph_[best_t].road_label){
                    road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                    visited_vertices.emplace(new_v);
                    auto es = add_edge(prev_v, new_v, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        more_edges.emplace_back(es.first);
                    }
                    visited_vertices.emplace(new_v);
                    road_graph_[new_v].pt = new_to_r_pt;
                    road_graph_[new_v].road_label = new_v_road_label;
                    es = add_edge(new_v, best_t, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        more_edges.emplace_back(es.first);
                    }
                }
                else{
                    auto es = add_edge(prev_v, best_t, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        more_edges.emplace_back(es.first);
                    }
                }
            }
            else{
                // Two non-registered roads in junc_segments
                vector<road_graph_vertex_descriptor>& source_road = related_roads[a_link.source_related_road_idx];
                vector<road_graph_vertex_descriptor>& target_road = related_roads[a_link.target_related_road_idx];
                RoadPt& from_r_pt = road_graph_[a_link.source_vertex].pt;
                RoadPt& to_r_pt = road_graph_[a_link.target_vertex].pt;
                // Find best connection
                road_graph_vertex_descriptor best_s, best_t; 
                float cur_min_score = 1e9;
                int extension = 20;
                for(int s = a_link.source_idx_in_related_roads - extension; s <= a_link.source_idx_in_related_roads + extension; ++s){
                    if(s < 0)
                        continue;
                    if(s >= source_road.size())
                        break;

                    RoadPt& s_pt = road_graph_[source_road[s]].pt;
                    Eigen::Vector2d vec(to_r_pt.x - s_pt.x,
                                        to_r_pt.y - s_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(to_r_pt.head, vec_head));
                    
                    float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                    if(score < cur_min_score){
                        best_s = source_road[s];
                        best_t = a_link.target_vertex;
                        cur_min_score = score;
                    }
                }
                for(int s = a_link.target_idx_in_related_roads - extension; s < a_link.target_idx_in_related_roads + extension; ++s){
                    if(s < 0)
                        continue;
                    if(s >= target_road.size())
                        break;
                    RoadPt& t_pt = road_graph_[target_road[s]].pt;
                    Eigen::Vector2d vec(t_pt.x - from_r_pt.x,
                                        t_pt.y - from_r_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(from_r_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                    
                    float score = (delta_h1 + delta_h2) * (vec_length + 1.0f);
                    if(score < cur_min_score){
                        best_s = a_link.source_vertex;
                        best_t = target_road[s];
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
                        more_edges.emplace_back(es.first);
                    }
                    prev_v = new_v;
                }
                
                if(new_v_road_label != road_graph_[best_t].road_label){
                    road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                    visited_vertices.emplace(new_v);
                    auto es = add_edge(prev_v, new_v, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        more_edges.emplace_back(es.first);
                    }
                    visited_vertices.emplace(new_v);
                    road_graph_[new_v].pt = new_to_r_pt;
                    road_graph_[new_v].road_label = new_v_road_label;
                    es = add_edge(new_v, best_t, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        more_edges.emplace_back(es.first);
                    }
                }
                else{
                    auto es = add_edge(prev_v, best_t, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        more_edges.emplace_back(es.first);
                    }
                }
            }
        }
    }

    // Visualize added edges
    for (const auto& new_edge : already_added_edges) { 
        road_graph_vertex_descriptor source_v = source(new_edge.second, road_graph_);
        road_graph_vertex_descriptor target_v = target(new_edge.second, road_graph_);
        RoadPt& source_r_pt = road_graph_[source_v].pt;
        RoadPt& target_r_pt = road_graph_[target_v].pt;
        Color c = ColorMap::getInstance().getNamedColor(ColorMap::PINK);
        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(source_r_pt.x, source_r_pt.y, Z_DEBUG + 0.02f));
        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(target_r_pt.x, target_r_pt.y, Z_DEBUG + 0.02f));
        line_colors_.emplace_back(c);
        line_colors_.emplace_back(Color(c.r * 0.1f, c.g * 0.1f, c.b * 0.1f, 1.0f));
    } 

    for (const auto& new_edge : more_edges) { 
        road_graph_vertex_descriptor source_v = source(new_edge, road_graph_);
        road_graph_vertex_descriptor target_v = target(new_edge, road_graph_);
        RoadPt& source_r_pt = road_graph_[source_v].pt;
        RoadPt& target_r_pt = road_graph_[target_v].pt;
        Color c = ColorMap::getInstance().getNamedColor(ColorMap::PINK);
        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(source_r_pt.x, source_r_pt.y, Z_DEBUG + 0.02f));
        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(target_r_pt.x, target_r_pt.y, Z_DEBUG + 0.02f));
        line_colors_.emplace_back(c);
        line_colors_.emplace_back(Color(c.r * 0.1f, c.g * 0.1f, c.b * 0.1f, 1.0f));
    }

    /*
     *if(perp_relations.size() > 0){
     *    cout << "\tPerpendicular relations:" << endl;
     *    for (const auto& perp_relation : perp_relations) { 
     *        cout << "\t\t" << perp_relation.first << " is perpendicular to " << perp_relation.second << endl; 
     *    }
     *}
     */

    return;

    // Visualize edges and junctions
    if(debug_mode_){
        //line_colors_.clear();
        //lines_to_draw_.clear();
        //for (const auto& vertex : visited_vertices) { 
        //auto es = out_edges(vertex, road_graph_);

        //Color c = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE);
        //for(auto eit = es.first; eit != es.second; ++eit){
        //    road_graph_vertex_descriptor s_vertex, t_vertex;
        //    s_vertex = source(*eit, road_graph_);
        //    t_vertex = target(*eit, road_graph_);
        //    RoadPt& s_pt = road_graph_[s_vertex].pt;
        //    RoadPt& t_pt = road_graph_[t_vertex].pt;
        //    lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(s_pt.x, s_pt.y, Z_DEBUG + 0.01f));
        //    lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(t_pt.x, t_pt.y, Z_DEBUG + 0.01f));
        //    line_colors_.emplace_back(c);
        //    line_colors_.emplace_back(Color(0.1f*c.r, 0.1f*c.g, 0.1f*c.b, 1.0f));
        //}
        //}
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

    // Draw generated road map
    if(show_generated_map_){
        if(generated_map_render_mode_ == GeneratedMapRenderingMode::realistic){
            // Draw line loops
            for(int i = 0; i < generated_map_line_loops_.size(); ++i){
                vector<Vertex> &a_loop = generated_map_line_loops_[i];
                vector<Color> &a_loop_color = generated_map_line_loop_colors_[i];
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&a_loop[0], 3*a_loop.size()*sizeof(float));
                shader_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&a_loop_color[0], 4*a_loop_color.size()*sizeof(float));
                shader_program_->setupColorAttributes();
                glLineWidth(5.0);
                glDrawArrays(GL_LINE_LOOP, 0, a_loop.size());
            }

            // Draw junctions
            float junc_point_size = 20.0f;
            QOpenGLBuffer position_buffer;
            position_buffer.create();
            position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            position_buffer.bind();
            position_buffer.allocate(&generated_map_points_[0], 3*generated_map_points_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            QOpenGLBuffer color_buffer;
            color_buffer.create();
            color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            color_buffer.bind();
            color_buffer.allocate(&generated_map_point_colors_[0], 4*generated_map_point_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            glPointSize(junc_point_size);
            glDrawArrays(GL_POINTS, 0, generated_map_points_.size());
            
            // Draw links
            QOpenGLBuffer new_buffer;
            new_buffer.create();
            new_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_buffer.bind();
            new_buffer.allocate(&generated_map_lines_[0], 3*generated_map_lines_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            QOpenGLBuffer new_color_buffer;
            new_color_buffer.create();
            new_color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_color_buffer.bind();
            new_color_buffer.allocate(&generated_map_line_colors_[0], 4*generated_map_line_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            glLineWidth(8.0);
            glDrawArrays(GL_LINES, 0, generated_map_lines_.size()); 
        }
        else if(generated_map_render_mode_ == GeneratedMapRenderingMode::skeleton){
            // Draw links
            QOpenGLBuffer new_buffer;
            new_buffer.create();
            new_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_buffer.bind();
            new_buffer.allocate(&generated_map_lines_[0], 3*generated_map_lines_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            QOpenGLBuffer new_color_buffer;
            new_color_buffer.create();
            new_color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_color_buffer.bind();
            new_color_buffer.allocate(&generated_map_line_colors_[0], 4*generated_map_line_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            glLineWidth(8.0);
            glDrawArrays(GL_LINES, 0, generated_map_lines_.size());

            // Draw junctions
            float junc_point_size = 10.0f;
            QOpenGLBuffer position_buffer;
            position_buffer.create();
            position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            position_buffer.bind();
            position_buffer.allocate(&generated_map_points_[0], 3*generated_map_points_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            QOpenGLBuffer color_buffer;
            color_buffer.create();
            color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            color_buffer.bind();
            color_buffer.allocate(&generated_map_point_colors_[0], 4*generated_map_point_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            glPointSize(junc_point_size);
            glDrawArrays(GL_POINTS, 0, generated_map_points_.size());
        }
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

    // Clear generated map
    generated_map_points_.clear();
    generated_map_point_colors_.clear();
    generated_map_line_loops_.clear();
    generated_map_line_loop_colors_.clear();
    generated_map_lines_.clear();
    generated_map_line_colors_.clear();

    tmp_ = 0;
}

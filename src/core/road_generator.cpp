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

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/connected_components.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>

using namespace Eigen;

RoadGenerator::RoadGenerator(QObject *parent, std::shared_ptr<Trajectories> trajectories) : 
    Renderable(parent),
    trajectories_(trajectories),
    point_cloud_(new PclPointCloud),
    search_tree_(new pcl::search::FlannSearch<PclPoint>(false)),
    simplified_traj_points_(new PclPointCloud),
    simplified_traj_point_search_tree_(new pcl::search::FlannSearch<PclPoint>(false)),
    grid_points_(new PclPointCloud),
    grid_point_search_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    has_been_covered_.clear();
    if(trajectories_ != nullptr){
        if(trajectories_->data()->size() > 0){
            has_been_covered_.resize(trajectories_->data()->size(), false);
        }
    }
    
    tmp_ = 0;
}

RoadGenerator::~RoadGenerator(){
}

void RoadGenerator::applyRules(){
}

void RoadGenerator::pointBasedVoting(){
    simplified_traj_points_->clear();
   
    sampleGPSPoints(5.0f,
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
    map<int, vector<float> > grid_angle_votes;
    
    float max_vote = 0.0f;
    for (size_t i = 0; i < simplified_traj_points_->size(); ++i) {
        PclPoint& pt = simplified_traj_points_->at(i);
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
                
                float adjusted_sigma_w = sigma_w;
                if(pt.speed < 1.5f * avg_speed){
                    adjusted_sigma_w = sigma_w * avg_speed / (pt.speed + 0.1f);
                }

                float vote = pt.id_sample *
                exp(-1.0f * dot_value_sqr / 2.0f / sigma_h / sigma_h) *
                exp(-1.0f * perp_value_sqr / 2.0f / adjusted_sigma_w / adjusted_sigma_w);
                
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
                    pt.head = floor((idx + 0.5f) * 15.0f);
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
    float MAX_DELTA_HEADING = 5.0f; // in degree

    // Extract peaks
    for (size_t i = 0; i < grid_points_->size(); ++i) {
        if(grid_votes_[i] < VOTE_THRESHOLD) 
            continue;

        PclPoint& g_pt = grid_points_->at(i);
        Eigen::Vector2d dir = headingTo2dVector(g_pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        grid_point_search_tree_->radiusSearch(g_pt, 10.0f, k_indices, k_dist_sqrs);

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
    initial_roads_.clear();
   
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
    
    typedef adjacency_list<vecS, vecS, undirectedS >    graph_t;
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
                    PclPoint& target_g_pt = grid_points_->at(target_idx);
                    PclPoint& source_g_pt = grid_points_->at(best_fwd_candidate);
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
                    PclPoint& source_g_pt = grid_points_->at(source_idx);
                    PclPoint& target_g_pt = grid_points_->at(best_bwd_candidate);
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
        for(size_t j = 1; j < sorted_cluster.size()-1; ++j){
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
                               2.5f,
                               true);
            
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
        
        if(cum_length >= 100.0f){
            roads.push_back(a_road);
        }
    }
    
    // Visualization
    if (new_road == nullptr) {
        vertex_t u = add_vertex(symbol_graph_);
        graph_nodes_[u] = std::move(std::shared_ptr<Symbol>(new RoadSymbol));
        graph_nodes_[u]->vertex_descriptor() = u;
        new_road = dynamic_pointer_cast<RoadSymbol>(graph_nodes_[u]);
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
        for (size_t j = 0; j < a_road.size(); ++j) {
            RoadPt& r_pt = a_road[j];
            r_pt.n_lanes = cum_n_lanes;
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5f * cum_n_lanes * LANE_WIDTH * Eigen::Vector2f(-1*direction[1], direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            xs.push_back(v1.x());
            ys.push_back(v1.y());
        }
        
        for (int j = a_road.size() - 1; j >= 0; --j) {
            RoadPt& r_pt = a_road[j];
            
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5 * cum_n_lanes * LANE_WIDTH * Eigen::Vector2f(direction[1], -1.0f * direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            xs.push_back(v1.x());
            ys.push_back(v1.y());
        }
        
        xs.push_back(xs[0]);
        ys.push_back(ys[0]);
       
        float color_value = 1.0f - static_cast<float>(i) / roads.size(); 

        for (size_t j = 0; j < xs.size() - 1; ++j) {
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j], ys[j], Z_ROAD));
            //line_colors_.push_back(ColorMap::getInstance().getDiscreteColor(i));
            line_colors_.push_back(ColorMap::getInstance().getJetColor(color_value));
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j+1], ys[j+1], Z_ROAD));
            //line_colors_.push_back(ColorMap::getInstance().getDiscreteColor(i));
            line_colors_.push_back(ColorMap::getInstance().getJetColor(color_value));
        }
        initial_roads_.push_back(a_road);
    }
    
    return true;
}

void RoadGenerator::tmpFunc(){
    if(new_road == nullptr){
        return;
    }

    vector<vector<RoadPt> > branches;
    set<int> candidate_point_set;
    vector<vector<int> > segments;
    vector<float> segment_scores;

    getConsistentPointSetForRoad(new_road,
                                 candidate_point_set,
                                 segments,
                                 segment_scores);

    RoadPt junction_loc;

    branchPrediction(new_road->centerLine(),
                     candidate_point_set,
                     trajectories_,
                     junction_loc,
                     branches);

    // Visualization
    if (branches.size() > 0) {
        feature_vertices_.clear();
        feature_colors_.clear();
        points_to_draw_.clear();
        point_colors_.clear();
        
        // Visualize candidate point set
        //for(set<int>::iterator it = candidate_point_set.begin(); it != candidate_point_set.end(); ++it){
            //PclPoint& pt = trajectories_->data()->at(*it);
            //points_to_draw_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG+0.05f));
            //point_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
        //}
        //return;
        
        // Visualize segments
        //float max_score = 0.0f;
        //for(size_t i = 0; i < segment_scores.size(); ++i){
        //    if(segment_scores[i] > max_score){
        //        max_score = segment_scores[i];
        //    }
        //}
        
        //for (size_t i = 0; i < segments.size(); ++i) {
        //    vector<int>& segment = segments[i];
        //    float c_value = segment_scores[i] / max_score;
        //    // Draw points
        //    for (size_t j = 0; j < segment.size(); ++j) {
        //        PclPoint& pt = trajectories_->data()->at(segment[j]);
                
        //        points_to_draw_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
        //        point_colors_.push_back(ColorMap::getInstance().getJetColor(c_value));
        //    }
        //}
        
        //int n = segments.size();
        //if(n > 25){
        //    n = 25;
        //}
        //for (size_t i = 0; i < n; ++i) {
        //    // Draw lines
        //    vector<int>& segment = segments[i];
        //    float c_value = segment_scores[i] / max_score;
        //    for(int j = 1; j < segment.size(); ++j){
        //        PclPoint& p1 = trajectories_->data()->at(segment[j-1]);
        //        PclPoint& p2 = trajectories_->data()->at(segment[j]);
            
        //        lines_to_draw_.push_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG+0.05f));
        //        line_colors_.push_back(ColorMap::getInstance().getJetColor(c_value));
            
        //        lines_to_draw_.push_back(SceneConst::getInstance().normalize(p2.x, p2.y, Z_DEBUG+0.05f));
        //        line_colors_.push_back(ColorMap::getInstance().getJetColor(c_value));
        //    }
        //}
        //return;
        
         // Visualize grid_point voting field
        /*
        vector<RoadPt>& branch = branches[0];
        for(size_t s = 0; s < branch.size(); ++s){
            RoadPt& r_pt = branch[s];
            float c_value = static_cast<float>(r_pt.n_lanes) / 100.0f;
            points_to_draw_.push_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG+0.05f));
            point_colors_.push_back(ColorMap::getInstance().getJetColor(c_value));
        }
        return;
         */
        
        // Visualize local maximum
        /*
         vector<RoadPt>& branch = branches[0];
         for(size_t s = 0; s < branch.size(); ++s){
             RoadPt& r_pt = branch[s];
             float c_value = static_cast<float>(r_pt.n_lanes) / 100.0f;
             points_to_draw_.push_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG+0.05f));
             point_colors_.push_back(ColorMap::getInstance().getJetColor(c_value));
         }
         return;
         */
        
        // Visualize connecte component
//        vector<RoadPt>& branch = branches[0];
        /*
        int k = branch.size() / 2;
        for (int j = 0; j < k; ++j) {
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(branch[2*j].x, branch[2*j].y, Z_DEBUG + 0.05f));
            line_colors_.push_back(Color(1.0f, 0.0f, 0.0f, 1.0f));
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(branch[2*j+1].x, branch[2*j+1].y, Z_DEBUG + 0.05f));
            line_colors_.push_back(Color(1.0f, 0.0f, 0.0f, 1.0f));
        }
        return;
         */
        
        // Visualize the branch structure of a road
        for (size_t i = 0; i < branches.size(); ++i) {
            vector<RoadPt>& branch = branches[i];
            float total_n_pt = branch.size();
            //Color c = ColorMap::getInstance().getNamedColor(ColorMap::RED);
            float color_value = branch[0].n_lanes / 100.0f;
            Color c = ColorMap::getInstance().getJetColor(color_value);
            for (int j = 1; j < branch.size(); ++j) {
                float c_value1 = 1.0f - 0.5f * (j-1) / total_n_pt;
                float c_value2 = 1.0f - 0.5f * j / total_n_pt;
                lines_to_draw_.push_back(SceneConst::getInstance().normalize(branch[j-1].x, branch[j-1].y, Z_DEBUG + 0.05f));
                line_colors_.push_back(Color(c.r*c_value1, c.g*c_value1, c.b*c_value1, 1.0f));
                lines_to_draw_.push_back(SceneConst::getInstance().normalize(branch[j].x, branch[j].y, Z_DEBUG + 0.05f));
                line_colors_.push_back(Color(c.r*c_value2, c.g*c_value2, c.b*c_value2, 1.0f));
            }
        }
    }
}

bool RoadGenerator::addInitialRoad(){
    // Create initial roads and junctions by traversing these raw roads.
    if(initial_roads_.size() == 0){
        cout << "WARNING from addInitialRoad: generate initial roads first!" << endl;
        return false;
    }
    
    PclPoint point;
    radius = 25.0f;
    bool DEBUG = true;
    
    if(DEBUG){
        cout << "Running DEBUG mode from RoadGenerator::addInitialRoad()" << endl;
        
        feature_vertices_.clear();
        feature_colors_.clear();
        points_to_draw_.clear();
        point_colors_.clear();
        lines_to_draw_.clear();
        line_colors_.clear();
        
        bool INTERACTIVE_ROAD_GROWING_MODE = true;
        if (INTERACTIVE_ROAD_GROWING_MODE) {
            i_road = tmp_ % initial_roads_.size();
            
            vector<RoadPt>& road = initial_roads_[i_road];
            cout << "Current road: " << i_road <<", which has " << road.size() << " nodes" << endl;
            if (road.size() < 2) {
                tmp_++;
                return true;
            }
            
            if(new_road == nullptr){
                cout << "ERROR from tmpFunc: new_road is NULL!" << endl;
                exit(1);
            }
            
            cout << "new road size: " << new_road->centerLine().size() <<endl;
            new_road->clearCenter();
            
            for (size_t i = 0; i < road.size(); ++i) {
                RoadPt& r_pt = road[i];
                new_road->addRoadPtAtEnd(r_pt);
            }
        }
        else{
            bool grow_backward = false;
            int pt_idx = tmp_;
            int i_road = 0;
            int cum_pt_count = 0;
            while(true){
                if (cum_pt_count + initial_roads_[i_road].size() > tmp_) {
                    break;
                }
                else{
                    cum_pt_count += initial_roads_[i_road].size();
                    i_road++;
                }
            }
            pt_idx = tmp_ - cum_pt_count;
            
            if (i_road > initial_roads_.size()) {
                tmp_ = 0;
                return true;
            }
            
            vector<RoadPt> center_line;
            vector<RoadPt>& road = initial_roads_[i_road];
            
            for (int s = 0; s < pt_idx - 1; ++s) {
                center_line.push_back(road[s]);
                feature_vertices_.push_back(SceneConst::getInstance().normalize(road[s].x, road[s].y, Z_FEATURES));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::YELLOW));
            }
            
            center_line.push_back(road[pt_idx]);
            feature_vertices_.push_back(SceneConst::getInstance().normalize(road[pt_idx].x, road[pt_idx].y, Z_FEATURES));
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            
            float cur_cum_length = 0.0f;
            for (int s = 1; s < center_line.size(); ++s) {
                float delta_x = center_line[s].x - center_line[s-1].x;
                float delta_y = center_line[s].y - center_line[s-1].y;
                cur_cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
            }
        }
        
        tmp_++;
        
        return true;
    }
    
    return true;
}

void RoadGenerator::updateGPSPointsOnRoad(std::shared_ptr<RoadSymbol> road){
    if(road == NULL || trajectories_ == NULL){
        return;
    }
    
    const vector<RoadPt>& center_line = road->centerLine();
    vector<bool>&   center_pt_visited = road->centerPtVisited();
    
    if (center_line.size() < 2) {
        cout << "WARNING from computePointsOnRoad: center_line size less than 2." << endl;
        return;
    }
    
    if (center_line.size() != center_pt_visited.size()) {
        cout << "ERROR from computePointsOnRoad: center_line and center_pt_visited have different sizes." << endl;
        exit(1);
        return;
    }
    
    float heading_threshold = 10.0f; // in degrees
    float gps_sigma = Parameters::getInstance().gpsSigma();
    
    float search_radius = Parameters::getInstance().searchRadius();
    
    for (size_t i = 0; i < center_line.size(); ++i) {
        if (center_pt_visited[i]) {
            // The center_line point has been visited before, no need to update again.
            continue;
        }
        
        center_pt_visited[i] = true;
        
        PclPoint pt;
        pt.setCoordinate(center_line[i].x, center_line[i].y, 0.0f);
        
        const RoadPt& r_pt = center_line[i];
        
        Eigen::Vector3d dir = headingTo3dVector(r_pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        trajectories_->tree()->radiusSearch(pt,
                                            search_radius,
                                            k_indices,
                                            k_dist_sqrs);
        
        for (size_t s = 0; s < k_indices.size(); ++s) {
            int nb_pt_idx = k_indices[s];
            
            PclPoint& nb_pt = trajectories_->data()->at(nb_pt_idx);
            
            Eigen::Vector3d vec(nb_pt.x - r_pt.x,
                                nb_pt.y - r_pt.y,
                                0.0f);
            float parallel_dist = abs(dir.cross(vec)[2]);
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, r_pt.head));
            
            if (road->isOneway()) {
                if (delta_heading > heading_threshold) {
                    continue;
                }
                
                parallel_dist -= 0.5f * r_pt.n_lanes * LANE_WIDTH;
                if(parallel_dist < 0){
                    parallel_dist = 0.0f;
                }
                
                float probability = exp(-1.0f * parallel_dist * parallel_dist / 2.0f / gps_sigma / gps_sigma);
                road->insertPt(k_indices[s],
                               probability);
            }
            else{
                float bwd_delta_heading = 180.0f - delta_heading;
                parallel_dist -= 0.5f * r_pt.n_lanes * LANE_WIDTH;
                if(parallel_dist < 0){
                    parallel_dist = 0.0f;
                }
                
                float probability = exp(-1.0f * parallel_dist * parallel_dist / 2.0f / gps_sigma / gps_sigma);
                if(delta_heading < heading_threshold){
                    road->insertPt(k_indices[s],
                                   probability);
                }
                
                if(bwd_delta_heading < heading_threshold){
                    road->insertPt(k_indices[s],
                                   probability);
                }
            }
        }
    }
}

void RoadGenerator::getConsistentPointSetForRoad(std::shared_ptr<RoadSymbol> road,
                                                 set<int>& candidate_point_set,
                                                 vector<vector<int> >& segments,
                                                 vector<float>& segment_scores){
    /*
        Examines the GPS point that may falls on road, then extends the points to form segments w.r.t. maximum delta t extension and maximum radis.
        
        Output:
            - candidate_point_set: the point idxs w.r.t. trajectories_->data();
            - segments: each is a list of point idxs
            - segment_scores: summation of the probabilities of the points of this segments.
     */
    
    if(road == NULL || trajectories_ == NULL){
        return;
    }
    
    candidate_point_set.clear();
    
    Parameters& params = Parameters::getInstance();
    
    float MAX_RANGE = params.branchPredictorExtensionRatio() * params.searchRadius(); // in meters
    float MAX_T_EXTENSION = params.branchPredictorMaxTExtension();
    
    updateGPSPointsOnRoad(road);
    
    // Get all nearby gps_pts
    const set<int>& covered_pts = road->coveredPts();
    const map<int, float>& covered_pt_scores = road->coveredPtScores();

    set<int> already_marked_pts;
    for(set<int>::const_iterator it = covered_pts.begin(); it != covered_pts.end(); ++it){
        if(already_marked_pts.find(*it) != already_marked_pts.end()){
            continue;
        }

        // Extendthis this pt to form segments, and add corresponding points into candidate_point_set
        PclPoint& pt = trajectories_->data()->at(*it); 
        
        // Start a new segment
        vector<int> segment;
        float segment_score = 0.0f;
        segment.emplace_back(*it);
        candidate_point_set.emplace(*it);
        already_marked_pts.emplace(*it);
        segment_score += covered_pt_scores.at(*it);
        
        int id_trajectory = pt.id_trajectory;
        int id_sample = pt.id_sample;
        const vector<int>& traj = trajectories_->trajectories()[id_trajectory];

        // Look backward
        int last_on_road_pt_idx = *it; 
        for (int sid = id_sample - 1; sid >= 0; --sid) {
            int pt_idx = traj[sid];
            PclPoint& nb_pt = trajectories_->data()->at(pt_idx);  

            if(road->containsPt(pt_idx)){
                segment.emplace(segment.begin(), pt_idx);
                candidate_point_set.emplace(pt_idx);
                already_marked_pts.emplace(pt_idx);
                segment_score += covered_pt_scores.at(pt_idx);
                last_on_road_pt_idx = pt_idx;
            }
            else{
                PclPoint& last_pt_on_road = trajectories_->data()->at(last_on_road_pt_idx);  
                float delta_t = abs(nb_pt.t - last_pt_on_road.t);    
                if(delta_t > MAX_T_EXTENSION)  
                    break;
                float delta_x = last_pt_on_road.x - nb_pt.x;
                float delta_y = last_pt_on_road.y - nb_pt.y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                if(delta_dist > MAX_RANGE) 
                    break;

                // Add point to segment
                segment.emplace(segment.begin(), pt_idx);
                candidate_point_set.emplace(pt_idx);
            }
        }

        // Look forward 
        last_on_road_pt_idx = *it; 
        for (int sid = id_sample + 1; sid < traj.size(); ++sid) {
            int pt_idx = traj[sid];
            PclPoint& nb_pt = trajectories_->data()->at(pt_idx);  

            if(road->containsPt(pt_idx)){
                segment.emplace_back(pt_idx);
                candidate_point_set.emplace(pt_idx);
                already_marked_pts.emplace(pt_idx);
                segment_score += covered_pt_scores.at(pt_idx);
                last_on_road_pt_idx = pt_idx;
            }
            else{
                PclPoint& last_pt_on_road = trajectories_->data()->at(last_on_road_pt_idx);  
                float delta_t = abs(nb_pt.t - last_pt_on_road.t);    
                if(delta_t > MAX_T_EXTENSION)  
                    break;
                float delta_x = last_pt_on_road.x - nb_pt.x;
                float delta_y = last_pt_on_road.y - nb_pt.y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                if(delta_dist > MAX_RANGE) 
                    break;

                // Add point to segment
                segment.emplace_back(pt_idx);
                candidate_point_set.emplace(pt_idx);
            }
        }
        
        segments.emplace_back(segment);
        segment_scores.emplace_back(segment_score);
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
    // Draw generated road
    vector<Vertex> junction_vertices;
    int count = -1;
    for (map<vertex_t, std::shared_ptr<Symbol>>::iterator it = graph_nodes_.begin(); it != graph_nodes_.end(); ++it) {
        ++count;
        if (it->second->type() == ROAD) {
            std::shared_ptr<RoadSymbol> road = dynamic_pointer_cast<RoadSymbol>(it->second);
            vector<Vertex> v;
            if(road->getDrawingVertices(v)){
                vector<Color> color;
                color.resize(v.size(), ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
//                if (road->isOneway()) {
//                    color.resize(v.size(), ColorMap::getInstance().getNamedColor(ColorMap::ONEWAY_COLOR));
//                }
//                else{
//                    color.resize(v.size(), ColorMap::getInstance().getNamedColor(ColorMap::TWOWAY_COLOR));
//                }
                
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&v[0], 3*v.size()*sizeof(float));
                shader_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&color[0], 4*color.size()*sizeof(float));
                shader_program_->setupColorAttributes();
                glLineWidth(3.0);
                glDrawArrays(GL_LINE_LOOP, 0, v.size());
            }
            else{
                vector<Color> color;
                color.clear();
                color.resize(v.size(), Color(0,0,1,1));
                
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&v[0], 3*v.size()*sizeof(float));
                shader_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&color[0], 4*color.size()*sizeof(float));
                shader_program_->setupColorAttributes();
                glLineWidth(3.0);
                glDrawArrays(GL_LINES, 0, v.size());
            }
        }
        else{
            // Draw junction
            vector<Vertex> v;
            
            std::shared_ptr<JunctionSymbol> junction = dynamic_pointer_cast<JunctionSymbol>(it->second);
            junction->getDrawingVertices(v);
            junction_vertices.push_back(v[0]);
        }
    }
    
    if(junction_vertices.size() > 0){
        vector<Color> color;
        color.clear();
        color.resize(junction_vertices.size(), Color(1,1,0,1));
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&junction_vertices[0], 3*junction_vertices.size()*sizeof(float));
        shader_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&color[0], 4*color.size()*sizeof(float));
        shader_program_->setupColorAttributes();
        glPointSize(30);
        glDrawArrays(GL_POINTS, 0, junction_vertices.size());
    }
    
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
        glLineWidth(5.0);
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
    
    has_been_covered_.clear();
    
    feature_vertices_.clear();
    feature_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    
    cleanUp();
}

void RoadGenerator::cleanUp(){
    tmp_ = 0;
    initial_roads_.clear();
    symbol_graph_.clear();
    
    new_road.reset();
    
    graph_nodes_.clear();
}

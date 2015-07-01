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
#include <boost/graph/dijkstra_shortest_paths.hpp>

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
    test_i_ = 0;
    max_road_label_ = 0;
    max_junc_label_ = 0;
    cur_num_clusters_ = 0;
    debug_mode_ = false;

    osm_map_valid_ = false;
    road_graph_valid_ = false;

    traj_idx_to_show_ = 0;

    min_road_length_ = 50.0f;
    show_generated_map_ = true;
    has_unexplained_gps_pts_ = false;
    generated_map_render_mode_ = GeneratedMapRenderingMode::realistic;
}

RoadGenerator::~RoadGenerator(){
}

void RoadGenerator::updateOsmGraph(){
    osm_graph_.clear();
    osmMap_->updateMapSearchTree(5.0f);
    for (size_t i = 0; i < osmMap_->map_point_cloud()->size(); ++i) { 
        PclPoint& m_pt = osmMap_->map_point_cloud()->at(i);
        road_graph_vertex_descriptor v = boost::add_vertex(osm_graph_); 
        osm_graph_[v].pt.x = m_pt.x; 
        osm_graph_[v].pt.y = m_pt.y; 
    } 
    for (size_t i = 0; i < osmMap_->way_point_idxs().size(); ++i) { 
        vector<int>& a_way = osmMap_->way_point_idxs()[i];
        for(int j = 0; j < a_way.size() - 1; ++j){
            auto e = boost::add_edge(a_way[j], a_way[j+1], osm_graph_);  
            if(e.second){
                PclPoint& m_pt1 = osmMap_->map_point_cloud()->at(a_way[j]);
                PclPoint& m_pt2 = osmMap_->map_point_cloud()->at(a_way[j+1]);
                float dx = m_pt2.x - m_pt1.x;
                float dy = m_pt2.y - m_pt1.y;
                osm_graph_[e.first].length = sqrt(dx*dx + dy*dy);
            }
        }
    } 
    osm_map_valid_ = true;
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
    float MAX_DELTA_HEADING = 15.0f; // in degree
    float SEARCH_RADIUS = Parameters::getInstance().searchRadius();
    float SIGMA_W = Parameters::getInstance().roadSigmaW();
    // Extract peaks
    for (size_t i = 0; i < grid_points_->size(); ++i) {
        if(grid_votes_[i] < VOTE_THRESHOLD) 
            continue;
        PclPoint& g_pt = grid_points_->at(i);
        if(has_unexplained_gps_pts_){
            vector<int> rpt_k_indices;
            vector<float> rpt_k_dist_sqrs;
            road_point_search_tree_->radiusSearch(g_pt, 
                                                  SEARCH_RADIUS, 
                                                  rpt_k_indices, 
                                                  rpt_k_dist_sqrs);
            bool is_covered = false;
            for (size_t k = 0; k < rpt_k_indices.size(); ++k) { 
                PclPoint& nb_r_pt = road_points_->at(rpt_k_indices[k]);
                if(abs(deltaHeading1MinusHeading2(nb_r_pt.head, g_pt.head)) < MAX_DELTA_HEADING){
                    Eigen::Vector2d nb_r_pt_dir = headingTo2dVector(nb_r_pt.head);
                    Eigen::Vector2d vec(g_pt.x - nb_r_pt.x,
                                        g_pt.y - nb_r_pt.y);
                    float dv = vec.dot(nb_r_pt_dir);
                    float perp_dist = abs(rpt_k_dist_sqrs[k] - dv*dv);
                    if(perp_dist < 0.5f * LANE_WIDTH * nb_r_pt.t && abs(dv) < 5.0f){
                        is_covered = true;
                        break;
                    }
                }
            } 
            if(is_covered)
                continue;
        }
        Eigen::Vector2d dir = headingTo2dVector(g_pt.head);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        grid_point_search_tree_->radiusSearch(g_pt, 
                                              SEARCH_RADIUS, 
                                              k_indices, 
                                              k_dist_sqrs);
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
            float dv = vec.dot(dir);
            float perp_dist = sqrt(abs(k_dist_sqrs[j] - dv*dv));
            //float cross_value = Eigen::Vector3d(nb_g_pt.x - g_pt.x,
            //                              nb_g_pt.y - g_pt.y, 0.0f).cross(headingTo3dVector(g_pt.head))[2];
            //int bin_idx = floor(cross_value + 25.0f);
            //if(bin_idx < 0)
            //    bin_idx = 0;
            //if(bin_idx >= 50)
            //    bin_idx = 49;
            //for(int s = bin_idx - 2; s <= bin_idx + 2; ++s){
            //    if(s < 0)
            //        continue;
            //    if(s >= dist.size())
            //        continue;
            //    dist[s] += exp(-1.0f * (bin_idx - s) * (bin_idx - s) / 2.0f) * grid_votes_[k_indices[j]];
            //}
            if (abs(vec.dot(dir)) < 1.5f && perp_dist < SIGMA_W) {
                if (grid_votes_[k_indices[j]] > grid_votes_[i]) {
                    is_lateral_max = false;
                    break;
                }
            }
        }
        if (is_lateral_max) {
            grid_is_local_maximum_[i] = true;
            //xs.emplace_back(g_pt.x);
            //ys.emplace_back(g_pt.y);
            //lateral_distributions.emplace_back(dist);
        }
    }
    //ofstream output;
    //output.open("candidate_road_center_points.txt");
    //if (output.fail()) {
    //    return;
    //}
    //for(int i = 0; i < xs.size(); ++i){
    //    output << xs[i] << ", " << ys[i] << endl;
    //    for (size_t j = 0; j < lateral_distributions[i].size()-1; ++j) { 
    //        output << lateral_distributions[i][j] << ", ";
    //    } 
    //    output << lateral_distributions[i].back() << endl;
    //}
    //output.close();
}

void RoadGenerator::pointBasedVoting(){
    simplified_traj_points_->clear();
    if(!has_unexplained_gps_pts_){
        // Initial voting
        sampleGPSPoints(2.5f,
                        7.5f,
                        trajectories_->data(),
                        trajectories_->tree(),
                        simplified_traj_points_,
                        simplified_traj_point_search_tree_);
    }
    else{
        PclPointCloud::Ptr tmp_points(new PclPointCloud);
        PclSearchTree::Ptr tmp_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
        for (const auto& idx : unexplained_gps_pt_idxs_) { 
            tmp_points->push_back(trajectories_->data()->at(idx));
        } 
        if(tmp_points->size() == 0)
            return;
        tmp_point_search_tree->setInputCloud(tmp_points);
        sampleGPSPoints(2.5f,
                        7.5f,
                        tmp_points,
                        tmp_point_search_tree,
                        simplified_traj_points_,
                        simplified_traj_point_search_tree_);
    }
    float avg_speed = trajectories_->avgSpeed(); 
    // Start voting
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
    //cout << "Starting point based road voting: \n\tsigma_h= " << sigma_h <<
    //        "m, \tsigma_w= " << sigma_w <<
    //        "m, \tgrid_size= " << delta << "m" << endl;
    int half_search_window = floor(sigma_h / delta + 0.5f);
    int N_ANGLE_BINS = 36;
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
        if(pt.speed < trajectories_->avgSpeed() / 5.0f){
            continue;
        }
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
                float vote = pt.id_sample *
                             exp(-1.0f * dot_value_sqr / 2.0f / sigma_h / sigma_h) *
                             exp(-1.0f * perp_value_sqr / 2.0f / adjusted_sigma_w / adjusted_sigma_w);
                if (vote > 0.0f) {
                    for (int s = heading_bin_idx - 2; s <= heading_bin_idx + 2; ++s) {
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
                         6,
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

void RoadGenerator::estimateRoadWidthAndSpeed(vector<RoadPt>& road, float threshold){
    /*
     *threshold should be (0, 1.0)
     */
    if(road.size() < 2)
        return;
    float SEARCH_RADIUS = Parameters::getInstance().searchRadius();
    float delta_perp_bin = 2.5f; // in meters 
    int N_PERP_BIN = floor(2.0f * SEARCH_RADIUS / delta_perp_bin);
    delta_perp_bin = 2.0f * SEARCH_RADIUS / N_PERP_BIN;
    float cum_width = 0.0f;
    for (size_t i = 0; i < road.size(); ++i) { 
        PclPoint pt;
        pt.setCoordinate(road[i].x, road[i].y, 0.0f);
        Eigen::Vector3d pt_dir = headingTo3dVector(road[i].head);
        vector<float> perp_hist(N_PERP_BIN, 0.0f);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        trajectories_->tree()->radiusSearch(pt, SEARCH_RADIUS, k_indices, k_dist_sqrs);
        float avg_speed = 0.0f;
        int n_avg_pt = 0;
        for (size_t j = 0; j < k_indices.size(); ++j) { 
            PclPoint& nb_pt = trajectories_->data()->at(k_indices[j]);
            int dh = abs(deltaHeading1MinusHeading2(nb_pt.head, road[i].head));
            if(dh > 7.5f)
                continue;
            Eigen::Vector3d vec(nb_pt.x - pt.x,
                                nb_pt.y - pt.y,
                                0.0f); 
            float perp_dist = pt_dir.cross(vec)[2];
            float parallel_dist = abs(pt_dir.dot(vec));
            if(parallel_dist > 15.0f)
                continue;
            int bin_idx = floor(perp_dist + SEARCH_RADIUS) / delta_perp_bin;
            avg_speed += nb_pt.speed;
            n_avg_pt += 1;
            for(int k = bin_idx - 2; k <= bin_idx + 2; ++k){
                if(k < 0 || k >= N_PERP_BIN)
                    continue;
                float d = perp_dist + SEARCH_RADIUS - (k + 0.5f) * delta_perp_bin;
                perp_hist[k] += exp(-1.0f * d * d / 2.0f / delta_perp_bin / delta_perp_bin);
            }
        } 
        if(n_avg_pt > 0){
            avg_speed /= n_avg_pt;
            road[i].speed = avg_speed;
        }
        else{
            road[i].speed = 0;
        }
        int max_idx = -1;
        vector<int> peak_idxs;
        int window_size = floor(3.7f * 4 / delta_perp_bin);
        peakDetector(perp_hist,
                     window_size,
                     1.5f,
                     peak_idxs,
                     false);
        if(peak_idxs.size() > 0){
            float closest_dist = POSITIVE_INFINITY;
            for (const auto& idx : peak_idxs) { 
                float delta_dist = abs(idx * delta_perp_bin - SEARCH_RADIUS);
                if(delta_dist < closest_dist){
                    closest_dist = delta_dist;
                    max_idx = idx;
                } 
            } 
        } 
        if(max_idx != -1){
            // Update road width
            int left_border = max_idx;
            while (left_border >= 0) {
                if (perp_hist[left_border] < threshold * perp_hist[max_idx]) {
                    break;
                }
                left_border--;
            }
            if(left_border < 0){
                left_border = 0;
            }
            
            int right_border = max_idx;
            while (right_border < N_PERP_BIN) {
                if (perp_hist[right_border] < threshold * perp_hist[max_idx]) {
                    break;
                }
                right_border++;
            }
            if(right_border >= N_PERP_BIN){
                right_border = N_PERP_BIN-1;
            }
            
            float road_width = (right_border - left_border + 1) * LANE_WIDTH;
            if (road_width < LANE_WIDTH) {
                road_width = LANE_WIDTH;
            }
            cum_width += road_width;
        }
        else{
            cum_width += LANE_WIDTH;
        }
    } 
    int n_lanes = floor((cum_width / road.size() + 0.5f) / LANE_WIDTH);
    if(n_lanes < 1)
        n_lanes = 1;
    for (size_t i = 0; i < road.size(); ++i) { 
        road[i].n_lanes = n_lanes;
    } 
}

void RoadGenerator::adjustOppositeRoads(){
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    vector<vector<int>> road_idxs;
    vector<pair<int, float>> road_lengths;
    for(size_t i = 0; i < road_pieces_.size(); ++i){
        vector<RoadPt> a_road = road_pieces_[i];
        vector<int> r_idx;
        float cum_length = 0.0f;
        for (size_t j = 0; j < a_road.size(); ++j) { 
            PclPoint pt;
            pt.setCoordinate(a_road[j].x, a_road[j].y, 0.0f);
            pt.head = a_road[j].head;
            pt.id_trajectory = i;
            pt.id_sample = j;
            pt.t = a_road[j].n_lanes;
            pt.speed = a_road[j].speed;
            r_idx.emplace_back(points->size());
            points->push_back(pt);
            if(j > 0){
                float dx = a_road[j].x - a_road[j-1].x;
                float dy = a_road[j].y - a_road[j-1].y;
                cum_length += sqrt(dx*dx + dy*dy);
            }
        } 
        road_lengths.emplace_back(pair<int, float>(i, cum_length));
        road_idxs.emplace_back(r_idx);
    }
    sort(road_lengths.begin(), road_lengths.end(), pairCompareDescend);
    if(points->size() == 0)
        return;
    search_tree->setInputCloud(points);
    // Add edges for opposite directions
    vector<vector<int>> projections(road_lengths.size());
    for (size_t s = 0; s < road_lengths.size(); ++s) { 
        int road_idx = road_lengths[s].first;
        vector<int>& a_road = road_idxs[road_idx];
        vector<vector<pair<int, float>>> candidate_projections(a_road.size()); 
        for (size_t j = 0; j < a_road.size(); ++j) { 
            PclPoint& pt = points->at(a_road[j]);
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            search_tree->radiusSearch(pt, 25.0f, k_indices, k_dist_sqrs);
            Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
            map<int, pair<int,float>> mapped_oppo_road_idx;
            for (size_t k = 0; k < k_indices.size(); ++k) { 
                PclPoint& nb_pt = points->at(k_indices[k]);
                Eigen::Vector3d vec(nb_pt.x - pt.x,
                                    nb_pt.y - pt.y,
                                    0.0f);
                int dh = 180 - abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
                if(dh > 25)
                    continue;
                float perp_dist = pt_dir.cross(vec)[2];
                float parallel_dist = abs(pt_dir.dot(vec));
                if(abs(parallel_dist) > 15.0f)
                    continue;
                if(perp_dist < -10.0f || perp_dist > 20.0f)
                    continue;
                float score = sqrt(k_dist_sqrs[k]) * dh + 1;
                if(mapped_oppo_road_idx.find(nb_pt.id_trajectory) != mapped_oppo_road_idx.end()){
                    if(mapped_oppo_road_idx[nb_pt.id_trajectory].second > score){
                        mapped_oppo_road_idx[nb_pt.id_trajectory].second = score;
                        mapped_oppo_road_idx[nb_pt.id_trajectory].first = k_indices[k];
                    }
                }
                else{
                    mapped_oppo_road_idx[nb_pt.id_trajectory] = pair<int, float>(k_indices[k], score);
                }
                for (const auto& m : mapped_oppo_road_idx) { 
                    candidate_projections[j].push_back(m.second);
                } 
            }
            candidate_projections[j].push_back(pair<int, float>(-1, 1000.0f));
        } 
        // Dynamic programming to get min_score projection
        vector<vector<int>>   pre(candidate_projections.size(), vector<int>());
        vector<vector<float>> scores(candidate_projections.size(), vector<float>());
        // Initialize scores
        for (size_t i = 0; i < candidate_projections.size(); ++i) { 
            vector<float>& score = scores[i];
            vector<int>&   prei  = pre[i];
            vector<pair<int, float>>& candidate_projection = candidate_projections[i];
            score.resize(candidate_projection.size(), POSITIVE_INFINITY); 
            prei.resize(candidate_projection.size(), -1);
            if(i == 0){
                for (size_t j = 0; j < candidate_projection.size(); ++j) { 
                    score[j] = log10(candidate_projection[j].second); 
                }  
            } 
        } 
        for (size_t i = 1; i < candidate_projections.size() ; ++i) { 
            vector<pair<int, float>>& R = candidate_projections[i-1];
            vector<pair<int, float>>& L = candidate_projections[i];
            vector<int>&           prei = pre[i];
            vector<float>& scorer = scores[i-1];
            vector<float>& scorel = scores[i];
            for (size_t j = 0; j < L.size(); ++j) { 
                int cur_min_idx = -1;
                float cur_min = POSITIVE_INFINITY;
                for (size_t k = 0; k < R.size(); ++k) { 
                    float p_r_l = 10; 
                    if(R[k].first != -1 && L[j].first != -1){
                        if(points->at(R[k].first).id_trajectory == points->at(L[j].first).id_trajectory)
                            p_r_l = 1;
                    }
                    float s_r_p_r_l = scorer[k] + log10(p_r_l);
                    if(cur_min > s_r_p_r_l){
                        cur_min = s_r_p_r_l;
                        cur_min_idx = k;
                    } 
                } 
                scorel[j] = log10(L[j].second) + cur_min;   
                prei[j]   = cur_min_idx;
            } 
        }
        // Trace projection results
        // find min idx
        float last_min_score = POSITIVE_INFINITY;
        float last_min_idx = -1;
        vector<float>& last_score = scores.back(); 
        for (size_t i = 0; i < last_score.size(); ++i) { 
            if(last_min_score > last_score[i]) {
                last_min_score = last_score[i];
                last_min_idx = i;
            }
        } 
        if(last_min_idx != -1){
            vector<int>& projection = projections[road_idx];
            projection.resize(a_road.size(), -1);
            int last_idx = last_min_idx;
            projection[a_road.size()-1] = candidate_projections[a_road.size()-1][last_min_idx].first;
            for(int i = a_road.size()-1; i >= 1; --i){
                projection[i-1] = candidate_projections[i-1][pre[i][last_idx]].first;
                last_idx = pre[i][last_idx];
            }
        }
    } 
    // Move road points
    vector<vector<RoadPt>> new_roads;
    for (size_t s = 0; s < road_lengths.size(); ++s) { 
        int road_idx = road_lengths[s].first;
        vector<int>& a_road = road_idxs[road_idx];
        vector<int>& projection = projections[road_idx];
        vector<RoadPt> a_new_road;
        map<int, float> avg_perp_dist;
        map<int, int>   avg_perp_n_pt;
        int n_matched = 0;
        for (size_t j = 0; j < projection.size(); ++j) { 
            PclPoint& r_pt = points->at(a_road[j]);
            Eigen::Vector3d r_pt_dir = headingTo3dVector(r_pt.head);
            if(projection[j] != -1){
                PclPoint& nb_pt = points->at(projection[j]);
                Eigen::Vector3d vec(nb_pt.x - r_pt.x,
                                    nb_pt.y - r_pt.y,
                                    0.0f);
                float perp_dist = -1.0f * r_pt_dir.cross(vec)[2];
                if(avg_perp_n_pt.find(nb_pt.id_trajectory) != avg_perp_n_pt.end()){
                    avg_perp_n_pt[nb_pt.id_trajectory] ++;
                    avg_perp_dist[nb_pt.id_trajectory] += perp_dist;
                }
                else{
                    avg_perp_n_pt[nb_pt.id_trajectory] = 1;
                    avg_perp_dist[nb_pt.id_trajectory] = perp_dist;
                }
                n_matched++;
            }
        }
        for (auto& m : avg_perp_dist) { 
            m.second /= avg_perp_n_pt[m.first];
            float touching_perp_dist = -0.5f * (points->at(a_road[0]).t + points->at(road_idxs[m.first][0]).t) * LANE_WIDTH;
            if(m.second > touching_perp_dist - 2.0f * LANE_WIDTH){
                m.second = touching_perp_dist;
            }
        } 
        bool valid = true;
        if(n_matched < 0.5f * a_road.size())
            valid = false;

        map<int, vector<int>> mapped_points;
        for (size_t j = 0; j < projection.size(); ++j) { 
            PclPoint& r_pt = points->at(a_road[j]);
            Eigen::Vector3d new_r_pt_loc(r_pt.x, r_pt.y, 0.0f);
            Eigen::Vector3d r_pt_dir = headingTo3dVector(r_pt.head);
            Eigen::Vector3d r_pt_perp_dir(r_pt_dir[1], -r_pt_dir[0], 0.0f);
            if(projection[j] != -1){
                PclPoint& nb_pt = points->at(projection[j]);
                RoadPt new_pt;
                new_pt.x = r_pt.x;
                new_pt.y = r_pt.y;
                new_pt.head = r_pt.head;
                new_pt.n_lanes = r_pt.t;
                new_pt.speed = r_pt.speed;
                if(avg_perp_n_pt[nb_pt.id_trajectory] < 20 || !valid){
                    a_new_road.emplace_back(new_pt);
                    continue;
                }
                Eigen::Vector3d vec(nb_pt.x - r_pt.x,
                                    nb_pt.y - r_pt.y,
                                    0.0f);
                float perp_dist = -1.0f * r_pt_dir.cross(vec)[2];
                new_r_pt_loc += 0.5f * r_pt_perp_dir * (perp_dist + abs(avg_perp_dist[nb_pt.id_trajectory]));
                new_pt.x = new_r_pt_loc.x();
                new_pt.y = new_r_pt_loc.y();
                new_pt.head = r_pt.head;
                new_pt.n_lanes = r_pt.t;
                new_pt.speed = r_pt.speed;
                a_new_road.emplace_back(new_pt);
                if(mapped_points.find(projection[j]) != mapped_points.end()){
                    mapped_points[projection[j]] = vector<int>();
                    mapped_points[projection[j]].emplace_back(j);
                }
                else{
                    mapped_points[projection[j]] = vector<int>();
                    mapped_points[projection[j]].emplace_back(j);
                }
            }
            else{
                RoadPt new_pt;
                new_pt.x = r_pt.x;
                new_pt.y = r_pt.y;
                new_pt.head = r_pt.head;
                new_pt.n_lanes = r_pt.t;
                new_pt.speed = r_pt.speed;
                a_new_road.emplace_back(new_pt);
            }
        } 
        smoothCurve(a_new_road, false);
        uniformSampleCurve(a_new_road);
        new_roads.emplace_back(a_new_road);
        //continue;
        //// Correct points on other road
        //for (const auto& m : mapped_points) { 
        //    PclPoint& r_pt = points->at(m.first);
        //    Eigen::Vector2d avg_heading(0.0f, 0.0f);
        //    Eigen::Vector2d avg_loc(0.0f, 0.0f);
        //    for (const auto& idx : m.second) { 
        //        avg_heading -= headingTo2dVector(a_new_road[idx].head);
        //        avg_loc += Eigen::Vector2d(a_new_road[idx].x, a_new_road[idx].y);
        //    } 
        //    avg_loc /= m.second.size();
        //    int new_heading = vector2dToHeading(avg_heading);
        //    Eigen::Vector2d r_pt_dir = headingTo2dVector(new_heading);
        //    Eigen::Vector2d r_pt_perp_dir(r_pt_dir[1], -r_pt_dir[0]);
        //    Eigen::Vector2d vec = avg_loc - Eigen::Vector2d(r_pt.x, r_pt.y);
        //    float perp_dist = r_pt_dir.dot(vec);
        //    Eigen::Vector2d new_r_pt_loc(r_pt.x, r_pt.y);
        //    new_r_pt_loc += 0.5f * r_pt_perp_dir * (perp_dist + abs(avg_perp_dist[r_pt.id_trajectory]));
        //    r_pt.x = new_r_pt_loc.x();
        //    r_pt.y = new_r_pt_loc.y();
        //    r_pt.head = new_heading;
        //} 
    }
    road_pieces_.clear();
    for(size_t i = 0; i < new_roads.size(); ++i)
        road_pieces_.emplace_back(new_roads[i]);
}

bool RoadGenerator::computeInitialRoadGuess(){
    float MIN_ROAD_LENGTH = min_road_length_; 
    cout << "Current Min Length: " << min_road_length_ << endl;
    int max_iter = 20;
    road_pieces_.clear();
    int i_iter = 0;
    clock_t t_begin = clock();
    float SIGMA_W = Parameters::getInstance().roadSigmaW();
    float SEARCH_RADIUS = Parameters::getInstance().searchRadius();
    while(true){
        i_iter++;
        pointBasedVoting();
        // Extract peaks
        computeVoteLocalMaxima();
        // Extend existing roads to cover unexplained points.
        vector<vector<RoadPt>> previous_roads;
        set<int>               marked_grid_points;
        float EXTEND_ANGLE_THRESHOLD = 15.0f;
        bool has_update = false;
        if(road_pieces_.size() > 0){
            for (size_t i = 0; i < road_pieces_.size(); ++i) { 
                previous_roads.emplace_back(road_pieces_[i]);
            } 
            for (size_t i = 0; i < previous_roads.size(); ++i) { 
                vector<RoadPt>& p_road = previous_roads[i];
                PclPoint pt;
                // Front direction
                pt.setCoordinate(p_road[0].x, p_road[0].y, 0.0f);
                pt.head = p_road[0].head;
                while(true){
                    // Find the best candidate to connect
                    int best_bwd_candidate = -1;
                    float closest_bwd_distance = 1e6;
                    vector<int> k_indices;
                    vector<float> k_dist_sqrs;
                    grid_point_search_tree_->radiusSearch(pt, SEARCH_RADIUS, k_indices, k_dist_sqrs);
                    Eigen::Vector2d pt_dir = headingTo2dVector(pt.head);
                    for(size_t k = 0; k < k_indices.size(); ++k){
                        if(!grid_is_local_maximum_[k_indices[k]]){
                            continue;
                        }
                        if(marked_grid_points.find(k_indices[k]) != marked_grid_points.end())
                            continue;
                        PclPoint& nb_g_pt = grid_points_->at(k_indices[k]);
                        float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, pt.head));
                        if(delta_heading > EXTEND_ANGLE_THRESHOLD){
                            continue;
                        }
                        Eigen::Vector2d nb_g_pt_dir = headingTo2dVector(nb_g_pt.head);
                        Eigen::Vector2d vec(nb_g_pt.x - pt.x,
                                            nb_g_pt.y - pt.y);
                        float dot_value = vec.dot(pt_dir);
                        float perp_dist = sqrt(vec.dot(vec) - dot_value*dot_value);
                        if(perp_dist < SIGMA_W && dot_value > -5.0f)
                            marked_grid_points.emplace(k_indices[k]);
                        float vec_length = vec.norm();
                        if(vec_length < 1e-3)
                            continue;
                        dot_value /= vec_length;
                        float dh1 = abs(deltaHeading1MinusHeading2(pt.head, nb_g_pt.head));
                        float dot_head = vector2dToHeading(vec);
                        float dh2 = abs(deltaHeading1MinusHeading2(pt.head, dot_head));
                        // Update best_fwd_candidate
                        if (dot_value < -0.1 && perp_dist < SIGMA_W) {
                            float this_bwd_distance = (vec_length + 1.0f) * (dh1 + 1) * (dh2 + 1);
                            if(closest_bwd_distance > this_bwd_distance){
                                closest_bwd_distance = this_bwd_distance;
                                best_bwd_candidate = k_indices[k];
                            }
                        }
                    }
                    if(best_bwd_candidate == -1)
                        break;
                    else{
                        PclPoint& nb_g_pt = grid_points_->at(best_bwd_candidate);
                        RoadPt new_pt;
                        new_pt.x = nb_g_pt.x;
                        new_pt.y = nb_g_pt.y;
                        Eigen::Vector2d vec(0.0f, 0.0f);
                        for (size_t k = 0; k < 3; ++k) { 
                            vec += headingTo2dVector(p_road[k].head);
                        } 
                        vec += headingTo2dVector(nb_g_pt.head);
                        new_pt.head = vector2dToHeading(vec);
                        pt.setCoordinate(nb_g_pt.x, nb_g_pt.y, 0.0f);
                        pt.head = nb_g_pt.head;
                        p_road.emplace(p_road.begin(), new_pt);
                        has_update = true;
                    }
                }
                // Rear direction 
                pt.setCoordinate(p_road.back().x, p_road.back().y, 0.0f);
                pt.head = p_road.back().head;
                while(true){
                    // Find the best candidate to connect
                    int best_fwd_candidate = -1;
                    float closest_fwd_distance = 1e6;
                    vector<int> k_indices;
                    vector<float> k_dist_sqrs;
                    grid_point_search_tree_->radiusSearch(pt, SEARCH_RADIUS, k_indices, k_dist_sqrs);
                    Eigen::Vector2d pt_dir = headingTo2dVector(pt.head);
                    for(size_t k = 0; k < k_indices.size(); ++k){
                        if(!grid_is_local_maximum_[k_indices[k]]){
                            continue;
                        }
                        if(marked_grid_points.find(k_indices[k]) != marked_grid_points.end())
                            continue;
                        PclPoint& nb_g_pt = grid_points_->at(k_indices[k]);
                        float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, pt.head));
                        if(delta_heading > EXTEND_ANGLE_THRESHOLD){
                            continue;
                        }
                        Eigen::Vector2d nb_g_pt_dir = headingTo2dVector(nb_g_pt.head);
                        Eigen::Vector2d vec(nb_g_pt.x - pt.x,
                                            nb_g_pt.y - pt.y);
                        float dot_value = vec.dot(pt_dir);
                        float perp_dist = sqrt(vec.dot(vec) - dot_value*dot_value);
                        if(perp_dist < SIGMA_W && dot_value < 5.0f)
                            marked_grid_points.emplace(k_indices[k]);
                        float vec_length = vec.norm();
                        if(vec_length < 1e-3)
                            continue;
                        dot_value /= vec_length;
                        float dh1 = abs(deltaHeading1MinusHeading2(pt.head, nb_g_pt.head));
                        float dot_head = vector2dToHeading(vec);
                        float dh2 = abs(deltaHeading1MinusHeading2(pt.head, dot_head));
                        // Update best_fwd_candidate
                        if (dot_value > 0.1 && perp_dist < 10.0f) {
                            float this_fwd_distance = (vec_length + 1.0f) * (dh1 + 1) * (dh2 + 1);
                            if(closest_fwd_distance > this_fwd_distance){
                                closest_fwd_distance = this_fwd_distance;
                                best_fwd_candidate = k_indices[k];
                            }
                        }
                    }
                    if(best_fwd_candidate == -1)
                        break;
                    else{
                        PclPoint& nb_g_pt = grid_points_->at(best_fwd_candidate);
                        RoadPt new_pt;
                        new_pt.x = nb_g_pt.x;
                        new_pt.y = nb_g_pt.y;
                        Eigen::Vector2d vec(0.0f, 0.0f);
                        int n_p_road = p_road.size();
                        for (int k = n_p_road-3; k < p_road.size(); ++k) { 
                            vec += headingTo2dVector(p_road[k].head);
                        } 
                        vec += headingTo2dVector(nb_g_pt.head);
                        new_pt.head = vector2dToHeading(vec);
                        pt.setCoordinate(nb_g_pt.x, nb_g_pt.y, 0.0f);
                        pt.head = nb_g_pt.head;
                        p_road.emplace_back(new_pt);
                        has_update = true;
                    }
                }
            } 
            for (size_t i = 0; i < previous_roads.size(); ++i) { 
                vector<RoadPt>& a_road = previous_roads[i];
                smoothCurve(a_road, false);
                uniformSampleCurve(a_road);
                // Estimate road width
                estimateRoadWidthAndSpeed(a_road, 0.8f);
            } 
        }
        road_pieces_.clear();
        if(previous_roads.size() != 0){
            for (size_t i = 0; i < previous_roads.size(); ++i) { 
                road_pieces_.emplace_back(previous_roads[i]);
            } 
        }
        // Trace roads from grid_points_
        // vector<pair<grid_pt_idx, grid_pt_vote>>: grid_votes has all grid_points_ 
        vector<pair<int, float>> grid_votes;
        for(size_t i = 0; i < grid_votes_.size(); ++i){
            grid_votes.push_back(pair<int, float>(i, grid_votes_[i]));
        }
        sort(grid_votes.begin(), grid_votes.end(), pairCompareDescend);
        float STOPPING_THRESHOLD = Parameters::getInstance().roadVoteThreshold();
        typedef adjacency_list<vecS, vecS, undirectedS>    graph_t;
        typedef graph_traits<graph_t>::vertex_descriptor    vertex_descriptor;
        typedef graph_traits<graph_t>::edge_descriptor      edge_descriptor;
        graph_t G(grid_votes.size());
        float search_radius = 25.0f;
        vector<bool> grid_pt_visited(grid_votes.size(), false);
        for (int i = 0; i < grid_votes.size(); ++i) {
            int grid_pt_idx = grid_votes[i].first;
            if(marked_grid_points.find(grid_pt_idx) != marked_grid_points.end())
                continue;
            if(!grid_is_local_maximum_[grid_pt_idx]){
                continue;
            }
            if(grid_votes_[grid_pt_idx] < STOPPING_THRESHOLD) 
                break;
            grid_pt_visited[grid_pt_idx] = true;
            float grid_pt_vote = grid_votes[i].second;
            
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
                // Has to connect an already visited grid_point
                if(!grid_pt_visited[k_indices[k]]){
                    continue;
                }
                PclPoint& nb_g_pt = grid_points_->at(k_indices[k]);
                Eigen::Vector2d nb_g_pt_dir = headingTo2dVector(nb_g_pt.head);
                float delta_heading = abs(deltaHeading1MinusHeading2(nb_g_pt.head, g_pt.head));
                if(delta_heading > 20.0f){
                    continue;
                }
                Eigen::Vector2d vec(nb_g_pt.x - g_pt.x,
                                    nb_g_pt.y - g_pt.y);
                float dot_value = vec.dot(g_pt_dir);
                float perp_dist = sqrt(vec.dot(vec) - dot_value*dot_value);
                float vec_length = vec.norm();
                dot_value /= vec_length;
                float dh1 = abs(deltaHeading1MinusHeading2(g_pt.head, nb_g_pt.head));
                float dot_head = vector2dToHeading(vec);
                float dh2 = abs(deltaHeading1MinusHeading2(g_pt.head, dot_head));
                //if(abs(dot_value) < 0.25f)
                //    continue;
                // Update best_fwd_candidate
                if (dot_value > 0.1 && perp_dist < SIGMA_W) {
                    //float this_fwd_distance = vec_length * (2.0f - g_pt_dir.dot(nb_g_pt_dir)) * (2.0f - dot_value);
                    float this_fwd_distance = (vec_length + 1.0f) * (dh1 + 1) * (dh2 + 1);
                    int n_degree = out_degree(k_indices[k], G);
                    if(n_degree <= 1){
                        this_fwd_distance /= (1.0f + 2.0f * n_degree);
                        if(closest_fwd_distance > this_fwd_distance){
                            closest_fwd_distance = this_fwd_distance;
                            best_fwd_candidate = k_indices[k];
                        }
                    }
                }
                // Update best_bwd_candidate
                if(dot_value < -0.1 && perp_dist < SIGMA_W){
                    //float this_bwd_distance = vec_length * (2.0f - g_pt_dir.dot(nb_g_pt_dir)) * (2.0f - dot_value);
                    float this_bwd_distance = (vec_length + 1.0f) * (dh1 + 1) * (dh2 + 1);
                    int n_degree = out_degree(k_indices[k], G);
                    if(n_degree <= 1){
                        this_bwd_distance /= (1.0f + 2.0f * n_degree);
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
        //Visualize graph
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
        vector<vector<RoadPt>> raw_roads;
        vector<pair<int, float>> raw_road_lengths;
        for (size_t i = 0; i < cluster_scores.size(); ++i) {
            vector<int>& cluster = clusters[cluster_scores[i].first];
            float cluster_score = cluster_scores[i].second;
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
            for(int j = 0; j < sorted_cluster.size(); ++j){
                PclPoint& pt = grid_points_->at(sorted_cluster[j]);
                RoadPt r_pt(pt.x,
                            pt.y,
                            pt.head);
                if(sorted_cluster[j] < grid_votes.size()){
                    r_pt.n_lanes = 0; 
                }
                else{
                    r_pt.n_lanes = pt.t;
                }
                a_road.push_back(r_pt);
            }
            smoothCurve(a_road, false);
            // Estimate road width
            estimateRoadWidthAndSpeed(a_road, 0.8f); 
            // Check road length
            float cum_length = 0.0f;
            for(size_t j = 1; j < a_road.size(); ++j){
                float delta_x = a_road[j].x - a_road[j-1].x;
                float delta_y = a_road[j].y - a_road[j-1].y;
                cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            if(cum_length >= MIN_ROAD_LENGTH){
                raw_road_lengths.emplace_back(pair<int, float>(raw_roads.size(), cum_length));
                raw_roads.push_back(a_road);
            }
        }
        // Remove overlapping roads
        PclPointCloud::Ptr tmp_points(new PclPointCloud);
        PclSearchTree::Ptr tmp_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
        vector<vector<int>> raw_road_idxs;
        for(size_t i = 0; i < raw_roads.size(); ++i){
            vector<RoadPt> a_road = raw_roads[i];
            vector<int>    a_road_idx;
            for (size_t j = 0; j < a_road.size(); ++j) { 
                PclPoint pt;
                pt.setCoordinate(a_road[j].x, a_road[j].y, 0.0f);
                pt.head = a_road[j].head;
                pt.id_trajectory = i;
                a_road_idx.emplace_back(tmp_points->size());
                tmp_points->push_back(pt);
            } 
            raw_road_idxs.emplace_back(a_road_idx);
        }
        if(tmp_points->size() > 0){
            has_update = true;
            tmp_point_search_tree->setInputCloud(tmp_points);
            sort(raw_road_lengths.begin(), raw_road_lengths.end(), pairCompareDescend);
            vector<bool> covered_by_others(tmp_points->size(), false);
            float cover_radius = 10.0f;
            float cover_heading_threshold = 15.0f;
            if(previous_roads.size() > 0){
                // Cover points with previous roads
                PclPoint pt;
                for (size_t i = 0; i < previous_roads.size(); ++i) { 
                    vector<RoadPt>& a_road = previous_roads[i];
                    for (size_t j = 0; j < a_road.size(); ++j) { 
                        pt.setCoordinate(a_road[j].x, a_road[j].y, 0.0f);
                        pt.head = a_road[j].head;
                        vector<int> k_indices;
                        vector<float> k_dist_sqrs;
                        tmp_point_search_tree->radiusSearch(pt, cover_radius, k_indices, k_dist_sqrs);
                        Eigen::Vector2d pt_dir = headingTo2dVector(pt.head);
                        Eigen::Vector2d pt_perp_dir(-pt_dir[1], pt_dir[0]);
                        for (size_t k = 0; k < k_indices.size(); ++k) { 
                            PclPoint& nb_pt = tmp_points->at(k_indices[k]);
                            int dh = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
                            if(dh < cover_heading_threshold){
                                Eigen::Vector2d vec(nb_pt.x - pt.x,
                                                    nb_pt.y - pt.y);
                                float perp_dist = abs(vec.dot(pt_perp_dir));
                                if(perp_dist < 0.5f * cover_radius)
                                    covered_by_others[k_indices[k]] = true;
                            }
                        }
                    } 
                } 
            }
            vector<vector<RoadPt>> roads;
            for (size_t i = 0; i < raw_road_lengths.size(); ++i) { 
                int raw_road_idx = raw_road_lengths[i].first;
                vector<RoadPt>& a_road = raw_roads[raw_road_idx];
                vector<bool> covered(a_road.size(), false);
                for (size_t j = 0; j < a_road.size(); ++j) { 
                    if(covered_by_others[raw_road_idxs[raw_road_idx][j]])
                        covered[j] = true;
                } 
                // Remove noise
                for(size_t j = 0; j < a_road.size(); ++j){
                    if(covered[j]){
                        bool is_noise = true;
                        if(j > 0){
                            if(covered[j-1])
                                is_noise = false;
                        }
                        if(j < a_road.size()-1){
                            if(covered[j+1])
                                is_noise = false;
                        }
                        if(is_noise)
                            covered[j] = false;
                    }
                }
                // Get sub road segments 
                int start_idx = 0;
                int end_idx = 0;
                while(start_idx < a_road.size()){
                    if(!covered[start_idx]){
                        end_idx = start_idx + 1;
                        float cum_length = 0.0f;
                        while(end_idx < a_road.size()){
                            if(covered[end_idx])
                                break;
                            float dx = a_road[end_idx].x - a_road[end_idx-1].x;
                            float dy = a_road[end_idx].y - a_road[end_idx-1].y;
                            cum_length += sqrt(dx*dx + dy*dy);
                            end_idx++;
                        }
                        if(cum_length > MIN_ROAD_LENGTH){
                            vector<RoadPt> road_seg;
                            for(int s = start_idx; s < end_idx; ++s){
                                road_seg.emplace_back(a_road[s]);
                                if(covered_by_others[raw_road_idxs[raw_road_idx][s]])
                                    continue;
                                PclPoint& pt = tmp_points->at(raw_road_idxs[raw_road_idx][s]);
                                Eigen::Vector2d pt_dir = headingTo2dVector(pt.head);
                                Eigen::Vector2d pt_perp_dir(-pt_dir[1], pt_dir[0]);
                                covered_by_others[raw_road_idxs[raw_road_idx][s]] = true;
                                vector<int> k_indices;
                                vector<float> k_dist_sqrs;
                                tmp_point_search_tree->radiusSearch(pt, cover_radius, k_indices, k_dist_sqrs);
                                for (size_t k = 0; k < k_indices.size(); ++k) { 
                                    PclPoint& nb_pt = tmp_points->at(k_indices[k]);
                                    if(nb_pt.id_trajectory == pt.id_trajectory)
                                        continue;
                                    int dh = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
                                    if(dh < cover_heading_threshold){
                                        Eigen::Vector2d vec(nb_pt.x - pt.x,
                                                            nb_pt.y - pt.y);
                                        float perp_dist = abs(vec.dot(pt_perp_dir));
                                        if(perp_dist < 0.5f * cover_radius)
                                            covered_by_others[k_indices[k]] = true;
                                    }
                                }
                            }
                            roads.emplace_back(road_seg);
                        }
                        start_idx = end_idx;
                    }
                    start_idx++;
                }
            } 
            for(size_t i = 0; i < roads.size(); ++i){
                // Smooth road width
                vector<RoadPt> a_road = roads[i];
                int cum_n_lanes = 0;
                float cum_length = 0.0f;
                for (size_t j = 0; j < a_road.size(); ++j) {
                    cum_n_lanes += a_road[j].n_lanes;
                    if(j > 0){
                        float dx = a_road[j].x - a_road[j-1].x;
                        float dy = a_road[j].y - a_road[j-1].y;
                        cum_length += sqrt(dx*dx + dy*dy);
                    }
                }
                cum_n_lanes /= a_road.size();
                if(cum_n_lanes > 5){
                    cum_n_lanes = 5;
                }
                for (size_t j = 0; j < a_road.size(); ++j) { 
                    a_road[j].n_lanes = cum_n_lanes;
                } 
                uniformSampleCurve(a_road);
                road_pieces_.push_back(a_road);
            }
            // Merge roads in road_pieces_
            tmp_points->clear();
            vector<vector<RoadPt>> merged_roads;
            vector<bool>           merged_road_valid;
            for (size_t i = 0; i < road_pieces_.size(); ++i) { 
                vector<RoadPt> i_road = road_pieces_[i];
                merged_roads.emplace_back(i_road);
                merged_road_valid.push_back(true);
                for (size_t j = 0; j < i_road.size(); ++j) { 
                    PclPoint pt;
                    RoadPt& r_pt = i_road[j];
                    pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                    pt.head = r_pt.head;
                    pt.t    = r_pt.n_lanes;
                    pt.speed = r_pt.speed;
                    pt.id_trajectory = i;
                    pt.id_sample = j;
                    tmp_points->push_back(pt);
                } 
            } 
            if(tmp_points->size() > 0){
                tmp_point_search_tree->setInputCloud(tmp_points);
            }
            // Start merging
            for (size_t i = 0; i < road_pieces_.size(); ++i) { 
                if(!merged_road_valid[i])
                    continue;
                set<int> road_idxs_to_merge;
                road_idxs_to_merge.emplace(i);
                vector<int> list_road_idxs_to_merge;
                list_road_idxs_to_merge.emplace_back(i);
                // Extend in front direction
                int cur_front_road_idx = i;
                while(true){
                    vector<RoadPt>& a_road = merged_roads[cur_front_road_idx];
                    RoadPt& front_r_pt = a_road.front();
                    Eigen::Vector2d front_r_pt_dir = headingTo2dVector(front_r_pt.head);
                    Eigen::Vector2d front_r_pt_perp_dir(-front_r_pt_dir[1], front_r_pt_dir[0]);
                    PclPoint pt;
                    pt.setCoordinate(front_r_pt.x, front_r_pt.y, 0.0f);
                    vector<int> k_indices;
                    vector<float> k_dist_sqrs;
                    tmp_point_search_tree->radiusSearch(pt, 45.0f, k_indices, k_dist_sqrs);
                    int front_merge_road_idx = -1;
                    for (size_t k = 0; k < k_indices.size(); ++k) { 
                        PclPoint& nb_pt = tmp_points->at(k_indices[k]);
                        if(!merged_road_valid[nb_pt.id_trajectory]){
                            continue;
                        }
                        if(road_idxs_to_merge.find(nb_pt.id_trajectory) != road_idxs_to_merge.end()){
                            continue;
                        }
                        float dh = abs(deltaHeading1MinusHeading2(front_r_pt.head, nb_pt.head));
                        if(dh > 15.0f)
                            continue;
                        Eigen::Vector2d vec(nb_pt.x - front_r_pt.x,
                                            nb_pt.y - front_r_pt.y);
                        float perp_dist = abs(front_r_pt_perp_dir.dot(vec));
                        if(perp_dist > 10.0f)
                            continue;
                        bool source_near_end = false;
                        float cum_length = 0.0f;
                        for (size_t k = nb_pt.id_sample + 1; k < merged_roads[nb_pt.id_trajectory].size(); ++k) { 
                            RoadPt& r_pt1 = merged_roads[nb_pt.id_trajectory][k];
                            RoadPt& r_pt2 = merged_roads[nb_pt.id_trajectory][k-1];
                            float dx = r_pt1.x - r_pt2.x;
                            float dy = r_pt1.y - r_pt2.y;
                            cum_length += sqrt(dx*dx + dy*dy);
                        }
                        if(cum_length < 25.0f){
                            source_near_end = true;
                            front_merge_road_idx = nb_pt.id_trajectory;
                            break;
                        }
                    }
                    if(front_merge_road_idx != -1){
                        list_road_idxs_to_merge.insert(list_road_idxs_to_merge.begin(), front_merge_road_idx);
                        road_idxs_to_merge.emplace(front_merge_road_idx);
                        cur_front_road_idx = front_merge_road_idx;
                    }
                    else{
                        break;
                    }
                }
                // Extend in rear direction
                int cur_rear_road_idx = i;
                while(true){
                    vector<RoadPt>& a_road = merged_roads[cur_rear_road_idx];
                    RoadPt& rear_r_pt = a_road.back();
                    Eigen::Vector2d rear_r_pt_dir = headingTo2dVector(rear_r_pt.head);
                    Eigen::Vector2d rear_r_pt_perp_dir(-rear_r_pt_dir[1], rear_r_pt_dir[0]);
                    PclPoint pt;
                    pt.setCoordinate(rear_r_pt.x, rear_r_pt.y, 0.0f);
                    vector<int> k_indices;
                    vector<float> k_dist_sqrs;
                    tmp_point_search_tree->radiusSearch(pt, 45.0f, k_indices, k_dist_sqrs);
                    int rear_merge_road_idx = -1;
                    for (size_t k = 0; k < k_indices.size(); ++k) { 
                        PclPoint& nb_pt = tmp_points->at(k_indices[k]);
                        if(!merged_road_valid[nb_pt.id_trajectory]){
                            continue;
                        }
                        if(road_idxs_to_merge.find(nb_pt.id_trajectory) != road_idxs_to_merge.end()){
                            continue;
                        }
                        float dh = abs(deltaHeading1MinusHeading2(rear_r_pt.head, nb_pt.head));
                        if(dh > 15.0f)
                            continue;
                        Eigen::Vector2d vec(nb_pt.x - rear_r_pt.x,
                                            nb_pt.y - rear_r_pt.y);
                        float perp_dist = abs(rear_r_pt_perp_dir.dot(vec));
                        if(perp_dist > 10.0f)
                            continue;
                        bool target_near_front = false;
                        float cum_length = 0.0f;
                        for (size_t k = 1; k <= nb_pt.id_sample; ++k) { 
                            RoadPt& r_pt1 = merged_roads[nb_pt.id_trajectory][k];
                            RoadPt& r_pt2 = merged_roads[nb_pt.id_trajectory][k-1];
                            float dx = r_pt1.x - r_pt2.x;
                            float dy = r_pt1.y - r_pt2.y;
                            cum_length += sqrt(dx*dx + dy*dy);
                        }
                        if(cum_length < 25.0f){
                            target_near_front = true;
                            rear_merge_road_idx = nb_pt.id_trajectory;
                            break;
                        }
                    }
                    if(rear_merge_road_idx != -1){
                        list_road_idxs_to_merge.emplace_back(rear_merge_road_idx);
                        road_idxs_to_merge.emplace(rear_merge_road_idx);
                        cur_rear_road_idx = rear_merge_road_idx;
                    }
                    else{
                        break;
                    }
                }
                // Actual merging
                if(list_road_idxs_to_merge.size() <= 1)
                    continue;
                vector<RoadPt> new_road;
                for (size_t j = 0; j < list_road_idxs_to_merge.size(); ++j) { 
                    int cur_road_idx = list_road_idxs_to_merge[j];
                    vector<RoadPt>& a_road = merged_roads[cur_road_idx];
                    merged_road_valid[cur_road_idx] = false;
                    int start_recording_idx = 0;
                    if(new_road.size() > 0){
                        RoadPt& cur_rear_pt = new_road.back();
                        Eigen::Vector2d rear_dir = headingTo2dVector(cur_rear_pt.head);
                        for (size_t k = 0; k < a_road.size(); ++k) { 
                            RoadPt& r_pt = a_road[k];
                            Eigen::Vector2d vec(r_pt.x - cur_rear_pt.x,
                                                r_pt.y - cur_rear_pt.y);
                            if(rear_dir.dot(vec) > 5.0f){
                                start_recording_idx = k;
                                break;
                            }
                        }
                    }
                    for (size_t j = start_recording_idx; j < a_road.size(); ++j) { 
                        new_road.emplace_back(a_road[j]);
                    } 
                } 
                smoothCurve(new_road, false);
                // Estimate road width
                estimateRoadWidthAndSpeed(new_road, 0.8f); 
                for (size_t j = 0; j < new_road.size(); ++j) { 
                    PclPoint pt;
                    RoadPt& r_pt = new_road[j];
                    pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                    pt.head = r_pt.head;
                    pt.t    = r_pt.n_lanes;
                    pt.speed = r_pt.speed;
                    pt.id_trajectory = merged_roads.size();
                    pt.id_sample = j;
                    tmp_points->push_back(pt);
                } 
                merged_roads.emplace_back(new_road);
                merged_road_valid.push_back(true);
                tmp_point_search_tree->setInputCloud(tmp_points);
            } 
            road_pieces_.clear();
            for (size_t i = 0; i < merged_roads.size(); ++i) { 
                if(merged_road_valid[i]){
                    road_pieces_.emplace_back(merged_roads[i]);
                }
            }
            //Recompute road_graph_.
            road_graph_.clear();
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
            if(!updateRoadPointCloud()){
                cout << "Failed to update road_points_." << endl;
                return false;
            }
            computeUnexplainedGPSPoints();
            // Update road_points_
            //road_points_->clear();
            //for (size_t i = 0; i < road_pieces_.size(); ++i) { 
            //    vector<RoadPt>& a_road = road_pieces_[i];
            //    for (size_t j = 0; j < a_road.size(); ++j) { 
            //        RoadPt& r_pt = a_road[j];
            //        PclPoint pt;
            //        pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
            //        pt.head = r_pt.head;
            //        pt.id_trajectory = i;
            //        pt.t             = r_pt.n_lanes;
            //        pt.speed         = r_pt.speed;
            //        pt.id_sample = j;
            //        road_points_->push_back(pt);
            //    } 
            //} 
            //if(road_points_->size() > 0){
            //    road_point_search_tree_->setInputCloud(road_points_); 
            //    computeUnexplainedGPSPoints();
            //} 
            //else{ 
            //    break;
            //}
        }
        if(!has_update)
            break;
        if(i_iter >= max_iter)
            break;
    }
    // Remove noisy roads
    vector<pair<int, float>> road_piece_lengths;
    float min_threshold = 1000.0f;
    for (size_t i = 0; i < road_pieces_.size(); ++i) { 
        float cum_length = 0.0f;
        vector<RoadPt>& a_road = road_pieces_[i];
        for (size_t j = 1; j < a_road.size(); ++j) { 
            cum_length += roadPtDistance(a_road[j], a_road[j-1]);
        } 
        road_piece_lengths.emplace_back(pair<int, float>(i, cum_length));
    } 
    sort(road_piece_lengths.begin(), road_piece_lengths.end(), pairCompare);
    vector<bool> road_piece_valid(road_pieces_.size(), true);
    for (size_t i = 0; i < road_piece_lengths.size(); ++i) { 
        float search_radius = 50.0f;
        if(road_piece_lengths[i].second > min_threshold){
            search_radius = 15.0f;
        }
        bool to_remove = false;
        int road_idx = road_piece_lengths[i].first;
        vector<RoadPt>& a_road = road_pieces_[road_idx];
        int n_covered_pt = 0;
        for (size_t j = 0; j < a_road.size(); ++j) { 
            RoadPt& r_pt = a_road[j];
            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            PclPoint pt;
            pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
            vector<int> k_indices;
            vector<float> k_dist_sqrs;
            road_point_search_tree_->radiusSearch(pt, search_radius, k_indices, k_dist_sqrs);
            for (size_t k = 0; k < k_indices.size(); ++k) { 
                PclPoint& nb_pt = road_points_->at(k_indices[k]);
                if(nb_pt.id_trajectory == road_piece_lengths[i].first)
                    continue;
                if(!road_piece_valid[nb_pt.id_trajectory])
                    continue;
                float dh = abs(deltaHeading1MinusHeading2(nb_pt.head, r_pt.head));
                if(dh > 7.5f)
                    continue;
                Eigen::Vector2d vec(nb_pt.x - r_pt.x,
                                    nb_pt.y - r_pt.y);
                if(abs(vec.dot(r_pt_dir)) < 10.0f){
                    n_covered_pt++;
                    break;
                }
            } 
        } 
        int point_left = a_road.size() - n_covered_pt;
        if(n_covered_pt > 0.8f * a_road.size() || point_left < 3)
            road_piece_valid[road_piece_lengths[i].first] = false;
    } 
    vector<vector<RoadPt>> clean_roads;
    for (size_t i = 0; i < road_pieces_.size(); ++i) { 
        if(road_piece_valid[i]){
            clean_roads.emplace_back(road_pieces_[i]);
        }
    } 
    road_pieces_.clear();
    road_pieces_ = std::move(clean_roads);
    // Adjust parallel opposite roads
    adjustOppositeRoads();
    //Recompute road_graph_.
    road_graph_.clear();
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
    if(!updateRoadPointCloud()){
        cout << "Failed to update road_points_." << endl;
        return false;
    }
    clock_t t_end = clock();
    double elapsed_secs = double(t_end - t_begin) / CLOCKS_PER_SEC;
    printf("Total iterations: %d. Time elapsed: %.2fs\n", i_iter, elapsed_secs);
    printf("\t%lu initial road pieces.\n", road_pieces_.size());
    // Visualize road_pieces_
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
    for(size_t i = 0; i < road_pieces_.size(); ++i){
        // Smooth road width
        vector<RoadPt> a_road = road_pieces_[i];
        int cum_n_lanes = a_road[0].n_lanes;
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
        float cv = 1.0f - static_cast<float>(i) / road_pieces_.size(); 
        Color c = ColorMap::getInstance().getJetColor(cv);   
        for (size_t j = 0; j < xs.size() - 1; ++j) {
            float color_ratio = 1.0f - 0.5f * color_value[j]; 
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j], ys[j], Z_ROAD));
            line_colors_.push_back(Color(color_ratio * c.r, color_ratio * c.g, color_ratio * c.b, 1.0f));
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j+1], ys[j+1], Z_ROAD));
            line_colors_.push_back(Color(color_ratio * c.r, color_ratio * c.g, color_ratio * c.b, 1.0f));
        }
    }
    return true;
}

void RoadGenerator::computeUnexplainedGPSPoints(){
    has_unexplained_gps_pts_ = false;
    unexplained_gps_pt_idxs_.clear();
    float CUT_OFF_PROBABILITY = 0.2f; // this is only for projection
    float SIGMA_W = Parameters::getInstance().roadSigmaW();
    float SIGMA_L = Parameters::getInstance().roadSigmaH();
    float SIGMA_H = Parameters::getInstance().gpsMaxHeadingError();
    float SEARCH_RADIUS = Parameters::getInstance().searchRadius();
    for (size_t i = 0; i < trajectories_->data()->size(); ++i) { 
        PclPoint& pt = trajectories_->data()->at(i);
        // Search nearby road_points_
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        road_point_search_tree_->radiusSearch(pt,
                                              SEARCH_RADIUS,
                                              k_indices,
                                              k_dist_sqrs); 
        bool is_covered = false;
        for (size_t j = 0; j < k_indices.size() ; ++j) { 
            PclPoint& r_pt = road_points_->at(k_indices[j]); 
            float prob = probOnRoad(pt, r_pt, SIGMA_W, 0.5f * SIGMA_L, SIGMA_H);
            if(prob < CUT_OFF_PROBABILITY) 
                continue;
            if(prob > CUT_OFF_PROBABILITY){
                is_covered = true;
                break;
            }
        }
        if(!is_covered){
            unexplained_gps_pt_idxs_.emplace_back(i);
        }
    } 
    //for (size_t i = 0; i < trajectories_->trajectories().size(); ++i){ 
    //    vector<int> projection;
    //    partialMapMatching(i, 25.0f, 0.01f, projection); 
    //    if(projection.size() < 2) 
    //        continue;
    //    const vector<int>& traj = trajectories_->trajectories()[i]; 
    //    int last_projected_idx = -1;
    //    for (size_t j = 0; j < projection.size(); ++j) { 
    //        if(projection[j] == -1) {
    //            continue;
    //        } 
    //        else{
    //            if(last_projected_idx == -1){
    //                last_projected_idx = j;
    //                for(int s = 0; s < j; ++s){
    //                    unexplained_gps_pt_idxs_.emplace_back(traj[s]);
    //                }
    //                continue;
    //            }
    //            else{
    //                // Check if there is shortestPath
    //                float dist = POSITIVE_INFINITY;
    //                if(road_points_->at(projection[last_projected_idx]).id_trajectory == road_points_->at(projection[j]).id_trajectory){
    //                    dist = 0;
    //                }
    //                if(dist > 1e6){
    //                    for(int s = last_projected_idx+1; s < j; ++s)
    //                        unexplained_gps_pt_idxs_.emplace_back(traj[s]);
    //                }
    //                else{
    //                    for(int s = last_projected_idx+1; s < j; ++s)
    //                        unexplained_gps_pt_idxs_.emplace_back(traj[s]);
    //                }
    //                last_projected_idx = j;
    //            }
    //        }
    //    } 
    //    if(last_projected_idx < projection.size() && last_projected_idx > 0){
    //        for(int s = last_projected_idx+1; s < projection.size(); ++s)
    //            unexplained_gps_pt_idxs_.emplace_back(traj[s]);
    //    }
    //}
    if(unexplained_gps_pt_idxs_.size() > 0)
        has_unexplained_gps_pts_ = true;
    // Visualization
    points_to_draw_.clear();
    point_colors_.clear();
    for (const auto& idx : unexplained_gps_pt_idxs_) { 
        PclPoint& pt = trajectories_->data()->at(idx);
        points_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
        point_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE));
    } 
}

void RoadGenerator::tmpFunc(){
    // ############ 
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
    adjustOppositeRoads(); 
    // Visualize road_pieces_
    for(size_t i = 0; i < road_pieces_.size(); ++i){
        // Smooth road width
        vector<RoadPt> a_road = road_pieces_[i];
        int cum_n_lanes = a_road[0].n_lanes;
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
        float cv = 1.0f - static_cast<float>(i) / road_pieces_.size(); 
        Color c = ColorMap::getInstance().getJetColor(cv);   
        for (size_t j = 0; j < xs.size() - 1; ++j) {
            float color_ratio = 1.0f - 0.5f * color_value[j]; 
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j], ys[j], Z_ROAD));
            line_colors_.push_back(Color(color_ratio * c.r, color_ratio * c.g, color_ratio * c.b, 1.0f));
            lines_to_draw_.push_back(SceneConst::getInstance().normalize(xs[j+1], ys[j+1], Z_ROAD));
            line_colors_.push_back(Color(color_ratio * c.r, color_ratio * c.g, color_ratio * c.b, 1.0f));
        }
    }
    //vector<int> unexplained_pts;
    //for (int i = 0; i < trajectories_->trajectories().size(); ++i){ 
    //    vector<int> projection;
    //    partialMapMatching(i, projection); 

    //    if(projection.size() < 2) 
    //        continue;

    //    const vector<int>& traj = trajectories_->trajectories()[i];
    //    if(projection[0] == -1) 
    //        unexplained_pts.emplace_back(traj[0]); 

    //    for (size_t j = 1; j < projection.size(); ++j) { 
    //        if(projection[j] == -1){
    //            unexplained_pts.emplace_back(traj[j]); 
    //            continue; 
    //        }
    //    }
    //}

    //for (size_t i = 0; i < unexplained_pts.size(); ++i) { 
    //    PclPoint& pt = trajectories_->data()->at(unexplained_pts[i]);
    //    points_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_TRAJECTORIES + 0.01f));
    //    point_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE));
    //} 

    //float explained_pt_percent = 100 - static_cast<float>(unexplained_pts.size()) / trajectories_->data()->size() * 100.0f;
    //cout << explained_pt_percent << "%% data point explained." << endl;
    //
    // ############ 
    // Remove floating roads
    //for (size_t i = 0; i < indexed_roads_.size(); ++i) { 
    //}   
    // ############ 
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
        pt.speed         = r_pt.speed;
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
        //cout << "num of points: " << road_points_->size() << endl;
    } 
    else{
        cout << "WARNING from addInitialRoad: road_points is an empty point cloud!" << endl;
        return false;
    }

    return true;
}

bool RoadGenerator::addInitialRoad(){
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
    //Recompute road_graph_.
    //auto es = edges(road_graph_);
    //vector<road_graph_edge_descriptor> edges_to_remove;
    //for(auto eit = es.first; eit != es.second; ++eit){
    //    if(road_graph_[*eit].type == RoadGraphEdgeType::linking)
    //        edges_to_remove.emplace_back(*eit);
    //}
    //for (size_t i = 0; i < edges_to_remove.size(); ++i) { 
    //    remove_edge(edges_to_remove[i], road_graph_);
    //} 

    //vector<bool> point_explained(trajectories_->data()->size(), false);
    map<pair<int, int>, pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>> additional_edges;
    map<pair<int, int>, int>   additional_edge_support;
    // Partial map matching each trajectory
    int max_extension = 20; // look back and look forward
    for (size_t i = 0; i < trajectories_->trajectories().size(); ++i){ 
        vector<int> projection;
        partialMapMatching(i, 35.0f, 0.1f, projection); 
        if(projection.size() < 2) 
            continue;
        const vector<int>& traj = trajectories_->trajectories()[i]; 
        // Add edges
        int prev_road_pt_idx = projection[0];
        float prev_road_pt_t   = trajectories_->data()->at(traj[0]).t;
        for (size_t j = 1; j < projection.size(); ++j) { 
            if(projection[j] == -1){
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
                    // Check if they are already connected
                    vector<road_graph_vertex_descriptor>& first_road = indexed_roads_[first_road_idx];
                    vector<road_graph_vertex_descriptor>& second_road = indexed_roads_[second_road_idx];
                    vector<road_graph_vertex_descriptor> path;
                    //float dist = shortestPath(first_road.front(), 
                    //                          second_road.back(), 
                    //                          road_graph_,
                    //                          path);
                    //if(dist < 500)
                    //    continue;
                    if(additional_edges.find(pair<int, int>(first_road_idx, second_road_idx)) == additional_edges.end()){
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
                        if(min_dist < 75.0f){
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
    road_graph_valid_ = true;
    if(!debug_mode_){
        connectRoads();
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
    for(auto vit = vs.first; vit != vs.second; ++vit){
        if(!road_graph_[*vit].is_valid){
            continue;
        }
        road_graph_[*vit].cluster_id = -1;
        if(vertex_visited[*vit])
            continue;
        vertex_visited[*vit] = true;
        vector<road_graph_vertex_descriptor> a_road;
        int old_road_label = road_graph_[*vit].road_label;
        int new_road_label = max_road_label_;
        a_road.emplace_back(*vit);
        // backward
        road_graph_vertex_descriptor cur_vertex = *vit;
        while(true){
            auto es = in_edges(cur_vertex, road_graph_);
            bool has_prev_node = false;
            road_graph_vertex_descriptor source_vertex;
            for(auto eit = es.first; eit != es.second; ++eit){
                source_vertex = source(*eit, road_graph_);
                if(vertex_visited[source_vertex])
                    continue;
                if(road_graph_[source_vertex].road_label == old_road_label){
                    has_prev_node = true;
                    break;
                }
            }
            if(!has_prev_node){
                break;
            }
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
                if(vertex_visited[target_vertex])
                    continue;
                if(road_graph_[target_vertex].road_label == old_road_label){
                    has_nxt_node = true;
                    break;
                }
            }
            if(!has_nxt_node){
                break;
            }
            a_road.emplace_back(target_vertex);
            vertex_visited[target_vertex] = true;
            cur_vertex = target_vertex;
        }
        if(a_road.size() >= 3){
            for (int j = 0; j < a_road.size(); ++j) { 
                road_graph_[a_road[j]].idx_in_road = j;
            } 
            indexed_roads_.emplace_back(a_road);
            max_road_label_++;
        }
        else{
            new_road_label = -1;
        }
        // Correct road label
        if(new_road_label == -1){
            for (size_t j = 0; j < a_road.size(); ++j) { 
                road_graph_[a_road[j]].idx_in_road = -1;
                road_graph_[a_road[j]].is_valid = false;
                road_graph_[a_road[j]].road_label = -1; 
                clear_vertex(a_road[j], road_graph_);
            }
        }
        else{
            for (size_t j = 0; j < a_road.size(); ++j) { 
                road_graph_[a_road[j]].road_label = new_road_label; 
            }
        }
    }
    // Smooth road width
    for (size_t i = 0; i < indexed_roads_.size(); ++i) { 
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        int cum_n_lanes = 0;
        for (size_t j = 0; j < a_road.size(); ++j) { 
            cum_n_lanes += road_graph_[a_road[j]].pt.n_lanes;
        } 
        cum_n_lanes /= a_road.size();
        for (size_t j = 0; j < a_road.size(); ++j) { 
            road_graph_[a_road[j]].pt.n_lanes = cum_n_lanes;
        } 
    }
    // Add to point cloud
    for (size_t i = 0; i < indexed_roads_.size(); ++i) { 
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        //cout << "\tlane num: " << road_graph_[a_road[0]].pt.n_lanes << endl;
        for (size_t j = 0; j < a_road.size(); ++j) { 
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            PclPoint pt;
            pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
            pt.head = r_pt.head;
            pt.id_trajectory = i;
            pt.speed = r_pt.speed;
            pt.t             = r_pt.n_lanes;
            pt.id_sample = j;
            road_points_->push_back(pt);
        } 
    }
    if(road_points_->size() > 0)
        road_point_search_tree_->setInputCloud(road_points_);
    cout << "There are " << indexed_roads_.size() << " roads." << endl;
    // Visualization
    prepareGeneratedMap();
}

void RoadGenerator::computeJunctionClusters(vector<set<road_graph_vertex_descriptor>>& junc_clusters){
    // Cluster nodes with linking edge into clusters
    float search_radius = Parameters::getInstance().searchRadius();
    PclPointCloud::Ptr points(new PclPointCloud);
    PclSearchTree::Ptr search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    typedef adjacency_list<vecS, vecS, undirectedS >    graph_t;
    typedef graph_traits<graph_t>::vertex_descriptor    vertex_descriptor;
    typedef graph_traits<graph_t>::edge_descriptor      edge_descriptor;
    graph_t G;
    auto es = edges(road_graph_);
    int max_extension = 5;
    for (auto eit = es.first; eit != es.second; ++eit) { 
        if(road_graph_[*eit].type == RoadGraphEdgeType::linking){
            PclPoint first_r_pt, second_r_pt;
            RoadGraphNode& first_node = road_graph_[source(*eit, road_graph_)];
            RoadGraphNode& second_node = road_graph_[target(*eit, road_graph_)];
            first_r_pt.setCoordinate(first_node.pt.x, first_node.pt.y, 0.0f);
            first_r_pt.head = first_node.pt.head;
            first_r_pt.t = first_node.pt.n_lanes;
            first_r_pt.id_trajectory = first_node.road_label;
            first_r_pt.id_sample = first_node.idx_in_road;
            second_r_pt.setCoordinate(second_node.pt.x, second_node.pt.y, 0.0f);
            second_r_pt.head = second_node.pt.head;
            second_r_pt.t = second_node.pt.n_lanes;
            second_r_pt.id_trajectory = second_node.road_label;
            second_r_pt.id_sample = second_node.idx_in_road;
            int start_v_idx = points->size();
            points->push_back(first_r_pt);
            points->push_back(second_r_pt);
            vertex_descriptor source_v = add_vertex(G); 
            vertex_descriptor target_v = add_vertex(G); 
            add_edge(source_v, target_v, G);
        }
    }
    if(points->size() > 0)
        search_tree->setInputCloud(points); 
    else{
        cout << "No edge to add." << endl;
        return;
    }
    // Add edge for closeby same road points
    for (size_t i = 0; i < points->size(); ++i) { 
        PclPoint& e_pt = points->at(i);
        // Search nearby
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        search_tree->radiusSearch(e_pt, search_radius, k_indices, k_dist_sqrs);
        for (const auto& idx : k_indices) { 
            if(idx == i){
                continue;
            }
            PclPoint& nb_pt = points->at(idx);
            add_edge(i, idx, G);
        } 
    } 
    // Compute connected component
    vector<int> component(num_vertices(G));
    int         num = connected_components(G, &component[0]);
    cur_num_clusters_ = num;
    
    junc_clusters.resize(num, set<road_graph_vertex_descriptor>());
    for (int i = 0; i != component.size(); ++i){
        PclPoint& e_pt = points->at(i);
        junc_clusters[component[i]].emplace(indexed_roads_[e_pt.id_trajectory][e_pt.id_sample]);
    }
}

void RoadGenerator::localAdjustments(){
    vector<set<road_graph_vertex_descriptor>> junc_clusters;
    computeJunctionClusters(junc_clusters);
    for(size_t i = 0; i < junc_clusters.size(); ++i){
        set<road_graph_vertex_descriptor>& cluster = junc_clusters[i];
        for(const auto& v : cluster){
            RoadPt& r_pt = road_graph_[v].pt;
            points_to_draw_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
            point_colors_.emplace_back(ColorMap::getInstance().getDiscreteColor(i));
        }
    }
    /*
     *Adjust the generated graph to be more like a road map
     */
    map<int, int> road_idx_old_to_new_map;
    vector<vector<RoadPt>> new_roads;
    for (size_t i = 0; i < junc_clusters.size(); ++i) { 
        set<road_graph_vertex_descriptor>& cluster = junc_clusters[i];
        set<int> related_roads;
        for (const auto& v : cluster) { 
            related_roads.emplace(road_graph_[v].road_label);
        } 
    } 
}

void RoadGenerator::prepareGeneratedMap(){
    feature_vertices_.clear();
    feature_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();

    generated_map_points_.clear();
    generated_map_point_colors_.clear(); 
    generated_map_triangle_strips_.clear();
    generated_map_triangle_strip_colors_.clear();
    generated_map_lines_.clear();
    generated_map_line_colors_.clear();
    generated_map_links_.clear();
    generated_map_link_colors_.clear();
 
    Color skeleton_color = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE);
    Color junc_color = ColorMap::getInstance().getNamedColor(ColorMap::YELLOW);
    Color link_color = ColorMap::getInstance().getNamedColor(ColorMap::RED);
    cout << "indexed_road_size: " << indexed_roads_.size() << endl;
    for(size_t i = 0; i < indexed_roads_.size(); ++i){
        // Smooth road width
        vector<Vertex> road_triangle_strip;
        vector<Color>  road_triangle_strip_color;
        vector<road_graph_vertex_descriptor> a_road = indexed_roads_[i];
        Color road_color = ColorMap::getInstance().getDiscreteColor(i);
        for (size_t j = 0; j < a_road.size(); ++j) { 
            RoadPt& r_pt = road_graph_[a_road[j]].pt;
            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            Eigen::Vector2d r_pt_perp_dir(-1.0f * r_pt_dir.y(), r_pt_dir.x());
            float half_width = 0.5f * r_pt.n_lanes * LANE_WIDTH;
            Eigen::Vector2d left_pt(r_pt.x, r_pt.y);
            Eigen::Vector2d right_pt(r_pt.x, r_pt.y);
            left_pt += r_pt_perp_dir * half_width;
            right_pt -= r_pt_perp_dir * half_width;
            road_triangle_strip.emplace_back(SceneConst::getInstance().normalize(left_pt.x(), left_pt.y(), Z_ROAD));
            road_triangle_strip.emplace_back(SceneConst::getInstance().normalize(right_pt.x(), right_pt.y(), Z_ROAD));
            float color_ratio = 1.0f - 0.5f * static_cast<float>(j) / a_road.size();
            road_triangle_strip_color.emplace_back(Color(color_ratio*road_color.r, color_ratio*road_color.g, color_ratio*road_color.b, 1.0f));
            road_triangle_strip_color.emplace_back(Color(color_ratio*road_color.r, color_ratio*road_color.g, color_ratio*road_color.b, 1.0f));
            if(j > 0){
                // Add skeleton
                RoadPt& prev_r_pt = road_graph_[a_road[j]].pt;
                generated_map_lines_.emplace_back(SceneConst::getInstance().normalize(prev_r_pt.x, prev_r_pt.y, Z_ROAD));
                generated_map_lines_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_ROAD));
                generated_map_line_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
                generated_map_line_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
            }
        } 
        generated_map_triangle_strips_.emplace_back(road_triangle_strip);
        generated_map_triangle_strip_colors_.emplace_back(road_triangle_strip_color);
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
            generated_map_links_.emplace_back(SceneConst::getInstance().normalize(source_r_pt.x, source_r_pt.y, Z_ROAD + 0.01f));
            generated_map_links_.emplace_back(SceneConst::getInstance().normalize(target_r_pt.x, target_r_pt.y, Z_ROAD + 0.01f));
            generated_map_link_colors_.emplace_back(link_color);
            generated_map_link_colors_.emplace_back(Color(0.5f* link_color.r, 0.5f * link_color.g, 0.5f * link_color.b, 1.0f));
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

float nonPerpLinkScore(int dh1, int dh2, float length){
    return (dh1 + dh2 + 1) * (length + 1.0f);
}

void RoadGenerator::resolveLink(const Link& link){
    // Check if the link is already resolved.
    vector<road_graph_vertex_descriptor> path;
    float dist = shortestPath(link.source_vertex, 
                              link.target_vertex, 
                              road_graph_,
                              path);
    if(dist < 500){
        cout << "resolved" << endl;
        return;
    }
 
    int ANGLE_PERP_THRESHOLD = 30;
    vector<road_graph_vertex_descriptor>& source_road = indexed_roads_[link.source_related_road_idx];
    vector<road_graph_vertex_descriptor>& target_road = indexed_roads_[link.target_related_road_idx];
    if(link.dh < 90 + ANGLE_PERP_THRESHOLD && link.dh > 90 - ANGLE_PERP_THRESHOLD){
        cout << "perp link" << endl;
        int max_extension = 20;
        // Find best merging point
        float min_dist = POSITIVE_INFINITY;
        int min_source_idx = -1;
        int min_target_idx = -1;
        for(int s = link.source_idx_in_related_roads - max_extension; s <= link.source_idx_in_related_roads + max_extension; ++s){
            if(s < 0)
                continue;
            if(s >= source_road.size())
                break;
            road_graph_vertex_descriptor source_v = source_road[s];
            RoadPt& source_v_pt = road_graph_[source_v].pt;
            Eigen::Vector2d source_v_dir = headingTo2dVector(source_v_pt.head);
            for(int t = link.target_idx_in_related_roads - max_extension; t <= link.target_idx_in_related_roads + max_extension; ++t){
                if(t < 0)
                    continue;
                if(t >= target_road.size())
                    break;
                road_graph_vertex_descriptor target_v = target_road[t];
                RoadPt& target_v_pt = road_graph_[target_v].pt;
                Eigen::Vector2d target_v_dir = headingTo2dVector(target_v_pt.head);
                Eigen::Vector2d vec(target_v_pt.x - source_v_pt.x,
                                    target_v_pt.y - source_v_pt.y);
                if(vec.dot(source_v_dir) > 0 && vec.dot(target_v_dir) > 0){
                    float dist = vec.norm();
                    if(min_dist > dist){
                        min_dist = dist;
                        min_source_idx = s;
                        min_target_idx = t;
                    }
                }
            }
        }
        if(min_source_idx != -1 && min_target_idx != -1){
            if(min_dist > 25.0f)
                return;
            // add link edge between min_source_idx and min_target_idx
            RoadPt& target_v_pt = road_graph_[target_road[min_target_idx]].pt;
            RoadPt& source_v_pt = road_graph_[source_road[min_source_idx]].pt;
            Eigen::Vector2d vec(target_v_pt.x - source_v_pt.x,
                                target_v_pt.y - source_v_pt.y);
            if(min_source_idx >= source_road.size() - 3){
                if(min_source_idx - 3 >= 0)
                    for(int k = min_source_idx - 3; k < source_road.size(); ++k){
                        road_graph_[source_road[k]].pt.head = road_graph_[source_road[min_source_idx-3]].pt.head;
                    }
            }
            if(min_source_idx >= source_road.size() - 1){
                Eigen::Vector2d source_v_dir = headingTo2dVector(source_v_pt.head);
                float dot_value = source_v_dir.dot(vec);
                if(dot_value > 5.0f){
                    // Add point to road
                    road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                    road_graph_[new_v].pt = source_v_pt;
                    Eigen::Vector2d new_loc(source_v_pt.x, source_v_pt.y);
                    new_loc += dot_value * source_v_dir;
                    road_graph_[new_v].pt.x = new_loc.x();
                    road_graph_[new_v].pt.y = new_loc.y();
                    road_graph_[new_v].road_label = road_graph_[source_road[min_source_idx]].road_label;
                    auto es = add_edge(source_road[min_source_idx], new_v, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        road_graph_[es.first].length = dot_value;
                    }
                }
            }
            if(min_target_idx < 3){
                if(min_target_idx + 3 < target_road.size())
                    for(int k = 0; k < min_target_idx + 3; ++k){
                        road_graph_[target_road[k]].pt.head = road_graph_[target_road[min_target_idx+3]].pt.head;
                    }
            }
            if(min_target_idx == 0){
                Eigen::Vector2d target_v_dir = headingTo2dVector(target_v_pt.head);
                float dot_value = target_v_dir.dot(vec);
                if(dot_value > 5.0f){
                    // Add point to road
                    road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                    road_graph_[new_v].pt = target_v_pt;
                    Eigen::Vector2d new_loc(target_v_pt.x, target_v_pt.y);
                    new_loc -= dot_value * target_v_dir;
                    road_graph_[new_v].pt.x = new_loc.x();
                    road_graph_[new_v].pt.y = new_loc.y();
                    road_graph_[new_v].road_label = road_graph_[target_road[min_target_idx]].road_label;
                    auto es = add_edge(new_v, target_road[min_target_idx], road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::normal;
                        road_graph_[es.first].length = dot_value;
                    }
                }
            }
            auto new_e = add_edge(source_road[min_source_idx],
                                  target_road[min_target_idx],
                                  road_graph_);
            if(new_e.second){
                road_graph_[new_e.first].type = RoadGraphEdgeType::linking;
                road_graph_[new_e.first].length = min_dist;
                road_graph_[new_e.first].link_support = 1;
            }
        }
        if(link.is_bidirectional){
            // Find best merging point
            float min_dist = POSITIVE_INFINITY;
            int min_source_idx = -1;
            int min_target_idx = -1;
            for(int t = link.target_idx_in_related_roads - max_extension; t <= link.target_idx_in_related_roads + max_extension; ++t){
                if(t < 0)
                    continue;
                if(t >= target_road.size())
                    break;
                road_graph_vertex_descriptor target_v = target_road[t];
                RoadPt& target_v_pt = road_graph_[target_v].pt;
                Eigen::Vector2d target_v_dir = headingTo2dVector(target_v_pt.head);
                for(int s = link.source_idx_in_related_roads - max_extension; s <= link.source_idx_in_related_roads + max_extension; ++s){
                    if(s < 0)
                        continue;
                    if(s >= source_road.size())
                        break;
                    road_graph_vertex_descriptor source_v = source_road[s];
                    RoadPt& source_v_pt = road_graph_[source_v].pt;
                    Eigen::Vector2d source_v_dir = headingTo2dVector(source_v_pt.head);
                    Eigen::Vector2d vec(source_v_pt.x - target_v_pt.x,
                                        source_v_pt.y - target_v_pt.y);
                    if(vec.dot(target_v_dir) > 0 && vec.dot(source_v_dir)){
                        float dist = vec.norm();
                        if(min_dist > dist){
                            min_dist = dist;
                            min_target_idx = t;
                            min_source_idx = s;
                        }
                    }
                }
            }
            if(min_source_idx != -1 && min_target_idx != -1){
                if(min_dist > 25.0f)
                    return;
                // add link edge between min_source_idx and min_target_idx
                auto new_e = add_edge(target_road[min_target_idx],
                                      source_road[min_source_idx],
                                      road_graph_);
                if(new_e.second){
                    road_graph_[new_e.first].type = RoadGraphEdgeType::linking;
                    road_graph_[new_e.first].length = min_dist;
                    road_graph_[new_e.first].link_support = 1;
                }
            }
        }
    }
    else{ // Non-perp link
        cout << "Non perp" << endl;
        // Check if existing junctions on source road can be used to connect to target road
        RoadPt& from_r_pt = road_graph_[link.source_vertex].pt;
        RoadPt& to_r_pt = road_graph_[link.target_vertex].pt;
        // Check if it is U-turn
        if(link.dh > 135){
            float dist = roadPtDistance(from_r_pt, to_r_pt);
            if(dist < LANE_WIDTH * (from_r_pt.n_lanes + to_r_pt.n_lanes)){
                return;
            }
        }
        float best_score0 = POSITIVE_INFINITY;
        road_graph_vertex_descriptor best_s0, best_t0; 
        road_graph_vertex_descriptor best_s, best_t; 
        for(int s = 0; s < source_road.size(); ++s){
            RoadPt& s_pt = road_graph_[source_road[s]].pt;
            for(int t = 0; t < target_road.size(); ++t){
                RoadPt& t_pt = road_graph_[target_road[t]].pt;
                Eigen::Vector2d vec(t_pt.x - s_pt.x,
                                    t_pt.y - s_pt.y);
                float vec_length = vec.norm();
                int vec_head = s_pt.head;
                if(vec_length > 1e-3)
                    vec_head = vector2dToHeading(vec);
                float delta_h1 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                float delta_h2 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                float score = nonPerpLinkScore(delta_h1, delta_h2, vec_length);
                if(score < best_score0){
                    best_s0 = source_road[s];
                    best_t0 = target_road[t];
                    best_score0 = score;
                }
            }
        }
        best_s = best_s0;
        best_t = best_t0;
        if(link.is_bidirectional){
            float best_score1 = POSITIVE_INFINITY;
            road_graph_vertex_descriptor best_s1, best_t1; 
            for(int t = 0; t < target_road.size(); ++t){
                    RoadPt& t_pt = road_graph_[target_road[t]].pt;
                for(int s = 0; s < source_road.size(); ++s){
                    RoadPt& s_pt = road_graph_[source_road[s]].pt;
                    Eigen::Vector2d vec(s_pt.x - t_pt.x,
                                        s_pt.y - t_pt.y);
                    float vec_length = vec.norm();
                    int vec_head = t_pt.head;
                    if(vec_length > 1e-3)
                        vec_head = vector2dToHeading(vec);
                    float delta_h1 = abs(deltaHeading1MinusHeading2(t_pt.head, vec_head));
                    float delta_h2 = abs(deltaHeading1MinusHeading2(s_pt.head, vec_head));
                    float score = nonPerpLinkScore(delta_h1, delta_h2, vec_length);
                    if(score < best_score0){
                        best_s1 = target_road[t];
                        best_t1 = source_road[s];
                        best_score1 = score;
                    }
                }
            }
            if(best_score1 < best_score0){
                best_s = best_s1;
                best_t = best_t1;
            }
        }
        // Connect from best_s to best_t
        RoadPt& new_from_r_pt = road_graph_[best_s].pt;
        RoadPt& new_to_r_pt = road_graph_[best_t].pt;
        float dx = new_from_r_pt.x - new_to_r_pt.x;
        float dy = new_from_r_pt.y - new_to_r_pt.y;
        float link_length = sqrt(dx*dx + dy*dy);
        if(link_length > 100.0f)
            return;
        Eigen::Vector2d new_from_r_pt_dir = headingTo2dVector(new_from_r_pt.head);
        Eigen::Vector2d new_to_r_pt_dir = headingTo2dVector(new_to_r_pt.head);
        int new_from_r_pt_degree = out_degree(best_s, road_graph_);
        int new_to_r_pt_degree = in_degree(best_t, road_graph_);
        int new_v_road_label = -1;
        int target_n_lanes = 1;
        int target_head = 0;
        if(new_from_r_pt_degree >= 1){
            if(new_to_r_pt_degree == 0){
                new_v_road_label = road_graph_[best_t].road_label;
                target_n_lanes = road_graph_[best_t].pt.n_lanes;
                target_head = road_graph_[best_t].pt.head;
            }
        }
        else{
            new_v_road_label = road_graph_[best_s].road_label;
            target_n_lanes = road_graph_[best_s].pt.n_lanes;
            target_head = road_graph_[best_s].pt.head;
        }
        if(new_v_road_label != -1){
            Eigen::Vector2d new_edge_vec(new_to_r_pt.x - new_from_r_pt.x,
                                     new_to_r_pt.y - new_from_r_pt.y);
            float new_edge_vec_length = new_edge_vec.norm();
            int n_v_to_add = floor(new_edge_vec_length / 5.0f);
            road_graph_vertex_descriptor prev_v;
            if(new_v_road_label != road_graph_[best_s].road_label){
                road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                road_graph_[new_v].pt = new_from_r_pt;
                road_graph_[new_v].pt.n_lanes = target_n_lanes;
                road_graph_[new_v].pt.head = target_head;
                road_graph_[new_v].road_label = new_v_road_label;
                auto es = add_edge(best_s, new_v, road_graph_);
                if(es.second){
                    road_graph_[es.first].type = RoadGraphEdgeType::linking;
                    road_graph_[es.first].link_support = 1;
                    road_graph_[es.first].length = 0.0f;
                }
                if(link.is_bidirectional){
                    auto es = add_edge(new_v, best_s, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        road_graph_[es.first].link_support = 1;
                        road_graph_[es.first].length = 0.0f;
                    }
                }
                prev_v = new_v;
            }
            else{
                prev_v = best_s;
            }
            float d = new_edge_vec_length / (n_v_to_add + 1);
            for(int k = 0; k < n_v_to_add; ++k){
                road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                float new_v_x = new_from_r_pt.x + new_edge_vec.x() * (k+1) * d / new_edge_vec_length;
                float new_v_y = new_from_r_pt.y + new_edge_vec.y() * (k+1) * d / new_edge_vec_length;
                Eigen::Vector2d head_dir = (k+1) * new_from_r_pt_dir + (n_v_to_add - k) * new_to_r_pt_dir;
                road_graph_[new_v].pt = new_from_r_pt;
                road_graph_[new_v].pt.x = new_v_x;
                road_graph_[new_v].pt.y = new_v_y;
                road_graph_[new_v].pt.head = vector2dToHeading(head_dir);
                road_graph_[new_v].pt.n_lanes = target_n_lanes;
                road_graph_[new_v].pt.head = target_head;
                road_graph_[new_v].road_label = new_v_road_label;
                auto es = add_edge(prev_v, new_v, road_graph_);
                if(es.second){
                    road_graph_[es.first].type = RoadGraphEdgeType::normal;
                    road_graph_[es.first].length = d;
                }
                prev_v = new_v;
            }
            if(new_v_road_label != road_graph_[best_t].road_label){
                road_graph_vertex_descriptor new_v = add_vertex(road_graph_);
                auto es = add_edge(prev_v, new_v, road_graph_);
                if(es.second){
                    road_graph_[es.first].type = RoadGraphEdgeType::normal;
                    road_graph_[es.first].length = d;
                }
                road_graph_[new_v].pt = new_to_r_pt;
                road_graph_[new_v].pt.n_lanes = target_n_lanes;
                road_graph_[new_v].pt.head = target_head;
                road_graph_[new_v].road_label = new_v_road_label;
                es = add_edge(new_v, best_t, road_graph_);
                if(es.second){
                    road_graph_[es.first].type = RoadGraphEdgeType::linking;
                    road_graph_[es.first].link_support = 1;
                    road_graph_[es.first].length = 0.0f;
                }
                if(link.is_bidirectional){
                    auto es = add_edge(best_t, new_v, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        road_graph_[es.first].link_support = 1;
                        road_graph_[es.first].length = 0.0f;
                    }
                }
            }
            else{
                auto es = add_edge(prev_v, best_t, road_graph_);
                if(es.second){
                    road_graph_[es.first].type = RoadGraphEdgeType::normal;
                    road_graph_[es.first].length = d;
                }
            }
        }
        else{
            float dx = road_graph_[best_s].pt.x - road_graph_[best_t].pt.x;
            float dy = road_graph_[best_s].pt.y - road_graph_[best_t].pt.y;
            float length = sqrt(dx*dx + dy*dy);
            if(length < 25.0f){
                auto es = add_edge(best_s, best_t, road_graph_);
                if(es.second){
                    road_graph_[es.first].type = RoadGraphEdgeType::linking;
                    road_graph_[es.first].link_support = 1;
                    road_graph_[es.first].length = length;
                }
                if(link.is_bidirectional){
                    auto es = add_edge(best_t, best_s, road_graph_);
                    if(es.second){
                        road_graph_[es.first].type = RoadGraphEdgeType::linking;
                        road_graph_[es.first].link_support = 1;
                        road_graph_[es.first].length = length;
                    }
                }
            }
        }
    }
}

void RoadGenerator::connectRoads(){
    /*
     *This is THE FUNCTION that modify road_graph_ structures to include junctions!!! 
     *      This function deal with one linking cluster at a time.
     */
    // Collect all links 
    vector<Link> orig_links;
    set<road_graph_vertex_descriptor> visited_vertices;
    vector<vector<road_graph_vertex_descriptor>> related_roads;
    map<pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>, int> existing_links;
    // map<pair<road_idx1, road_idx2>, link_idx in links>
    map<pair<int, int>, int> existing_link_pairs;
    auto es = edges(road_graph_);
    vector<road_graph_edge_descriptor> edges_to_remove;
    for(auto eit = es.first; eit != es.second; ++eit){
        if(road_graph_[*eit].type == RoadGraphEdgeType::auxiliary){
            road_graph_vertex_descriptor from_vertex = source(*eit, road_graph_);
            road_graph_vertex_descriptor to_vertex = target(*eit, road_graph_);
            // Check if the reverse link exists 
            if(existing_links.find(pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>(to_vertex, from_vertex)) != existing_links.end()){
                orig_links[existing_links[pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>(to_vertex, from_vertex)]].is_bidirectional = true;
            }
            else{
                existing_links[pair<road_graph_vertex_descriptor, road_graph_vertex_descriptor>(from_vertex, to_vertex)] = orig_links.size();
                Link new_link;
                new_link.source_vertex = from_vertex;
                new_link.target_vertex = to_vertex;
                new_link.source_related_road_idx = road_points_->at(from_vertex).id_trajectory;
                new_link.source_idx_in_related_roads = road_points_->at(from_vertex).id_sample;
                new_link.target_related_road_idx = road_points_->at(to_vertex).id_trajectory;
                new_link.target_idx_in_related_roads = road_points_->at(to_vertex).id_sample;
                orig_links.emplace_back(new_link);
            }
            edges_to_remove.emplace_back(*eit);
        }
    }
    // Remove edges
    for (size_t i = 0; i < edges_to_remove.size(); ++i) { 
        remove_edge(edges_to_remove[i], road_graph_);
    } 
    cout << orig_links.size() << " links" << endl;
    float max_delta_speed = 0.0f;
    for(size_t i = 0; i < orig_links.size(); ++i){
        PclPoint& source_pt = road_points_->at(orig_links[i].source_vertex);
        PclPoint& target_pt = road_points_->at(orig_links[i].target_vertex);
        Eigen::Vector2d vec(target_pt.x - source_pt.x,
                            target_pt.y - source_pt.y);
        Eigen::Vector2d source_speed = source_pt.speed * headingTo2dVector(source_pt.head);
        Eigen::Vector2d target_speed = target_pt.speed * headingTo2dVector(target_pt.head);
        Eigen::Vector2d delta_speed = target_speed - source_speed;
        orig_links[i].dh = abs(deltaHeading1MinusHeading2(source_pt.head, target_pt.head));
        float delta_speed_norm = delta_speed.norm();
        orig_links[i].delta_speed = delta_speed_norm;
        orig_links[i].length = vec.norm();
        if(max_delta_speed < delta_speed_norm){
            max_delta_speed = delta_speed_norm;
        }
    }
    cout << "max delta speed: " << max_delta_speed << endl;
    // Sort link by scores
    vector<pair<int, float>> link_scores;
    float max_score = -1.0f;
    for (size_t i = 0; i < orig_links.size(); ++i) { 
        float tmp_length = (orig_links[i].length > 25.0f) ? 25.0f : orig_links[i].length;
        float score = orig_links[i].delta_speed / (tmp_length + 0.01f);
        //if(orig_links[i].dh < 120 && orig_links[i].dh > 60)
        //    score *= 2.0f;
        if(max_score < score)
            max_score = score;
        link_scores.emplace_back(pair<int, float>(i, score));
    }
    // Visualize links
    //lines_to_draw_.clear();
    //line_colors_.clear();
    //for (size_t i = 0; i < orig_links.size(); ++i) { 
    //    RoadPt& source_r_pt = road_graph_[orig_links[i].source_vertex].pt;
    //    RoadPt& target_r_pt = road_graph_[orig_links[i].target_vertex].pt;
    //    Color c = ColorMap::getInstance().getJetColor(link_scores[i].second / max_score);
    //    lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(source_r_pt.x, source_r_pt.y, Z_DEBUG + 0.02f));
    //    lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(target_r_pt.x, target_r_pt.y, Z_DEBUG + 0.02f));
    //    line_colors_.emplace_back(c);
    //    line_colors_.emplace_back(Color(c.r * 0.1f, c.g * 0.1f, c.b * 0.1f, 1.0f));
    //}
    sort(link_scores.begin(), link_scores.end(), pairCompare);
    vector<Link> links;
    for (size_t i = 0; i < link_scores.size(); ++i) { 
        Link& o_link = orig_links[link_scores[i].first];
        bool v1 = false;
        if(o_link.source_idx_in_related_roads < 5 || o_link.source_idx_in_related_roads + 5 > indexed_roads_[o_link.source_related_road_idx].size())
            v1 = true;
        bool v2 = false;
        if(o_link.target_idx_in_related_roads < 5 || o_link.target_idx_in_related_roads + 5 > indexed_roads_[o_link.target_related_road_idx].size())
            v2 = true;

        if(o_link.dh < 120 && o_link.dh > 60){
            if(o_link.delta_speed > 800)
                if(!v1 && !v2)
                    continue;
        }
        links.emplace_back(orig_links[link_scores[i].first]);
    }
    // resolve links
    for (size_t i = 0; i < links.size(); ++i) { 
    //int n = (links.size() > 10) ? 10 : links.size();
    //for (size_t i = 0; i < n; ++i) { 
        Link& a_link = links[i];
        cout << "link " << i << endl;
        cout << "\t";
        resolveLink(a_link);
    } 
}

void RoadGenerator::finalAdjustment(){
    /*
     *Final adjustment step, which add junctions for floating roads when possible, remove floating roads, and remove short road pieces. Also merge road, adjust road widths, etc. Parallel same direction.
     */
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

float RoadGenerator::shortestPath(road_graph_vertex_descriptor source, 
                                  road_graph_vertex_descriptor target, 
                                  road_graph_t&                graph,              
                                  vector<road_graph_vertex_descriptor>& path){
    path.clear();
    
    vector<road_graph_vertex_descriptor> p(num_vertices(graph));
    vector<float> d(num_vertices(graph));

    try{
        astar_search_tree(graph, 
                          source, 
                          distance_heuristic<road_graph_t, float>(target, graph), 
                          //predecessor_map(make_iterator_property_map(p.begin(), get(boost::vertex_index, road_graph_))).
                          //distance_map(make_iterator_property_map(d.begin(), get(boost::vertex_index, road_graph_))).
                          predecessor_map(&p[0]).
                          distance_map(&d[0]).
                          weight_map(get(&RoadGraphEdge::length, graph)).
                          visitor(astar_goal_visitor<road_graph_vertex_descriptor>(target)));

    }catch (found_goal fg){
        for(road_graph_vertex_descriptor v = target; ; v = p[v]){
            path.insert(path.begin(), v);
            if(p[v] == v)
                break;
        }
        return d[target];
    }

    return POSITIVE_INFINITY;
} 

float RoadGenerator::probOnRoad(const PclPoint& pt, 
                                const PclPoint& r_pt,
                                float sigma_w,
                                float sigma_l,
                                float sigma_h){
    float SIGMA_W = sigma_w;
    float SIGMA_L = sigma_l;
    float SIGMA_H = sigma_h;
    float CUT_OFF_PROBABILITY = 0.01f; // this is only for projection
    float delta_h = abs(deltaHeading1MinusHeading2(pt.head, r_pt.head));
    if(delta_h > 45.0f) 
        return 0.0f;
    Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
    Eigen::Vector2d vec(pt.x - r_pt.x,
                        pt.y - r_pt.y);
    float delta_l = vec.dot(r_pt_dir);
    float delta_w = sqrt(abs(vec.dot(vec) - delta_l*delta_l));
    delta_w -= 0.25f * r_pt.t * LANE_WIDTH;
    if(delta_w < 0.0f) 
        delta_w = 0.0f;
    float prob = exp(-1.0f * delta_w * delta_w / 2.0f / SIGMA_W / SIGMA_W) 
               * exp(-1.0f * delta_l * delta_l / 2.0f / SIGMA_L / SIGMA_L) ;
               //* exp(-1.0f * delta_h * delta_h / 2.0f / SIGMA_H / SIGMA_H);
    if(prob < CUT_OFF_PROBABILITY) 
        return 0.0f;
    return prob;
}

bool RoadGenerator::mapMatching(size_t traj_idx, vector<road_graph_vertex_descriptor>& projection){
    projection.clear();

    if(traj_idx > trajectories_->getNumTraj() - 1){
        cout << "Warning from RoadGenerator::partialMapMatching: traj_idx greater than the actual number of trajectories." << endl;
        return false;
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
        return false;
    } 

    vector<vector<pair<int, float>>> candidate_projections; // idx in road_points_, not vertex!

    for (size_t i = 0; i < traj.size(); ++i) { 
        PclPoint& pt = trajectories_->data()->at(traj[i]);  
        vector<pair<int, float>> candidate_projection;
        // Search nearby road_points_
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        road_point_search_tree_->radiusSearch(pt,
                                              SEARCH_RADIUS,
                                              k_indices,
                                              k_dist_sqrs); 
        float cur_max_projection = 0.0f;
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
        if(candidate_projection.size() > 0){
            candidate_projections.emplace_back(candidate_projection);
        }
    } 
    //cout << "projection results:" << endl;
    //for (size_t i = 0; i < candidate_projections.size() ; ++i) { 
    //    cout << "\t " << i << ": ";
    //    vector<pair<int, float>>& candidate_projection = candidate_projections[i];
    //    for (size_t j = 0; j < candidate_projection.size() ; ++j) { 
    //        if(candidate_projection[j].first != -1) 
    //            cout << "(" << candidate_projection[j].first << ", " << road_points_->at(candidate_projection[j].first).id_trajectory << ", " << candidate_projection[j].second <<"), ";
    //        else{
    //            cout << "(-1,"<< candidate_projection[j].second << "), ";
    //        }
    //    } 
    //    cout << endl;
    //} 
    //cout << endl;
    if(candidate_projections.size() == 0){
        return false;
    }
    //HMM map matching
    float MIN_TRANSISTION_SCORE = 1e-9; // Everything is connected to everything with this minimum probability
    float MIN_LOG_TRANSISTION_SCORE = log10(MIN_TRANSISTION_SCORE);
    vector<vector<int>>   pre(candidate_projections.size(), vector<int>());
    vector<vector<float>> scores(candidate_projections.size(), vector<float>());
    // Initialize scores
    for (size_t i = 0; i < candidate_projections.size(); ++i) { 
        vector<float>& score = scores[i];
        vector<int>&   prei  = pre[i];
        vector<pair<int, float>>& candidate_projection = candidate_projections[i];
        score.resize(candidate_projection.size(), MIN_TRANSISTION_SCORE); 
        prei.resize(candidate_projection.size(), -1);
        if(i == 0){
            for (size_t j = 0; j < candidate_projection.size(); ++j) { 
                score[j] = log10(candidate_projection[j].second); 
            }  
        } 
    } 
    for (size_t i = 1; i < candidate_projections.size() ; ++i) { 
        vector<pair<int, float>>& R = candidate_projections[i-1];
        vector<pair<int, float>>& L = candidate_projections[i];
        vector<int>&           prei = pre[i];
        vector<float>& scorer = scores[i-1];
        vector<float>& scorel = scores[i];
        for (size_t j = 0; j < L.size(); ++j) { 
            int cur_max_idx = -1;
            float cur_max   = -POSITIVE_INFINITY;
            for (size_t k = 0; k < R.size(); ++k) { 
                // Compute prob(L[j].first -> R[k].first) based on road_graph_
                float p_r_l = MIN_TRANSISTION_SCORE; 
                // Compute shortest path
                vector<road_graph_vertex_descriptor> path;
                road_graph_vertex_descriptor source_idx, target_idx;
                PclPoint& R_pt = road_points_->at(R[k].first);
                PclPoint& L_pt = road_points_->at(L[j].first);
                source_idx = indexed_roads_[R_pt.id_trajectory][R_pt.id_sample];
                target_idx = indexed_roads_[L_pt.id_trajectory][L_pt.id_sample];
                if(source_idx == target_idx){
                    p_r_l = 1.0f;
                }
                else{
                    float dist = shortestPath(source_idx, 
                                              target_idx, 
                                              road_graph_,
                                              path);
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

    // Raw projection results
    vector<road_graph_vertex_descriptor> raw_projection;
    if(last_max_idx != -1){
        raw_projection.resize(candidate_projections.size(), -1); 
        int last_idx = last_max_idx;
        PclPoint& pt = road_points_->at(candidate_projections[candidate_projections.size()-1][last_max_idx].first);
        raw_projection.back() = indexed_roads_[pt.id_trajectory][pt.id_sample];
        for(int i = candidate_projections.size()-1; i >= 1; --i){
            PclPoint& n_pt = road_points_->at(candidate_projections[i-1][pre[i][last_idx]].first);
            raw_projection[i-1] = indexed_roads_[n_pt.id_trajectory][n_pt.id_sample];
            last_idx = pre[i][last_idx];
        }
    } 
    //cout << "Projection results: " << endl;
    //cout << "\t";
    //for (size_t i = 0; i < raw_projection.size(); ++i) { 
    //    cout << raw_projection[i] << ", ";
    //    projection.emplace_back(raw_projection[i]);
    //} 
    //cout << endl;
    // Connect the path using shortest path
    if(raw_projection.size() > 0){
        bool is_explainable = true;
        for (int i = 0; i < raw_projection.size(); ++i) { 
            if(i > 0){
                // Check edge
                if(raw_projection[i-1] == raw_projection[i]) 
                    continue;
                auto e = edge(raw_projection[i-1], raw_projection[i], road_graph_);
                if(!e.second){
                    // Get shortest path
                    vector<road_graph_vertex_descriptor> path;
                    float dist = shortestPath(raw_projection[i-1], 
                                              raw_projection[i], 
                                              road_graph_,
                                              path);
                    if(path.size() <= 2){
                        is_explainable = false;
                        break;
                    }
                    else{
                        for (size_t j = 1; j < path.size() - 1; ++j) { 
                            projection.emplace_back(path[j]);
                        } 
                    }
                }
            }
            projection.emplace_back(raw_projection[i]);
        } 
        if(is_explainable){
            return true;
        }
        else{
            projection.clear();
            return false;
        }
    }
    else{
        return false;
    }
    //cout << "Projection results: " << endl;
    //cout << "\t";
    //for (size_t i = 0; i < projection.size(); ++i) { 
    //    cout << projection[i] << ", ";
    //} 
    //cout << endl;
    return true;
} 

bool RoadGenerator::mapMatchingToOsm(size_t traj_idx, vector<road_graph_vertex_descriptor>& projection){
    projection.clear();
    if(traj_idx > trajectories_->getNumTraj() - 1){
        cout << "Warning from RoadGenerator::mapMatchingToOsm: traj_idx greater than the actual number of trajectories." << endl;
        return false;
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
        return false;
    } 
    vector<vector<pair<int, float>>> candidate_projections; // idx in road_points_, not vertex!
    for (size_t i = 0; i < traj.size(); ++i) { 
        PclPoint& pt = trajectories_->data()->at(traj[i]);  
        vector<pair<int, float>> candidate_projection;
        // Search nearby road_points_
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        osmMap_->map_search_tree()->radiusSearch(pt,
                                                 SEARCH_RADIUS,
                                                 k_indices,
                                                 k_dist_sqrs); 
        float cur_max_projection = 0.0f;
        map<int, pair<int, float>> road_max_prob;
        for (size_t j = 0; j < k_indices.size() ; ++j) { 
            PclPoint& r_pt = osmMap_->map_point_cloud()->at(k_indices[j]); 
            Eigen::Vector2d vec(pt.x - r_pt.x,
                                pt.y - r_pt.y);
            float dist = vec.norm();
            float prob = exp(-1.0f * dist * dist / 2.0f / SIGMA_L / SIGMA_L);
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
        if(candidate_projection.size() > 0){
            candidate_projections.emplace_back(candidate_projection);
        }
    } 
    if(candidate_projections.size() == 0){
        return false;
    }
    //HMM map matching
    float MIN_TRANSISTION_SCORE = 1e-9; // Everything is connected to everything with this minimum probability
    float MIN_LOG_TRANSISTION_SCORE = log10(MIN_TRANSISTION_SCORE);
    vector<vector<int>>   pre(candidate_projections.size(), vector<int>());
    vector<vector<float>> scores(candidate_projections.size(), vector<float>());
    // Initialize scores
    for (size_t i = 0; i < candidate_projections.size(); ++i) { 
        vector<float>& score = scores[i];
        vector<int>&   prei  = pre[i];
        vector<pair<int, float>>& candidate_projection = candidate_projections[i];
        score.resize(candidate_projection.size(), MIN_TRANSISTION_SCORE); 
        prei.resize(candidate_projection.size(), -1);
        if(i == 0){
            for (size_t j = 0; j < candidate_projection.size(); ++j) { 
                score[j] = log10(candidate_projection[j].second); 
            }  
        } 
    } 
    for (size_t i = 1; i < candidate_projections.size() ; ++i) { 
        vector<pair<int, float>>& R = candidate_projections[i-1];
        vector<pair<int, float>>& L = candidate_projections[i];
        vector<int>&           prei = pre[i];
        vector<float>& scorer = scores[i-1];
        vector<float>& scorel = scores[i];
        for (size_t j = 0; j < L.size(); ++j) { 
            int cur_max_idx = -1;
            float cur_max   = -POSITIVE_INFINITY;
            for (size_t k = 0; k < R.size(); ++k) { 
                // Compute prob(L[j].first -> R[k].first) based on road_graph_
                float p_r_l = MIN_TRANSISTION_SCORE; 
                // Compute shortest path
                vector<road_graph_vertex_descriptor> path;
                road_graph_vertex_descriptor source_idx, target_idx;
                PclPoint& R_pt = osmMap_->map_point_cloud()->at(R[k].first);
                PclPoint& L_pt = osmMap_->map_point_cloud()->at(L[j].first);
                source_idx = R[k].first;
                target_idx = L[j].first;
                if(source_idx == target_idx){
                    p_r_l = 1.0f;
                }
                else{
                    float dist = shortestPath(source_idx, 
                                              target_idx, 
                                              osm_graph_,
                                              path);
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
    // Raw projection results
    vector<road_graph_vertex_descriptor> raw_projection;
    if(last_max_idx != -1){
        raw_projection.resize(candidate_projections.size(), -1); 
        int last_idx = last_max_idx;
        raw_projection.back() = candidate_projections[candidate_projections.size()-1][last_max_idx].first;
        for(int i = candidate_projections.size()-1; i >= 1; --i){
            raw_projection[i-1] = candidate_projections[i-1][pre[i][last_idx]].first;
            last_idx = pre[i][last_idx];
        }
    } 
    //cout << "Projection results: " << endl;
    //cout << "\t";
    //for (size_t i = 0; i < raw_projection.size(); ++i) { 
    //    cout << raw_projection[i] << ", ";
    //} 
    //cout << endl;
    // Connect the path using shortest path
    if(raw_projection.size() > 0){
        bool is_explainable = true;
        for (int i = 0; i < raw_projection.size(); ++i) { 
            if(i > 0){
                // Check edge
                if(raw_projection[i-1] == raw_projection[i]) 
                    continue;
                auto e = edge(raw_projection[i-1], raw_projection[i], osm_graph_);
                if(!e.second){
                    // Get shortest path
                    vector<road_graph_vertex_descriptor> path;
                    float dist = shortestPath(raw_projection[i-1], 
                                              raw_projection[i],
                                              osm_graph_,
                                              path);
                    if(path.size() < 2){
                        is_explainable = false;
                        break;
                    }
                    else{
                        for (size_t j = 1; j < path.size() - 1; ++j) { 
                            projection.emplace_back(path[j]);
                        } 
                    }
                }
            }
            projection.emplace_back(raw_projection[i]);
        } 
        if(is_explainable){
            return true;
        }
        else{
            projection.clear();
            return false;
        }
    }
    else{
        return false;
    }
    return true;
}

void RoadGenerator::partialMapMatching(size_t traj_idx, float search_radius, float cut_off_probability, vector<int>& projection){
    projection.clear();
    if(traj_idx > trajectories_->getNumTraj() - 1){
        cout << "Warning from RoadGenerator::partialMapMatching: traj_idx greater than the actual number of trajectories." << endl;
        return;
    } 
    float SEARCH_RADIUS = search_radius;  
    float CUT_OFF_PROBABILITY = cut_off_probability; // this is only for projection
    float SIGMA_W = Parameters::getInstance().roadSigmaW();
    float SIGMA_L = 2.5f;
    float SIGMA_H = Parameters::getInstance().gpsMaxHeadingError();
    // Project each GPS points to nearby road_points_
    const vector<int>& traj = trajectories_->trajectories()[traj_idx];
    if(traj.size() < 2){
        return;
    } 
    // Compute candidate projections
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
        // map<id_road, pair<id_road_pt, probability>>
        map<int, pair<int, float>> road_max_prob;
        for (size_t j = 0; j < k_indices.size() ; ++j) { 
            PclPoint& r_pt = road_points_->at(k_indices[j]); 
            float delta_h = abs(deltaHeading1MinusHeading2(pt.head, r_pt.head));
            //if(delta_h > 2.0f * SIGMA_H) 
            //    continue;
            if(delta_h > 45.0f) 
                continue;
            Eigen::Vector2d r_pt_dir = headingTo2dVector(r_pt.head);
            Eigen::Vector2d vec(pt.x - r_pt.x,
                                pt.y - r_pt.y);
            float delta_l = vec.dot(r_pt_dir);
            float delta_w = sqrt(abs(vec.dot(vec) - delta_l*delta_l));
            delta_w -= 0.5f * r_pt.t * LANE_WIDTH;
            if(delta_w < 0.0f) 
                delta_w = 0.0f;
            float prob = exp(-1.0f * delta_w * delta_w / 2.0f / SIGMA_W / SIGMA_W) 
                       * exp(-1.0f * delta_l * delta_l / 2.0f / SIGMA_L / SIGMA_L) ;
                       //* exp(-1.0f * delta_h * delta_h / 2.0f / SIGMA_H / SIGMA_H);
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
    float MIN_TRANSISTION_SCORE = 0.001f; // Everything is connected to everything with this minimum probability
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
                        vector<road_graph_vertex_descriptor> path;
                        road_graph_vertex_descriptor source_idx, target_idx;
                        PclPoint& R_pt = road_points_->at(R[k].first);
                        PclPoint& L_pt = road_points_->at(L[j].first);
                        source_idx = indexed_roads_[R_pt.id_trajectory][R_pt.id_sample];
                        target_idx = indexed_roads_[L_pt.id_trajectory][L_pt.id_sample];
                        float dist = shortestPath(source_idx, 
                                                  target_idx, 
                                                  road_graph_, 
                                                  path);
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

void RoadGenerator::compareDistance(){
    precisionRecall();
}

void RoadGenerator::showTrajectory(){
    lines_to_draw_.clear();
    line_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();
    if(osm_trajectories_.size() == 0){
        cout << "Please do map matching to OSM first!" << endl;
        return;
    }
    int traj_idx = traj_idx_to_show_ % trajectories_->trajectories().size();
    // Draw trajectory
    const vector<int>& traj = trajectories_->trajectories()[traj_idx];
    Color orig_color = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE);
    Color osm_traj_color = ColorMap::getInstance().getNamedColor(ColorMap::GREEN);
    Color mm_traj_color = ColorMap::getInstance().getNamedColor(ColorMap::RED);
    PclPoint& p0 = trajectories_->data()->at(traj[0]); 
    points_to_draw_.emplace_back(SceneConst::getInstance().normalize(p0.x, p0.y, Z_DEBUG));
    point_colors_.emplace_back(orig_color);
    for(int i = 1; i < traj.size(); ++i){
        PclPoint& p1 = trajectories_->data()->at(traj[i]); 
        PclPoint& p2 = trajectories_->data()->at(traj[i-1]); 
        float color_ratio = 1.0f - 0.5f * i / traj.size();
        Color c(color_ratio*orig_color.r, color_ratio*orig_color.g, color_ratio*orig_color.b, 1.0f);
        points_to_draw_.emplace_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG));
        point_colors_.emplace_back(c);
        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG));
        line_colors_.emplace_back(c);
        lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(p2.x, p2.y, Z_DEBUG));
        line_colors_.emplace_back(c);
    }
    cout << "Trajectory " << traj_idx << endl;
    if(osm_trajectories_[traj_idx].size() > 0){
        cout << "\tis explained by OpenStreetMap." << endl;
        vector<road_graph_vertex_descriptor>& osm_traj = osm_trajectories_[traj_idx];
        RoadPt& p0 = osm_graph_[osm_traj[0]].pt;
        points_to_draw_.emplace_back(SceneConst::getInstance().normalize(p0.x, p0.y, Z_DEBUG));
        point_colors_.emplace_back(osm_traj_color);
        for(int i = 1; i < osm_traj.size(); ++i){
            RoadPt& p1 = osm_graph_[osm_traj[i]].pt; 
            RoadPt& p2 = osm_graph_[osm_traj[i-1]].pt; 
            float color_ratio = 1.0f - 0.5f * i / osm_traj.size();
            Color c(color_ratio*osm_traj_color.r, color_ratio*osm_traj_color.g, color_ratio*osm_traj_color.b, 1.0f);
            points_to_draw_.emplace_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG));
            point_colors_.emplace_back(c);
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG));
            line_colors_.emplace_back(c);
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(p2.x, p2.y, Z_DEBUG));
            line_colors_.emplace_back(c);
        }
    }
    else{
        cout << "\tis NOT explained by OpenStreetMap." << endl;
    }
    if(map_matched_trajectories_[traj_idx].size() > 0){
        cout << "\tis explained by our constructed map." << endl;
        vector<road_graph_vertex_descriptor>& mm_traj = map_matched_trajectories_[traj_idx];
        RoadPt& p0 = road_graph_[mm_traj[0]].pt;
        points_to_draw_.emplace_back(SceneConst::getInstance().normalize(p0.x, p0.y, Z_DEBUG));
        point_colors_.emplace_back(mm_traj_color);
        for(int i = 1; i < mm_traj  .size(); ++i){
            RoadPt& p1 = road_graph_[mm_traj[i]].pt; 
            RoadPt& p2 = road_graph_[mm_traj[i-1]].pt; 
            float color_ratio = 1.0f - 0.5f * i / mm_traj.size();
            Color c(color_ratio*mm_traj_color.r, color_ratio*mm_traj_color.g, color_ratio*mm_traj_color.b, 1.0f);
            points_to_draw_.emplace_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG));
            point_colors_.emplace_back(c);
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(p1.x, p1.y, Z_DEBUG));
            line_colors_.emplace_back(c);
            lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(p2.x, p2.y, Z_DEBUG));
            line_colors_.emplace_back(c);
        }
    }
    else{
        cout << "\tis NOT explained by our constructed map." << endl;
    }
    traj_idx_to_show_++;
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

/*
 *Evaluation Section
 */
void RoadGenerator::evaluationMapMatchingToOsm(){
    if(!osm_map_valid_){
        cout << "Please load OSM first." << endl;
        return;
    }
    osm_trajectories_.clear();
    int n_explainable = 0;
    int n_valid_traj =0;
    for(size_t i = 0; i < trajectories_->trajectories().size(); ++i){
        if(trajectories_->trajectories()[i].size() >= 2){
            n_valid_traj++;
            vector<road_graph_vertex_descriptor> projection;
            bool is_explainable = mapMatchingToOsm(i, projection); 
            osm_trajectories_.emplace_back(projection);
            if(is_explainable){
                n_explainable++;
            }
        }
        else{
            vector<road_graph_vertex_descriptor> projection;
            osm_trajectories_.emplace_back(projection);
        }
    }
    float explainable_percentage = static_cast<float>(n_explainable) * 100.0f / n_valid_traj;
    cout << "Map matching to OpenStreetMap:" << endl;
    cout << "\t" << n_explainable << " trajectories in " << n_valid_traj << " can be explained." << endl;
    cout << "\tExplanation rate = " << explainable_percentage << "%" << endl;

    // Overlapping
    int n_explained_by_both = 0;
    for (size_t i = 0; i < map_matched_trajectories_.size(); ++i) { 
        if(map_matched_trajectories_[i].size() >= 1 &&
           osm_trajectories_[i].size() >= 1){
            n_explained_by_both++;
        }
    } 

    float both_explainable_percentage = static_cast<float>(n_explained_by_both) * 100.0f / n_valid_traj;
    cout << endl;
    cout << "\t" << n_explained_by_both << " trajectories in " << n_valid_traj << " can be explained by both." << endl;
    cout << "\tOverlapping rate = " << both_explainable_percentage << "%" << endl;
    computeHausdorffDistance();
}

void RoadGenerator::evaluationMapMatching(){
    if(!road_graph_valid_){
        cout << "Please compute map first." << endl;
        return;
    }
    map_matched_trajectories_.clear();
    if(indexed_roads_.size() == 0 || num_vertices(road_graph_) == 0){
        cout << "Please generate a map first!" << endl;
        return;
    }
    bool mode = false;
    if(mode){
        int traj_idx = test_i_ % trajectories_->trajectories().size();
        cout << "Trajectory " << traj_idx << " has " << trajectories_->trajectories()[traj_idx].size() << " points." << endl;
        vector<road_graph_vertex_descriptor> projection;
        bool is_explainable = mapMatching(traj_idx, projection); 
        if(is_explainable){
            cout << "\tCan be explained!" << endl;
        }
        else{
            cout << "\tCan NOT be explained!" << endl;
        }
        // Visualization
        points_to_draw_.clear();
        point_colors_.clear();
        lines_to_draw_.clear();
        line_colors_.clear(); 
        const vector<int>& traj = trajectories_->trajectories()[traj_idx];
        Color traj_c = ColorMap::getInstance().getNamedColor(ColorMap::RED);
        for (size_t i = 0; i < traj.size(); ++i) { 
            float color_ratio = 1.0f - 0.5f * static_cast<float>(i) / traj.size();
            PclPoint& pt = trajectories_->data()->at(traj[i]);
            points_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG+0.01f));
            point_colors_.emplace_back(Color(color_ratio*traj_c.r, color_ratio*traj_c.g, color_ratio*traj_c.b, 1.0f));
            if(i > 0){
                PclPoint& prev_pt = trajectories_->data()->at(traj[i-1]);
                lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(prev_pt.x, prev_pt.y, Z_DEBUG+0.01f));
                lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
                line_colors_.emplace_back(Color(color_ratio*traj_c.r, color_ratio*traj_c.g, color_ratio*traj_c.b, 1.0f));
                line_colors_.emplace_back(Color(color_ratio*traj_c.r, color_ratio*traj_c.g, color_ratio*traj_c.b, 1.0f));
            }
        } 
        Color proj_c = ColorMap::getInstance().getNamedColor(ColorMap::GREEN);
        for (size_t i = 0; i < projection.size(); ++i) { 
            float color_ratio = 1.0f - 0.5f * static_cast<float>(i) / projection.size();
            RoadPt& r_pt = road_graph_[projection[i]].pt;
            points_to_draw_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG+0.01f));
            point_colors_.emplace_back(Color(color_ratio*proj_c.r, color_ratio*proj_c.g, color_ratio*proj_c.b, 1.0f));
            if(i > 0){
                RoadPt& prev_r_pt = road_graph_[projection[i-1]].pt;
                lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(prev_r_pt.x, prev_r_pt.y, Z_DEBUG));
                lines_to_draw_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
                line_colors_.emplace_back(Color(color_ratio*proj_c.r, color_ratio*proj_c.g, color_ratio*proj_c.b, 1.0f));
                line_colors_.emplace_back(Color(color_ratio*proj_c.r, color_ratio*proj_c.g, color_ratio*proj_c.b, 1.0f));
            }
        }
        test_i_++;
    }
    else{
        map_matched_trajectories_.clear();
        int n_explainable = 0;
        vector<int> explainable_traj_idxs;
        int n_valid_traj = 0;
        for(size_t i = 0; i < trajectories_->trajectories().size(); ++i){
            if(trajectories_->trajectories()[i].size() >= 2){
                n_valid_traj++;
                vector<road_graph_vertex_descriptor> projection;
                bool is_explainable = mapMatching(i, projection); 
                map_matched_trajectories_.emplace_back(projection);
                if(is_explainable){
                    n_explainable++;
                }
            }
            else{
                vector<road_graph_vertex_descriptor> projection;
                map_matched_trajectories_.emplace_back(projection);
            }
        }
        float explainable_percentage = static_cast<float>(n_explainable) * 100.0f / n_valid_traj;
        cout << "Map matching to our constructed map:" << endl;
        cout << "\t" << n_explainable << " trajectories in " << n_valid_traj << " can be explained." << endl;
        cout << "\tExplanation rate = " << explainable_percentage << "%" << endl;
        // Write results to file
        //ofstream output;
        //output.open("explainable_trajs.txt");
        //if (output.fail()) {
        //    return;
        //}
        //output << explainable_traj_idxs.size() << endl;
        //for (size_t i = 0; i < explainable_traj_idxs.size(); ++i) {
        //    int traj_idx = explainable_traj_idxs[i];
        //    vector<int> traj = trajectories_->trajectories()[traj_idx];
        //    output << traj.size() << endl;
        //    for (size_t j = 0; j < traj.size(); ++j) { 
        //        output << trajectories_->data()->at(traj[j]).x << ", " << trajectories_->data()->at(traj[j]).y << endl;
        //    } 
        //}
        //output.close();
        //output.open("map_matched_trajectories.txt");
        //if (output.fail()) {
        //    return;
        //}
        //output << mm_trajs.size() << endl;
        //for (size_t i = 0; i < mm_trajs.size(); ++i) {
        //    vector<road_graph_vertex_descriptor>& traj = mm_trajs[i];
        //    output << traj.size() << endl;
        //    for (size_t j = 0; j < traj.size(); ++j) { 
        //        output << road_graph_[traj[j]].pt.x << ", " << road_graph_[traj[j]].pt.y << endl;
        //    } 
        //}
        //output.close();
    }
}

void RoadGenerator::computeHausdorffDistance(){
    vector<float> h_dist;
    for (size_t i = 0; i < map_matched_trajectories_.size(); ++i) { 
        if(map_matched_trajectories_[i].size() >= 1 &&
           osm_trajectories_[i].size() >= 1){
            float dist = 0.0f;
            vector<road_graph_vertex_descriptor>& our_traj = map_matched_trajectories_[i];
            vector<road_graph_vertex_descriptor>& osm_traj = osm_trajectories_[i];
            for (size_t j = 0; j < our_traj.size(); ++j) { 
                RoadPt& our_pt = road_graph_[our_traj[j]].pt;
                float min_dist = POSITIVE_INFINITY;
                for (size_t k = 0; k < osm_traj.size(); ++k) { 
                    RoadPt& osm_pt = osm_graph_[osm_traj[k]].pt;
                    float d = roadPtDistance(our_pt, osm_pt);
                    if(d < min_dist){
                        min_dist = d;
                    }
                } 
                if(dist < min_dist)
                    dist = min_dist;
            } 
            h_dist.emplace_back(dist);
        }
    }
    // Write results to file
    ofstream output;
    output.open("hausdorff_dist.txt");
    if (output.fail()) {
        return;
    }
    output << h_dist.size() << endl;
    for (size_t i = 0; i < h_dist.size(); ++i) {
        output << h_dist[i] << endl;
    }
    output.close();
}

void RoadGenerator::precisionRecall(){
    int n_iter = trajectories_->data()->size() / 100;
    if(n_iter < 100)
        n_iter = 100;
    cout << "n_iter = " << n_iter << endl;
    float start_d = 5.0f;
    float end_d = 30.0f;
    float delta_h = 30.0f;
    float R = 100.0f;
    printf("d, precision, recall, F-score\n");
    for(float d = start_d; d < end_d + 0.1f; d += 5.0f){
        float cum_precision = 0.0f;
        float cum_recall = 0.0f;
        int count = 0;
        int i_iter = 0;
        int loc_idx = -1;
        int closest_pt_in_our_map = -1;
        int closest_pt_in_osm_map = -1;
        while(true){
            while(true){
                closest_pt_in_our_map = -1;
                closest_pt_in_osm_map = -1;
                bool has_osm_map = false;
                bool has_our_map = false;
                loc_idx = rand() % trajectories_->data()->size();
                PclPoint& pt = trajectories_->data()->at(loc_idx);
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                osmMap_->map_search_tree()->radiusSearch(pt, d, k_indices, k_dist_sqrs);
                float cur_min_dist_sqrs = POSITIVE_INFINITY;
                for (size_t i = 0; i < k_indices.size(); ++i) { 
                    PclPoint& map_pt = osmMap_->map_point_cloud()->at(k_indices[i]);
                        has_osm_map = true;
                        if(cur_min_dist_sqrs > k_dist_sqrs[i]){
                            closest_pt_in_osm_map = k_indices[i];
                            cur_min_dist_sqrs = k_dist_sqrs[i];
                        }
                }
                int heading = -1;
                if(has_osm_map){
                    auto es = in_edges(closest_pt_in_osm_map, osm_graph_);
                    for(auto eit = es.first; eit != es.second; ++eit){
                        if(osm_graph_[*eit].length > 1.0f){
                            road_graph_vertex_descriptor source_v = source(*eit, osm_graph_);
                            RoadPt& source_r_pt = osm_graph_[source_v].pt;
                            RoadPt& target_r_pt = osm_graph_[closest_pt_in_osm_map].pt;
                            heading = vector2dToHeading(Eigen::Vector2d(target_r_pt.x - source_r_pt.x,
                                                                        target_r_pt.y - source_r_pt.y));
                            break;
                        }
                    }
                    if(heading == -1){
                        auto es = out_edges(closest_pt_in_osm_map, osm_graph_);
                        for(auto eit = es.first; eit != es.second; ++eit){
                            if(osm_graph_[*eit].length > 1.0f){
                                road_graph_vertex_descriptor target_v = target(*eit, osm_graph_);
                                RoadPt& source_r_pt = osm_graph_[closest_pt_in_osm_map].pt;
                                RoadPt& target_r_pt = osm_graph_[target_v].pt;
                                heading = vector2dToHeading(Eigen::Vector2d(target_r_pt.x - source_r_pt.x,
                                                                            target_r_pt.y - source_r_pt.y));
                                break;
                            }
                        }
                    }
                }
                if(heading == -1)
                    continue;
                // on our map
                road_point_search_tree_->radiusSearch(pt, d, k_indices, k_dist_sqrs);
                cur_min_dist_sqrs = POSITIVE_INFINITY;
                for (size_t i = 0; i < k_indices.size(); ++i) { 
                    PclPoint& our_pt = road_points_->at(k_indices[i]);
                    if(abs(deltaHeading1MinusHeading2(heading, our_pt.head)) < delta_h){
                        has_our_map = true;
                        if(cur_min_dist_sqrs > k_dist_sqrs[i]){
                            closest_pt_in_our_map = indexed_roads_[our_pt.id_trajectory][our_pt.id_sample];
                            cur_min_dist_sqrs = k_dist_sqrs[i];
                        }
                    }
                } 
                k_indices.clear();
                k_dist_sqrs.clear();
                osmMap_->map_search_tree()->radiusSearch(pt, d, k_indices, k_dist_sqrs);
                cur_min_dist_sqrs = POSITIVE_INFINITY;
                for (size_t i = 0; i < k_indices.size(); ++i) { 
                    PclPoint& map_pt = osmMap_->map_point_cloud()->at(k_indices[i]);
                        has_osm_map = true;
                        if(cur_min_dist_sqrs > k_dist_sqrs[i]){
                            closest_pt_in_osm_map = k_indices[i];
                            cur_min_dist_sqrs = k_dist_sqrs[i];
                        }
                }
                if(has_our_map && has_osm_map)
                    break;
            }
            PclPoint& pt = trajectories_->data()->at(loc_idx);
            // Shortest path tree
            feature_vertices_.clear();
            feature_colors_.clear();
            feature_vertices_.emplace_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_DEBUG));
            feature_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::BLUE));
            vector<road_graph_vertex_descriptor> p(num_vertices(road_graph_));
            vector<float> dist(num_vertices(road_graph_));
            dijkstra_shortest_paths(road_graph_, 
                                    closest_pt_in_our_map, 
                                    predecessor_map(&p[0]).
                                    distance_map(&dist[0]).
                                    weight_map(get(&RoadGraphEdge::length, road_graph_)));
            vector<road_graph_vertex_descriptor> p1(num_vertices(osm_graph_));
            vector<float> dist1(num_vertices(osm_graph_));
            dijkstra_shortest_paths(osm_graph_, 
                                    closest_pt_in_osm_map, 
                                    predecessor_map(&p1[0]).
                                    distance_map(&dist1[0]).
                                    weight_map(get(&RoadGraphEdge::length, osm_graph_)));
            PclPointCloud::Ptr marble_points(new PclPointCloud);
            PclSearchTree::Ptr marble_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
            PclPointCloud::Ptr hole_points(new PclPointCloud);
            PclSearchTree::Ptr hole_point_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
            for(int i = 0; i < dist.size(); ++i){
                if(dist[i]<R){
                    RoadPt& r_pt = road_graph_[i].pt;
                    feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
                    feature_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
                    PclPoint pt;
                    pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                    marble_points->push_back(pt);
                }
            }
            for(int i = 0; i < dist1.size(); ++i){
                if(dist1[i]<R){
                    RoadPt& r_pt = osm_graph_[i].pt;
                    feature_vertices_.emplace_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_DEBUG));
                    feature_colors_.emplace_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
                    PclPoint pt;
                    pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
                    hole_points->push_back(pt);
                }
            }
            if(marble_points->size()>0)
                marble_point_search_tree->setInputCloud(marble_points);
            else{
                continue;
            }
            if(hole_points->size()>0)
                hole_point_search_tree->setInputCloud(hole_points);
            else{
                continue;
            }
            int n_matched_marble = 0;
            for (size_t i = 0; i < marble_points->size(); ++i) { 
                PclPoint& m_pt = marble_points->at(i);
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                hole_point_search_tree->radiusSearch(m_pt, d, k_indices, k_dist_sqrs);
                if(k_indices.size() > 0)
                    n_matched_marble++;
            } 
            int n_matched_hole = 0;
            for (size_t i = 0; i < hole_points->size(); ++i) { 
                PclPoint& h_pt = hole_points->at(i);
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                marble_point_search_tree->radiusSearch(h_pt, d, k_indices, k_dist_sqrs);
                if(k_indices.size() > 0)
                    n_matched_hole++;
            }
            float precision = static_cast<float>(n_matched_marble) / marble_points->size();
            float recall = static_cast<float>(n_matched_hole) / hole_points->size();
            cum_precision += precision;
            cum_recall += recall;
            count++;
            if(count >= n_iter){
                break;
            }
        }
        float overall_precision = cum_precision / count;
        float overall_recall = cum_recall / count;
        float F_score = 2.0f * overall_precision * overall_recall / (overall_precision + overall_recall);
        printf("[%.1f, %.4f, %.4f, %.4f],\n", d, overall_precision, overall_recall, F_score);
    }
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
        glLineWidth(10.0);
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
        glPointSize(15.0f);
        glDrawArrays(GL_POINTS, 0, points_to_draw_.size());
    }

    // Draw generated road map
    if(show_generated_map_){
        if(generated_map_render_mode_ == GeneratedMapRenderingMode::realistic){
            // Draw line loops
            for(int i = 0; i < generated_map_triangle_strips_.size(); ++i){
                vector<Vertex> &a_loop = generated_map_triangle_strips_[i];
                vector<Color> &a_loop_color = generated_map_triangle_strip_colors_[i];
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
                glDrawArrays(GL_TRIANGLE_STRIP, 0, a_loop.size());
            }
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
            
            // Draw links
            QOpenGLBuffer new_buffer;
            new_buffer.create();
            new_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_buffer.bind();
            new_buffer.allocate(&generated_map_links_[0], 3*generated_map_links_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            QOpenGLBuffer new_color_buffer;
            new_color_buffer.create();
            new_color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_color_buffer.bind();
            new_color_buffer.allocate(&generated_map_link_colors_[0], 4*generated_map_link_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            glLineWidth(8.0);
            glDrawArrays(GL_LINES, 0, generated_map_links_.size()); 
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

            // Draw links
            QOpenGLBuffer new_buffer1;
            new_buffer1.create();
            new_buffer1.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_buffer1.bind();
            new_buffer1.allocate(&generated_map_links_[0], 3*generated_map_links_.size()*sizeof(float));
            shader_program_->setupPositionAttributes();
            
            QOpenGLBuffer new_color_buffer1;
            new_color_buffer1.create();
            new_color_buffer1.setUsagePattern(QOpenGLBuffer::StaticDraw);
            new_color_buffer1.bind();
            new_color_buffer1.allocate(&generated_map_link_colors_[0], 4*generated_map_link_colors_.size()*sizeof(float));
            shader_program_->setupColorAttributes();
            glLineWidth(8.0);
            glDrawArrays(GL_LINES, 0, generated_map_links_.size()); 

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
    
    has_unexplained_gps_pts_ = false;
    min_road_length_ = 50.0f;
    unexplained_gps_pt_idxs_.clear();

    point_cloud_->clear();
    simplified_traj_points_->clear();
    grid_points_->clear();
    grid_votes_.clear();

    road_pieces_.clear(); 
    indexed_roads_.clear();
    max_road_label_ = 0;
    max_junc_label_ = 0;
    cur_num_clusters_ = 0;

    road_graph_valid_ = false;
    road_graph_.clear(); 
    road_points_->clear(); 
    map_matched_trajectories_.clear();

    osm_graph_.clear();
    osm_map_valid_ = false;
    osm_trajectories_.clear();
    
    feature_vertices_.clear(); 
    feature_colors_.clear();
    lines_to_draw_.clear();
    line_colors_.clear();
    points_to_draw_.clear();
    point_colors_.clear();

    // Clear generated map
    generated_map_points_.clear();
    generated_map_point_colors_.clear();
    generated_map_triangle_strips_.clear();
    generated_map_triangle_strip_colors_.clear();
    generated_map_lines_.clear();
    generated_map_line_colors_.clear();
    generated_map_links_.clear();
    generated_map_link_colors_.clear();

    tmp_ = 0;
    test_i_ = 0;
    traj_idx_to_show_ = 0;
}

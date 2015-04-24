#include "symbol.h"

Symbol::Symbol(const SymbolType type)
    :type_(type)
{
}

/////////////////////////////////////////////////////////////////
//                      Road Symbol
/////////////////////////////////////////////////////////////////
RoadSymbol::RoadSymbol(): Symbol(ROAD){
    // Initialize Attributes
    center_line_.clear();
    center_pt_visited_.clear();
    
    clearGPSInfo();
    
    start_state_    = QUERY_INIT;
    end_state_      = QUERY_INIT;
    
    lane_width_     =   3.7; // in meters
}

bool RoadSymbol::containsPt(int pt_idx){
    if(covered_pts_.find(pt_idx) != covered_pts_.end()){
        return true;
    }
    else{
        return false;
    }
}

pair<bool, bool> RoadSymbol::containsTraj(int traj_idx){
    pair<bool, bool> result(false, false);
    
    float score_range = cur_max_traj_score_ - cur_min_traj_score_;
    float ratio = 0.1f;
    float min_threshold = cur_min_traj_score_ + ratio * score_range;
    
    if(covered_traj_scores_[traj_idx] < min_threshold){
        return result;
    }
    
    if(covered_trajs_.find(traj_idx) != covered_trajs_.end()){
        result.first = true;
        if (covered_traj_aligned_scores_[traj_idx] >= covered_traj_unaligned_scores_[traj_idx]) {
            result.second = true;
        }
    }
    return result;
}

void RoadSymbol::insertPt(int pt_idx,
                          float pt_timestamp,
                          float probability,
                          int pt_traj_id,
                          bool is_aligned){
    float min_probability = 0.8f;
    if(probability < min_probability){
        return;
    }
    
    bool has_this_traj = true;
    if(covered_trajs_.find(pt_traj_id) == covered_trajs_.end()){
        has_this_traj = false;
        
        covered_trajs_.insert(pt_traj_id);
        covered_traj_scores_[pt_traj_id] = 0.0f;
        covered_traj_aligned_scores_[pt_traj_id] = 0.0f;
        covered_traj_unaligned_scores_[pt_traj_id] = 0.0f;
        traj_min_ts_[pt_traj_id] = pt_timestamp;
        traj_max_ts_[pt_traj_id] = pt_timestamp;
    }
    
    if (covered_pts_.emplace(pt_idx).second) {
        covered_pt_scores_[pt_idx] = probability;
        covered_traj_scores_[pt_traj_id] += probability;
        if(has_this_traj){
            if (traj_min_ts_[pt_traj_id] > pt_timestamp) {
                traj_min_ts_[pt_traj_id] = pt_timestamp;
            }
            if (traj_max_ts_[pt_traj_id] < pt_timestamp) {
                traj_max_ts_[pt_traj_id] = pt_timestamp;
            }
        }
    }
    else{
        if(covered_pt_scores_[pt_idx] < probability){
            if(has_this_traj){
                covered_traj_scores_[pt_traj_id] -= covered_pt_scores_[pt_idx];
                covered_traj_scores_[pt_traj_id] += probability;
            }
            else{
                covered_traj_scores_[pt_traj_id] += probability;
            }
            covered_pt_scores_[pt_idx] = probability;
        }
    }
    
    if(covered_traj_scores_[pt_traj_id] < cur_min_traj_score_){
        cur_min_traj_score_ = covered_traj_scores_[pt_traj_id];
    }
    
    if(covered_traj_scores_[pt_traj_id] > cur_max_traj_score_){
        cur_max_traj_score_ = covered_traj_scores_[pt_traj_id];
    }
    
    if(is_aligned){
        covered_traj_aligned_scores_[pt_traj_id] += probability;
    }
    else{
        covered_traj_unaligned_scores_[pt_traj_id] += probability;
    }
}

bool RoadSymbol::getDrawingVertices(std::vector<Vertex> &v){
    // true: regular drawing
    // false: query_init drowing mode
    bool return_value = true;
    if(start_state_ == QUERY_INIT || end_state_ == QUERY_INIT){
        return_value = false;
    }
    
    v.clear();
    if (center_line_.size() < 1) {
        return return_value;
    }
   
    if(return_value){
        // Regular drawing mode, return closed polygon
        for (size_t i = 0; i < center_line_.size(); ++i) {
            RoadPt& r_pt = center_line_[i];
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5f * r_pt.n_lanes * lane_width_ * Eigen::Vector2f(-1*direction[1], direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
        }
        
        for (int i = center_line_.size() - 1; i >= 0; --i) {
            RoadPt& r_pt = center_line_[i];
            
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5 * r_pt.n_lanes * lane_width_ * Eigen::Vector2f(direction[1], -1.0f * direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
        }
    }
    else{
        if (center_line_.size() == 1) {
            for (size_t i = 0; i < center_line_.size(); ++i) {
                RoadPt& r_pt = center_line_[i];
                v.push_back(SceneConst::getInstance().normalize(r_pt.x, r_pt.y, Z_ROAD));
            }
        }
    }
    
    return return_value;
}

/////////////////////////////////////////////////////////////////
//                      Branch Symbol
/////////////////////////////////////////////////////////////////
JunctionSymbol::JunctionSymbol(): Symbol(JUNCTION){
}

void JunctionSymbol::getDrawingVertices(Vertex &v){
    Vertex tmp = SceneConst::getInstance().normalize(loc_[0], loc_[1], Z_ROAD);
    v.x = tmp.x;
    v.y = tmp.y;
    v.z = tmp.z;
}
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

void RoadSymbol::insertPt(int pt_idx,
                          float probability){
    float min_probability = 0.8f;
    if(probability < min_probability){
        return;
    }
    
    if (covered_pts_.emplace(pt_idx).second) {
        covered_pt_scores_[pt_idx] = probability;
    }
    else{
        if(covered_pt_scores_[pt_idx] < probability){
            covered_pt_scores_[pt_idx] = probability;
        }
    }
}

bool RoadSymbol::getDrawingVertices(std::vector<Vertex> &v){
    // true: regular drawing
    // false: query_init drowing mode
    bool return_value = true;
    //if(start_state_ == QUERY_INIT || end_state_ == QUERY_INIT){
        //return_value = false;
    //}
    
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

bool JunctionSymbol::getDrawingVertices(std::vector<Vertex>& v){
    v.clear(); 
    Vertex tmp = SceneConst::getInstance().normalize(loc_[0], loc_[1], Z_ROAD);
    v.push_back(tmp);
    return true;
}

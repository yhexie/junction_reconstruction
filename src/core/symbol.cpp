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
    center_.clear();
    covered_pts_.clear();
    
    start_state_    = QUERY_INIT;
    end_state_      = QUERY_INIT;
    parent_symbol_  = NULL;
    child_symbol_   = NULL;
    
    lane_width_     =   3.7; // in meters
}

bool RoadSymbol::getDrawingVertices(std::vector<Vertex> &v){
    // true: regular drawing
    // false: query_init drowing mode
    bool return_value = true;
    if(start_state_ == QUERY_INIT || end_state_ == QUERY_INIT){
        return_value = false;
    }
    
    v.clear();
    if (center_.size() < 1) {
        return return_value;
    }
   
    if(return_value){
        // Regular drawing mode, return closed polygon
        for (size_t i = 0; i < center_.size(); ++i) {
            RoadPt& r_pt = center_[i];
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5f * r_pt.n_lanes * lane_width_ * Eigen::Vector2f(-1*direction[1], direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
        }
        
        for (int i = center_.size() - 1; i >= 0; --i) {
            RoadPt& r_pt = center_[i];
            
            Eigen::Vector2d direction = headingTo2dVector(r_pt.head);
            
            Eigen::Vector2f perp = 0.5 * r_pt.n_lanes * lane_width_ * Eigen::Vector2f(direction[1], -1.0f * direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(r_pt.x, r_pt.y) + perp;
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
        }
    }
    else{
        if (center_.size() == 1) {
            for (size_t i = 0; i < center_.size(); ++i) {
                RoadPt& r_pt = center_[i];
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
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
    
    n_lanes_        =   1;
    is_oneway_      =   true;
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
            float heading_in_radius = center_[i].z * PI / 180.0f;
            Eigen::Vector2f direction(cos(heading_in_radius), sin(heading_in_radius));
            Eigen::Vector2f perp = 0.5 * n_lanes_ * lane_width_ * Eigen::Vector2f(-1*direction[1], direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(center_[i].x, center_[i].y) + perp;
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
        }
        
        for (int i = center_.size() - 1; i >= 0; --i) {
            float heading_in_radius = center_[i].z * PI / 180.0f;
            Eigen::Vector2f direction(cos(heading_in_radius), sin(heading_in_radius));
            Eigen::Vector2f perp = 0.5 * n_lanes_ * lane_width_ * Eigen::Vector2f(direction[1], -1.0f * direction[0]);
            Eigen::Vector2f v1 = Eigen::Vector2f(center_[i].x, center_[i].y) + perp;
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
        }
    }
    else{
        if (center_.size() == 1) {
            float heading_in_radius = center_[0].z * PI / 180.0f;
            Eigen::Vector2f direction(cos(heading_in_radius), sin(heading_in_radius));
            Eigen::Vector2f v1 = Eigen::Vector2f(center_[0].x, center_[0].y) + 25.0f * direction;
            v.push_back(SceneConst::getInstance().normalize(center_[0].x, center_[0].y, Z_ROAD));
            v.push_back(SceneConst::getInstance().normalize(v1.x(), v1.y(), Z_ROAD));
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
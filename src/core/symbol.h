#ifndef SYMBOL_H_
#define SYMBOL_H_

#include <ostream>
#include <cstring>
#include "common.h"
#include <Eigen/Dense>
#include <vector>
#include <set>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using std::ostream;
using namespace std;
using namespace boost;

typedef adjacency_list<vecS, listS, undirectedS> symbol_graph_t;
typedef graph_traits<symbol_graph_t>::vertex_descriptor vertex_t;

enum SymbolType{
    ROAD = 0,
    JUNCTION = 1
};

enum QueryState {
    UNASSIGNED,
    FAIL,
    SUCCEED
};

class Symbol
{
public:
    Symbol(const SymbolType type);
    
    virtual ~Symbol() {}
    
    virtual const char* symbolName() const=0;
    
    friend ostream& operator<<(ostream& os, const Symbol* m){
        os << m->symbolName();
        return os;
    }
    
    const SymbolType& type() const { return type_; }
    vertex_t&       vertex_descriptor() { return vertex_descriptor_; }
    
protected:
    const SymbolType                type_;
    vertex_t                        vertex_descriptor_;
};

/////////////////////////////////////////////////////////////////
//                      Road Symbol
/////////////////////////////////////////////////////////////////
enum RoadSymbolState{
    QUERY_INIT = 0,
    QUERY_Q = 1,
    TERMINAL = 2
};

class RoadSymbol : public Symbol
{
public:
    RoadSymbol();
    
    ~RoadSymbol() {};
    
    const char* symbolName() const
    { return "R"; }
    
    // Access & update attributes
    const vector<RoadPt>&       centerLine() { return center_line_; }
    vector<bool>&               centerPtVisited() { return center_pt_visited_; }
   
    const map<int, float>&      trajMinTs() { return traj_min_ts_; }
    const map<int, float>&      trajMaxTs() { return traj_max_ts_; }
    
    
    void addRoadPtAtEnd(RoadPt &pt) {
        center_line_.push_back(pt);
        center_pt_visited_.push_back(false);
    }
    
    void addRoadPtAtFront(RoadPt &pt) {
        center_line_.insert(center_line_.begin(), pt);
        center_pt_visited_.insert(center_pt_visited_.begin(), false);
    }
    
    void clearCenter(){
        center_line_.clear();
        center_pt_visited_.clear();
        clearGPSInfo();
    }
    
    void clearGPSInfo(){
        covered_pts_.clear();
        covered_pt_scores_.clear();
        covered_trajs_.clear();
        cur_max_traj_score_ = 0.0f;
        cur_min_traj_score_ = 1e10;
        covered_traj_scores_.clear();
        covered_traj_aligned_scores_.clear();
        covered_traj_unaligned_scores_.clear();
        traj_min_ts_.clear();
        traj_max_ts_.clear();
    }
    
    bool isOneway() {
        if(center_line_.size() == 0){
            return false;
        }
        else{
            return center_line_[0].is_oneway;
        }
    }
    
    bool containsPt(int);
    
    pair<int, int> stats() {
        pair<int, int> result;
        result.first = covered_pts_.size();
        result.second = covered_trajs_.size();
        return result;
    }
    
    pair<bool, bool> containsTraj(int);
    
    void insertPt(int pt_idx,
                  float pt_timestamp,
                  float probability,
                  int pt_traj_id,
                  bool is_aligned);
    
    RoadSymbolState&            startState() { return start_state_; }
    RoadSymbolState&            endState()  { return end_state_; }
    
    // Rendering
    bool                        getDrawingVertices(std::vector<Vertex> &v);
    
private:
    // Road Attributes
    RoadSymbolState                 start_state_;
    RoadSymbolState                 end_state_;
    
    vector<RoadPt>                  center_line_;
    vector<bool>                    center_pt_visited_; // Used as an indicator to speed up computing GPS point on road
    
    float                           lane_width_;
    
    set<int>                        covered_pts_; // index of the point contained by this road
    map<int, float>                 covered_pt_scores_; // prpbability of each point belongs to this road
    set<int>                        covered_trajs_; // index of the trajectory covered by this road
    map<int, float>                 covered_traj_scores_; // sum of the prpbability of points belongs
   
    float                           cur_max_traj_score_;
    float                           cur_min_traj_score_;
    
    map<int, float>                 covered_traj_aligned_scores_; // all true for oneway, for twoway road, this mark whether the traj is in opposite direction to the center line.
    map<int, float>                 covered_traj_unaligned_scores_; // all true for oneway, for twoway road, this mark whether the traj is in opposite direction to the center line.
    map<int, float>                 traj_min_ts_; // Each trajectory has a min and max time on this road
    map<int, float>                 traj_max_ts_;
};

/////////////////////////////////////////////////////////////////
//                      Junction Symbol
/////////////////////////////////////////////////////////////////
class JunctionSymbol : public Symbol
{
public:
    JunctionSymbol();
    
    ~JunctionSymbol() {};
    
    void getDrawingVertices(Vertex &v);
    
    const char*                     symbolName() const { return "J"; }
    Eigen::Vector2d&                loc() { return loc_; }
    vector<RoadSymbol*>&            children() { return children_; }
    set<pair<int, int>>&            connection() { return connection_; }
private:
    // Branch Attributes
    Eigen::Vector2d                 loc_; // center of the branch
    vector<RoadSymbol*>             children_; // a the roads associated with this branch
    set<pair<int, int>>             connection_; // the connectivity
};

/////////////////////////////////////////////////////////////////
//                      Query Init Symbol
/////////////////////////////////////////////////////////////////
class QueryInitSymbol : public Symbol
{
public:
    QueryInitSymbol(RoadSymbol *, QueryState init_state=UNASSIGNED);
    
    ~QueryInitSymbol() {};
   
    RoadSymbol* getRoad() { return road_; }
    QueryState& state() { return state_; }
    
    const char* symbolName() const
    { return "?I"; }
    
private:
    QueryState      state_;
    RoadSymbol*     road_;
};

/////////////////////////////////////////////////////////////////
//                      Terminal Symbol
/////////////////////////////////////////////////////////////////
class TerminalSymbol : public Symbol
{
public:
    TerminalSymbol();
    
    ~TerminalSymbol() {};
    
    const char* symbolName() const
    { return "E"; }
};

#endif //SYMBOL_H_
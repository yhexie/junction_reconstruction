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

    virtual bool getDrawingVertices(std::vector<Vertex> &v) { return true; }
    
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
    const set<int>& coveredPts() { return covered_pts_; }
    const map<int, float>& coveredPtScores() { return covered_pt_scores_; }
    
    pair<int, int> stats() {
        pair<int, int> result;
        result.first = covered_pts_.size();
        return result;
    }
    
    pair<bool, bool> containsTraj(int);
    
    void insertPt(int pt_idx,
                  float probability);
    
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
};

/////////////////////////////////////////////////////////////////
//                      Junction Symbol
/////////////////////////////////////////////////////////////////
class JunctionSymbol : public Symbol
{
public:
    JunctionSymbol();
    
    ~JunctionSymbol() {};
    
    bool getDrawingVertices(std::vector<Vertex> &v);
    
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

#endif //SYMBOL_H_

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
    vector<RoadPt>&             center() { return center_; }
    
    bool                        isOneway() {
        if(center_.size() == 0){
            return false;
        }
        else{
            return center_[0].is_oneway;
        }
    }
    
    set<int>&                   coveredPts()  { return covered_trajs_; }
    map<int, set<int> >&        coveredTrajs() { return covered_pts_; }
    
    RoadSymbolState&            startState() { return start_state_; }
    RoadSymbolState&            endState()  { return end_state_; }
    void                        setParentSymbol(Symbol* parent) { parent_symbol_ = parent; }
    void                        setChildSymbol(Symbol* child) { child_symbol_ = child; }
    
    // Rendering
    bool                        getDrawingVertices(std::vector<Vertex> &v);
    
private:
    // Road Attributes
    RoadSymbolState                 start_state_;
    RoadSymbolState                 end_state_;
    
    Symbol*                         parent_symbol_;
    Symbol*                         child_symbol_;
    
    vector<RoadPt>                  center_;
    float                           lane_width_;
    
    set<int>                        covered_trajs_; // index of the point contained by this road
    map<int, set<int> >             covered_pts_; // index of the covered trajectory by this road
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
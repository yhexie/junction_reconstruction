#ifndef DISTACE_GRAPH_H_
#define DISTACE_GRAPH_H_

#include <QObject>
#include "common.h"
#include "color_map.h"
#include "renderable.h"

#include <iostream>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>

using namespace std;
using namespace boost;

struct location;

// BOOST GRAPH LIBRARY
typedef adjacency_list<listS, vecS, directedS, no_property, property<edge_weight_t, float> > graph_t;
typedef property_map<graph_t, edge_weight_t>::type WeightMap;
typedef pair<int, int> edge;
typedef pair<int, int> key_type;

class DistanceGraph{
public:
    DistanceGraph();
    ~DistanceGraph();
    
    PclPointCloud::Ptr& data(void) {return vertices_;}
    PclSearchTree::Ptr& tree(void) {return tree_;}
    
    int nVertices(void) {return vertices_->size();}
    int nEdges(void) {return edge_idxs_.size()/2;}
    vector<int> &edge_idxs() {return edge_idxs_;}
  
    void samplePointCloud(float grid_size);
    void computeGraphUsingGpsPointCloud(PclPointCloud::Ptr &data, PclSearchTree::Ptr &gps_pointcloud_search_tree, PclPointCloud::Ptr &vertices, PclSearchTree::Ptr &vertice_tree, float grid_size);
    void shortestPath(int start, int goal, vector<int> &path);
    float Distance(PclPoint &pt1, PclPoint &pt2);
    
private:
    float                               grid_size_;
    map<key_type, float>                computed_distances;
    graph_t                             *g_;
    PclPointCloud::Ptr                  vertices_;
    PclSearchTree::Ptr                  tree_;
    
    PclPointCloud::Ptr                  gps_pointcloud_;
    PclSearchTree::Ptr                  gps_pointcloud_search_tree_;
    
    vector<location>                    locations_;
    vector<int>                         edge_idxs_;
};

#endif //DISTANCE_GRAPH_
#ifndef GRAPH_H_
#define GRAPH_H_

#include <QObject>

#include "common.h"
#include "color_map.h"
#include "segment.h"
#include "renderable.h"

#include <iostream>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace boost;

struct location;

// BOOST GRAPH LIBRARY
typedef adjacency_list<listS, vecS, directedS, no_property, property<edge_weight_t, float> > graph_t;
typedef property_map<graph_t, edge_weight_t>::type WeightMap;
typedef pair<int, int> edge;

class Graph : public Renderable{
public:
    Graph(QObject *parent);
    ~Graph();
    
    void draw();
    void prepareForVisualization(QVector4D bound_box);
    
    PclPointCloud::Ptr& data(void) {return vertices_;}
    PclSearchTree::Ptr& tree(void) {return tree_;}
    
    int nVertices(void) {return vertices_->size();}
    int nEdges(void) {return edge_idxs_.size()/2;}
   
    void updateGraphUsingSamplesAndGpsPointCloud(PclPointCloud::Ptr &, PclSearchTree::Ptr &, PclPointCloud::Ptr &, PclSearchTree::Ptr &);
    void updateGraphUsingDBSCANClustersAndSamples(PclPointCloud::Ptr &, PclSearchTree::Ptr &, PclPointCloud::Ptr &, PclSearchTree::Ptr &, vector<vector<int>> &clusterSamples);
    void updateGraphUsingSamplesAndSegments(PclPointCloud::Ptr &, PclSearchTree::Ptr &, vector<Segment> &, vector<vector<int>> &, vector<vector<int>> &, PclPointCloud::Ptr &, PclSearchTree::Ptr &);
    void updateGraphUsingDescriptor(PclPointCloud::Ptr &cluster_centers, PclSearchTree::Ptr &cluster_center_search_tree, cv::Mat *descriptors, vector<int> &cluster_popularity, PclPointCloud::Ptr &, PclSearchTree::Ptr &);
    
    // Shortest path from source and destination
    bool shortestPath(int , int , vector<int>&);
    
    // Shortest path interpolation
    void shortestPathInterpolation(vector<int> &query_path, vector<int> &result_path);
    
    // Graph dynamic time warping distance
    float simpleSegDistance(Segment &seg1, Segment &seg2);
    float SegDistance(vector<int> &query_path1, vector<int> &query_path2);
    float DTWDistance(vector<int> &path1, vector<int> &path2);
    float Distance(int, int, bool, bool, bool, bool);
    
    // For visualization
    vector<vector<int>> &pathsToDraw(void) { return paths_to_draw_; }
    void insertPathToDraw(vector<int> &);
    void clearPathsToDraw(void);
    void drawShortestPathInterpolationFor(vector<int> &query_path);
    
private:
    graph_t                             *g_;
    PclPointCloud::Ptr                  vertices_;
    PclSearchTree::Ptr                  tree_;
    
    PclPointCloud::Ptr                  gps_pointcloud_;
    PclSearchTree::Ptr                  gps_pointcloud_search_tree_;
    
    vector<location>                    locations_;
    
    // For visualization
    float                               scale_factor_;
    float                               line_width_;
    float                               point_size_;
    vector<Vertex>                      normalized_vertices_locs_;
    vector<Vertex>                      edge_direction_vertices_;
    vector<int>                         edge_direction_idxs_;
    vector<Color>                       edge_direction_colors_;
    vector<Color>                       vertex_colors_;
    vector<int>                         edge_idxs_;
    vector<vector<int>>                 paths_to_draw_;
};

#endif //GRAPH_
//
//  road_generator.h
//  junction_reconstruction
//
//  Created by Chen Chen on 1/2/15.
//

#ifndef __junction_reconstruction__road_generator__
#define __junction_reconstruction__road_generator__

#include <stdio.h>
#include <vector>
#include "renderable.h"
#include "trajectories.h"
#include <pcl/common/centroid.h>
#include <pcl/search/impl/flann_search.hpp>
#include "features.h"
#include "common.h"
#include <cmath>
#include <random>
#include <Eigen/Dense>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/connected_components.hpp>

using namespace std;

enum class RoadGraphNodeType{
    road = 0,
    junction = 1
};

enum class RoadGraphEdgeType{
    normal = 0,
    linking = 1,
    auxiliary = 2 // only used for connecting edges, the final graph should not contain any auxiliary edges
};

struct RoadGraphNode{
    // Common parameters
    RoadGraphNodeType   type = RoadGraphNodeType::road;

    // Location of the node, if it's road, also contains heading, and width info
    RoadPt              pt;

    // Parameters for road node. Valid if type == RoadGraphNodeType::road
    int                 road_label = -1; // The index of corresponding road. NOTE: even if the node is a junction, road_label is still valid, which denotes the continuity of a road through a junction.
    int                 idx_in_road = -1; // The index in the corresponding road
    bool                inferred = false; // if this is true, meaning this road node is inferred which could be removed later.

    // Parameters for road node. Valid if type == RoadGraphNodeType::junction
    int                 junction_label = -1;     
    // auxiliary entries
    int                 cluster_id = -1; // used for clustering nodes to form junctions
};

struct RoadGraphEdge{
    RoadGraphEdgeType type = RoadGraphEdgeType::normal;
    float length = 1e9;

    // Parameter for RoadGraphEdgeType::linking
    int link_support = 0;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, RoadGraphNode, RoadGraphEdge> road_graph_t;
typedef boost::graph_traits<road_graph_t>::vertex_descriptor road_graph_vertex_descriptor;
typedef boost::graph_traits<road_graph_t>::edge_descriptor road_graph_edge_descriptor;

class RoadGenerator : public Renderable
{
public:
    RoadGenerator(QObject *parent, std::shared_ptr<Trajectories> trajectories = nullptr);
    ~RoadGenerator();
    
    void setTrajectories(std::shared_ptr<Trajectories>& new_trajectories) {
        trajectories_ = new_trajectories;
    }
    
    void applyRules();
    
    void pointBasedVoting();
    void computeVoteLocalMaxima();
    void pointBasedVotingVisualization();
    bool computeInitialRoadGuess();
    bool updateRoadPointCloud();
    bool addInitialRoad();
    void recomputeRoads();
    void connectRoads();
    
    void tmpFunc();
    
    // Trajectory - Road projection
    float shortestPath(int start, int target, vector<int>& path);
    void mapMatching(size_t traj_idx, vector<int>& projection);
    void partialMapMatching(size_t traj_idx, vector<int>& projection);    

    void computeGPSPointsOnRoad(const vector<RoadPt>& road,
                                set<int>& results); 

    // Grammar
    void production();
    void resolveQuery();
    
    // Util
    void localAdjustment();
    void updatePointCloud();
    
    // Road relationship
    void refit();
    
    // Rendering
    void draw();
    
    // Clear
    void clear();
    void cleanUp();
    
private:
    PclPointCloud::Ptr                            point_cloud_;
    PclSearchTree::Ptr                            search_tree_;
    
    PclPointCloud::Ptr                            simplified_traj_points_;
    PclSearchTree::Ptr                            simplified_traj_point_search_tree_;
    
    PclPointCloud::Ptr                            grid_points_; // a tmp point cloud for ease of use
    PclSearchTree::Ptr                            grid_point_search_tree_;
    vector<float>                                 grid_votes_;
    vector<bool>                                  grid_is_local_maximum_;
    
    std::shared_ptr<Trajectories>                 trajectories_;

    vector<vector<RoadPt>>                        road_pieces_;

    // Graph as the road network
    road_graph_t                                  road_graph_;
    PclPointCloud::Ptr                            road_points_;
    PclSearchTree::Ptr                            road_point_search_tree_;
    vector<vector<road_graph_vertex_descriptor>>  indexed_roads_;
    int                                           max_road_label_; // increase once a new road is created
    int                                           max_junc_label_;
    int                                           cur_num_clusters_;

    // For visualization
    vector<Vertex>                                feature_vertices_;
    vector<Color>                                 feature_colors_;
    
    // For DEBUG usage
    bool                                          debug_mode_;
    int                                           tmp_;
    vector<Vertex>                                lines_to_draw_;
    vector<Color>                                 line_colors_;
    vector<Vertex>                                points_to_draw_;
    vector<Color>                                 point_colors_;
    
    float                                         cur_cum_length;
    float                                         radius;
};

#endif /* defined(__junction_reconstruction__road_generator__) */

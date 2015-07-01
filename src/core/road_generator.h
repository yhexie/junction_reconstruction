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
    bool                is_valid = true; // Will remove the node if (is_valid == false);

    // Parameters for road node. Valid if type == RoadGraphNodeType::junction
    //int                 junction_label = -1;     
    // auxiliary entries
    int                 cluster_id = -1; // used for clustering nodes to form junctions
};

struct RoadGraphEdge{
    RoadGraphEdgeType type = RoadGraphEdgeType::normal;
    float length = POSITIVE_INFINITY;

    // Parameter for RoadGraphEdgeType::linking
    int link_support = 0;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, RoadGraphNode, RoadGraphEdge> road_graph_t;
typedef boost::graph_traits<road_graph_t>::vertex_descriptor road_graph_vertex_descriptor;
typedef boost::graph_traits<road_graph_t>::edge_descriptor road_graph_edge_descriptor;

struct Link{
    road_graph_vertex_descriptor source_vertex; // in road_graph_
    road_graph_vertex_descriptor target_vertex; // in road_graph_
    int source_related_road_idx; // in related_roads defined below
    int source_idx_in_related_roads; // in related roads defined below
    int target_related_road_idx; // in related roads defined below
    int target_idx_in_related_roads; // in related roads defined below
    bool is_bidirectional = false;
    int dh = 0;
    float length = 0.0f;
    float delta_speed = 0.0f;
};

enum class GeneratedMapRenderingMode{
    realistic = 0,
    skeleton = 1
};

class RoadGenerator : public Renderable
{
public:
    RoadGenerator(QObject *parent, std::shared_ptr<Trajectories> trajectories = nullptr);
    ~RoadGenerator();
    
    void setTrajectories(std::shared_ptr<Trajectories>& new_trajectories) {
        trajectories_ = new_trajectories;
    }

    void setOpenStreetMap(std::shared_ptr<OpenStreetMap>& new_osm){
        osmMap_ = new_osm;
        updateOsmGraph();
    }
    
    void updateOsmGraph();
    void pointBasedVoting(); 
    void computeVoteLocalMaxima();
    void pointBasedVotingVisualization();
    void estimateRoadWidthAndSpeed(vector<RoadPt>& road, float threshold);
    void adjustOppositeRoads();
    bool computeInitialRoadGuess();
    void computeUnexplainedGPSPoints();
    bool updateRoadPointCloud();
    bool addInitialRoad();
    void recomputeRoads();
    void computeJunctionClusters(vector<set<road_graph_vertex_descriptor>>& junc_clusters);
    void localAdjustments();
    void prepareGeneratedMap();
    void resolveLink(const Link& link);
    void connectRoads();
    void finalAdjustment();
    
    void tmpFunc();
    
    // Trajectory - Road projection
    float shortestPath(road_graph_vertex_descriptor source, 
                       road_graph_vertex_descriptor target, 
                       road_graph_t&                graph,              
                       vector<road_graph_vertex_descriptor>& path);
    float probOnRoad(const PclPoint& pt, 
                     const PclPoint& r_pt,
                     float sigma_w,
                     float sigma_l,
                     float sigma_h);
    bool mapMatching(size_t traj_idx, vector<road_graph_vertex_descriptor>& projection);
    bool mapMatchingToOsm(size_t traj_idx, vector<road_graph_vertex_descriptor>& projection);
    void partialMapMatching(size_t traj_idx, float search_radius, float cut_off_probability, vector<int>& projection);    

    void compareDistance();
    void showTrajectory();

    void computeGPSPointsOnRoad(const vector<RoadPt>& road,
                                set<int>& results); 

    // Evaluation
    void evaluationMapMatching(); 
    void evaluationMapMatchingToOsm();
    void computeHausdorffDistance();
    void precisionRecall();

    // Rendering
    void draw();
    void setShowGeneratedMap(bool v){
        show_generated_map_ = v;
    }

    void setGeneratedMapRenderMode(int v){
        switch (v) { 
            case 0: { 
                generated_map_render_mode_ = GeneratedMapRenderingMode::realistic;
            } 
            break; 
            case 1: {
                generated_map_render_mode_ = GeneratedMapRenderingMode::skeleton;
            }
        
            default: { 
            } 
            break; 
        } 
    }
    
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

    bool                                          has_unexplained_gps_pts_;
    vector<int>                                   unexplained_gps_pt_idxs_;
    
    std::shared_ptr<Trajectories>                 trajectories_;

    vector<vector<RoadPt>>                        road_pieces_;
    float                                         min_road_length_;

    // Graph as the road network
    bool                                          road_graph_valid_;
    road_graph_t                                  road_graph_;
    PclPointCloud::Ptr                            road_points_;
    PclSearchTree::Ptr                            road_point_search_tree_;
    vector<vector<road_graph_vertex_descriptor>>  indexed_roads_;
    int                                           max_road_label_; // increase once a new road is created
    int                                           max_junc_label_;
    int                                           cur_num_clusters_;

    // Evaluation
    vector<vector<road_graph_vertex_descriptor>>  map_matched_trajectories_;
    int                                           test_i_;

    // OpenStreetMap
    bool                                          osm_map_valid_;
    std::shared_ptr<OpenStreetMap>                osmMap_;
    road_graph_t                                  osm_graph_;
    vector<vector<road_graph_vertex_descriptor>>  osm_trajectories_;
    int                                           traj_idx_to_show_;

    // Visualize generated map
    bool                                          show_generated_map_;
    GeneratedMapRenderingMode                     generated_map_render_mode_;
    vector<Vertex>                                generated_map_points_;
    vector<Color>                                 generated_map_point_colors_; 
    vector<vector<Vertex>>                        generated_map_triangle_strips_;
    vector<vector<Color>>                         generated_map_triangle_strip_colors_;
    vector<Vertex>                                generated_map_lines_;
    vector<Color>                                 generated_map_line_colors_;
    vector<Vertex>                                generated_map_links_;
    vector<Color>                                 generated_map_link_colors_;

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

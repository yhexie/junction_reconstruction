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
#include "symbol.h"
#include "renderable.h"
#include "trajectories.h"
#include <pcl/common/centroid.h>
#include <pcl/search/impl/flann_search.hpp>
#include "features.h"
#include "common.h"
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace std;

class RoadGenerator : public Renderable
{
public:
    RoadGenerator(QObject *parent, std::shared_ptr<Trajectories> trajectories = nullptr);
    ~RoadGenerator();
    
    void setTrajectories(std::shared_ptr<Trajectories>& new_trajectories) {
        trajectories_ = new_trajectories;
        has_been_covered_.clear();
        if(trajectories_->data()->size() > 0){
            has_been_covered_.resize(trajectories_->data()->size(), false);
        }
    }
    
    void applyRules();
    
    void pointBasedVoting();
    void computeVoteLocalMaxima();
    void pointBasedVotingVisualization();
    bool computeInitialRoadGuess();
    bool addInitialRoad();
    
    void tmpFunc();
    
    // Trajectory - Road projection
    void updateGPSPointsOnRoad(std::shared_ptr<RoadSymbol> road);
    void getConsistentPointSetForRoad(std::shared_ptr<RoadSymbol> road,
                                      set<int>& candidate_point_set,
                                      vector<vector<int> >& segments,
                                      vector<float>& segment_scores);
    
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
    PclPointCloud::Ptr              point_cloud_;
    PclSearchTree::Ptr              search_tree_;
    
    PclPointCloud::Ptr              simplified_traj_points_;
    PclSearchTree::Ptr              simplified_traj_point_search_tree_;
    
    PclPointCloud::Ptr              grid_points_; // a tmp point cloud for ease of use
    PclSearchTree::Ptr              grid_point_search_tree_;
    vector<float>                   grid_votes_;
    vector<bool>                    grid_is_local_maximum_;
    
    map<vertex_t, std::shared_ptr<Symbol>>          graph_nodes_;
    symbol_graph_t                  symbol_graph_;
    std::shared_ptr<Trajectories>        trajectories_;
    vector<bool>                    has_been_covered_;

    vector<vector<RoadPt> >         initial_roads_;
    
    // For visualization
    vector<Vertex>                  feature_vertices_;
    vector<Color>                   feature_colors_;
    
    // For DEBUG usage
    int                             tmp_;
    vector<Vertex>                  lines_to_draw_;
    vector<Color>                   line_colors_;
    vector<Vertex>                  points_to_draw_;
    vector<Color>                   point_colors_;
    
    std::shared_ptr<RoadSymbol>          new_road;
    float                           cur_cum_length;
    float                           radius;
    int                             i_road;
};

#endif /* defined(__junction_reconstruction__road_generator__) */

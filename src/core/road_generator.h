//
//  road_generator.h
//  junction_reconstruction
//
//  Created by Chen Chen on 1/2/15.
//
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
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace boost;

//Used for polygonal line fitting

void polygonalFitting(vector<Vertex>& pts,
                      vector<Vertex>& results,
                      float& avg_dist);

void initialRoadFitting(vector<Vertex>& pts,
                        vector<Vertex>& results,
                        int max_n_vertices = 3);

void pointsToPolylineProjection(vector<Vertex>& pts,
                                vector<Vertex>& polyline,
                                vector<vector<int>>& assignment);

void pointsToPolylineProjectionWithHeading(vector<Vertex>& pts,
                                           vector<Vertex>& polyline,
                                           vector<vector<int>>& assignment);

float computeGnScoreAt(int idx,
                        vector<Vertex>& data,
                        vector<Vertex>& vertices, // the vertices of the poly line
                        vector<vector<int>>& assignment,
                        float lambda,
                        float r);

void computeGnApproxiGradientAt(int idx,
                                vector<Vertex>& data,
                                vector<Vertex>& vertices, // the vertices of the poly line
                                vector<vector<int>>& assignment,
                                float lambda,
                                float R,
                                Eigen::Vector2d& grad_dir);

void computeGnGradientAt(int idx,
                         vector<Vertex>& data,
                         vector<Vertex>& vertices, // the vertices of the poly line
                         vector<vector<int>>& assignment,
                         float lambda,
                         float R,
                         Eigen::Vector2d& grad_dir);

void computeDpPlusOneAt(int idx,
                        vector<Vertex>& vertices,
                        float R,
                        Eigen::Vector2d& d_pv_vi_plus_1);

void computeDpMinusOneAt(int idx,
                         vector<Vertex>& vertices,
                         float R,
                         Eigen::Vector2d& d_pv_vi_minus_1);

void computeDpAt(int idx,
                 vector<Vertex>& vertices,
                 float R,
                 Eigen::Vector2d& d_pv_vi);

void dividePointsIntoSets(vector<Vertex>& pts,
                          vector<Vertex>& center,
                          vector<vector<int>>& assignments,
                          map<int, float>& dist);

bool dividePointsIntoSets(set<int>& pts,
                          PclPointCloud::Ptr data,
                          vector<Vertex>& centerline,
                          vector<vector<int>>& assignments,
                          map<int, float>& dists,
                          bool with_checking = false,
                          bool is_oneway = true,
                          float dist_threshold = 15.f,
                          float heading_threshold = 15.0f,
                          int tolerance = 10);
struct RoadPt{
    float x;
    float y;
    int   head;
    bool  is_oneway;
    int   n_lanes;
};

class RoadGenerator : public Renderable
{
public:
    RoadGenerator(QObject *parent, Trajectories* trajectories = NULL);
    ~RoadGenerator();
    
    void setTrajectories(Trajectories* new_trajectories) {
        trajectories_ = new_trajectories;
        has_been_covered_.clear();
        if(trajectories_->data()->size() > 0){
            has_been_covered_.resize(trajectories_->data()->size(), false);
        }
    }
    
    // Tensor Voting
    void denseVoting(float sigma);
    void sparseVoting(float sigma);
    void tracing(float sigma);
    
    void applyRules();
    
    // Features & Prediction
    bool exportQueryInitFeatures(float radius, const string& filename);
    bool loadQueryInitFeatures(const string& filename);
    bool addQueryInitToString();
    void trace_road();
    void extend_road(int r_idx, vector<RoadPt>&, vector<bool>& mark_list, bool forward = true);
    
    void detectOverlappingRoadSeeds(vector<RoadSymbol*>& road_list, vector<vector<int>>& result);
    
    bool exportQueryQFeatures(const string& filename);
    bool loadQueryQPredictions(const string& filename);
    
    // Grammar
    void production();
    void resolveQuery();
    
    // Util
    void localAdjustment();
    void updatePointCloud();
    void detectRoadsToMerge(vector<vector<int>>& roads_to_merge);
    void mergeRoads(vector<vector<int>>& roads_to_merge);
    void getMergedRoadsFrom(vector<int>& candidate_roads, RoadSymbol* new_road);
    
    // Fitting
    bool fitARoadAt(PclPoint& loc, RoadSymbol* road, bool is_oneway, float heading);
    void fitRoadAt(int sample_idx, Eigen::Vector2d& start, Eigen::Vector2d& end, float& start_heading, float& end_heading, int& n_lanes, bool& is_oneway, float& lane_width, set<int>& covered_data);
    
    // Road relationship
    void refit();
    
    // Rendering
    void draw();
    
    // Clear
    void clear();
    void cleanUp();
    
private:
    PclPointCloud::Ptr           point_cloud_;
    PclSearchTree::Ptr           search_tree_;
    
    // Tensor voting
    vector<Eigen::Matrix2d>      tensor_votes_;
    vector<vector<int>>          traced_curves_;
    
    map<vertex_t, Symbol*>       graph_nodes_;
    symbol_graph_t               symbol_graph_;
    vector<Symbol*>              production_string_;
    Trajectories*                trajectories_;
    vector<bool>                 has_been_covered_;

    FeatureType                  current_feature_type_;
    vector<Vertex>               feature_properties_; // x, y, heading
    vector<int>                  labels_;
    
    vector<Vertex>               feature_vertices_;
    vector<Color>                feature_colors_;
};

#endif /* defined(__junction_reconstruction__road_generator__) */

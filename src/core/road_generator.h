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
    
    void applyRules();
    
    // Features & Prediction
    bool loadQueryInitClassifer(const string& filename);
    bool saveQueryInitResult(const string& filename);
    bool loadQueryInitResult(const string& filename);
    
    bool hasValidQueryInitDecisionFunction() const { return query_init_df_is_valid_; }
    void applyQueryInitClassifier(float radius);
    void extractQueryInitFeatures(float radius);
    int  nQueryInitFeatures() { return query_init_features_.size(); }
    int  nQueryInitLabels() { return query_init_labels_.size(); }
    bool exportQueryInitFeatures(float radius, const string& filename);
    bool loadQueryInitFeatures(const string& filename);
   
    bool loadQueryQClassifer(const string& filename);
    bool hasValidQueryQDecisionFunction() const { return query_q_df_is_valid_; }
    
    float estimateRoadWidth(RoadPt& seed_pt);
    
    bool computeInitialRoadGuess();
    bool addInitialRoad();
    
    void tmpFunc();
    
    void trace_roads();
    void extend_road(int r_idx,
                     vector<int>&,
                     vector<bool>& mark_list,
                     bool forward = true);
    void generateRoadFromPoints(vector<int>& candidate_pt_idxs,
                                vector<RoadPt>& road);
    
    
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
    PclPointCloud::Ptr              point_cloud_;
    PclSearchTree::Ptr              search_tree_;
    
    PclPointCloud::Ptr              tmp_point_cloud_; // a tmp point cloud for ease of use
    PclSearchTree::Ptr              tmp_search_tree_;
    
    map<vertex_t, Symbol*>          graph_nodes_;
    symbol_graph_t                  symbol_graph_;
    vector<Symbol*>                 production_string_;
    Trajectories*                   trajectories_;
    vector<bool>                    has_been_covered_;

    // Query Init Features and Classifiers
    query_init_decision_function    query_init_df_;
    bool                            query_init_df_is_valid_;
    vector<query_init_sample_type>  query_init_features_;
    vector<int>                     query_init_labels_;
    vector<Vertex>                  query_init_feature_properties_; // x, y, heading
    
    vector<vector<RoadPt> >         initial_roads_;
    
    query_q_decision_function       query_q_df_;
    bool                            query_q_df_is_valid_;
    vector<query_q_sample_type>     query_q_features_;
    vector<int>                     query_q_labels_;
    vector<Vertex>                  query_q_feature_properties_;
    
    // For visualization
    vector<Vertex>                  feature_vertices_;
    vector<Color>                   feature_colors_;
    
    // For DEBUG usage
    int                             tmp_;
    vector<Vertex>                  lines_to_draw_;
    vector<Color>                   line_colors_;
    vector<Vertex>                  points_to_draw_;
    vector<Color>                   point_colors_;
    
    
    int                             j;
    RoadSymbol*                     new_road;
    float                           cur_cum_length;
    float                           radius;
    int                             i_road;
};

#endif /* defined(__junction_reconstruction__road_generator__) */

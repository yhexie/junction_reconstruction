#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <vector>
#include <QVector4D>
#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>

#include "pcl_types.h"

#define PI 3.1415926
using namespace std;

// Default directories
const string default_trajectory_dir = "/Users/chenchen/Research/junction_reconstruction/data/trajectory_data/junction_trajectories";
const string default_map_dir = "/Users/chenchen/Research/junction_reconstruction/data/junction_data";
const string default_python_test_dir = "/Users/chenchen/Research/python_test";

typedef RichPoint                       PclPoint;
typedef pcl::PointCloud<PclPoint>       PclPointCloud;
typedef pcl::search::Search<PclPoint>   PclSearchTree;

static const float POSITIVE_INFINITY = 1e9;
static const float LANE_WIDTH        = 3.7f; // in meters

// Visualization Z values
    // Traj
static const float Z_TRAJECTORIES           = -1.0f;
static const float Z_TRAJECTORY_SPEED       = -1.0f;
static const float Z_PATHLETS               = -0.4f;
static const float Z_SELECTED_TRAJ          = -0.4f;
static const float Z_SAMPLES                = -0.5f;
static const float Z_FEATURES               = -0.4f;
static const float Z_SELECTED_SAMPLES       = 0.0f;
static const float Z_SEGMENTS               = -0.4f;
static const float Z_DISTANCE_GRAPH         = -0.3f;
static const float Z_ROAD                   = -0.3f;
static const float Z_SELECTION              = -0.3f;
    // OSM
static const float Z_OSM = -0.35f;
    // Graph
static const float Z_GRAPH = -0.3f;
static const float Z_PATH = -0.2f;

    // Debug
static const float Z_DEBUG = -0.1f;

enum TrajectoryColorMode{
    UNIFORM,
    TRAJECTORY,
    SAMPLE_TIME,
    SAMPLE_ORDER
};

enum ToolMode{
    NO_ACTIVE_TOOL,
    ADD_VALVE
};

struct Vertex{
    float x, y, z;
    Vertex(float tx, float ty, float tz) : x(tx), y(ty), z(tz){}
    Vertex(){x = 0.0f; y = 0.0f; z = 0.0f;}
};

struct RoadPt{
    float x;
    float y;
    int   head;
    int   n_lanes;
    int   speed = 0; // cm/s
    
    RoadPt() {
        x = 0.0f;
        y = 0.0f;
        head = 0;
        speed = 0;
        n_lanes = 1;
    }
    
    RoadPt(float tx,
           float ty,
           int thead,
           int tLanes) : x(tx),
                        y(ty),
                        head(thead),
                        n_lanes(tLanes) {}
    
    RoadPt(float tx,
           float ty,
           int thead) :  x(tx),
                            y(ty),
                            head(thead)
    {
        n_lanes = 1;
    }
    
    RoadPt(const RoadPt& pt){
        x = pt.x;
        y = pt.y;
        head = pt.head;
        n_lanes = pt.n_lanes;
        speed = pt.speed;
    }
};

namespace Common{
    std::string int2String(int i, int width);
    void randomK(std::vector<int>& random_k, int k, int N);
}

bool pairCompare(const pair<int, float>& firstElem, const pair<int, float>& secondElem);

bool pairCompareDescend(const pair<int, float>& firstElem, const pair<int, float>& secondElem);

float roadPtDistance(const RoadPt& p1, const RoadPt& p2);

void findMaxElement(const vector<float> hist, int& max_idx);

void peakDetector(vector<float>& hist, int window, float ratio, vector<int>& peak_idxs,bool is_closed = true);

float deltaHeading1MinusHeading2(float heading1, float heading2);
Eigen::Vector2d headingTo2dVector(int);
Eigen::Vector3d headingTo3dVector(int);

int vector2dToHeading(const Eigen::Vector2d);
int vector3dToHeading(const Eigen::Vector3d);
int increaseHeadingBy(int delta_heading,
                      const int orig_heading);
int decreaseHeadingBy(int delta_heading,
                      const int orig_heading);

void uniformSampleCurve(vector<RoadPt>& center_line, float interval_size = 5.0f);
void smoothCurve(vector<RoadPt>& center_line, bool fix_front = true);


/*
    sampleGPSPoints: speed up by sampling old GPS point cloud by radius, new_points to remove overlapping points. pt.id_sample in new_points represents the strength of the point.
 */
void sampleGPSPoints(float radius,
                     float heading_threshold,
                     const PclPointCloud::Ptr& points,
                     const PclSearchTree::Ptr& search_tree,
                     PclPointCloud::Ptr& new_points,
                     PclSearchTree::Ptr& new_search_tree);

/*
    sampleRoadSkeletonPoints: this function extract a set of candidate skeleton points from the original point cloud: points
 */
void sampleRoadSkeletonPoints(float search_radius,
                              float heading_threshold,
                              float delta_perp_bin,
                              float sigma_perp_s,
                              float delta_head_bin,
                              float sigma_head_s,
                              int   min_n,
                              bool  is_oneway,
                              const PclPointCloud::Ptr& points,
                              const PclSearchTree::Ptr& search_tree,
                              PclPointCloud::Ptr& new_points,
                              PclSearchTree::Ptr& new_search_tree);

void adjustRoadPtHeading(RoadPt& r_pt,
                         PclPointCloud::Ptr& points,
                         PclSearchTree::Ptr& search_tree,
                         float search_radius,
                         float heading_threshold,
                         float delta_bin,
                         bool pt_id_sample_store_weight = false);

/*
 This function projects qualified nearby points to the perpendicular line to pt.head, and compute a histogram of distribution. The road center is the maximum of that distribution.

    Note: r_pt.x, r_pt.y, r_pt.n_lanes will be updated accordingly
    
    The last argument: pt_id_sample_store_weight - an indicator whether the points are original GPS points. Sometimes, I use simplified point cloud by sampling original point cloud to speed up. In the simplified point cloud, the point's id_sample stores how many original points it covers, which is essentially the weight of the simplified point.
 */
void adjustRoadCenterAt(RoadPt&             r_pt,
                        PclPointCloud::Ptr& points,
                        PclSearchTree::Ptr& search_tree,
                        float               trajecotry_avg_speed,
                        float               search_radius,
                        float               heading_threshold,
                        float               delta_bin,
                        float               sigma_s,
                        bool                pt_id_sample_store_weight);

class Parameters{
public:
    static Parameters& getInstance() {
        static Parameters singleton_;
        return singleton_;
    }
   
    float& searchRadius() { return search_radius_; }
    float& deltaGrowingLength() { return delta_growing_length_; }
    float& gpsSigma() { return gps_sigma_; }
    float& gpsMaxHeadingError() { return gps_max_heading_error_; }
    
    float& roadSigmaH() { return road_sigma_h_; }
    float& roadSigmaW() { return road_sigma_w_; }
    float& roadVoteGridSize() { return road_vote_grid_size_; }
    float& roadVoteThreshold() { return road_vote_threshold_; }
    
    float& branchPredictorExtensionRatio() { return branch_predictor_extension_ratio_; }
    float& branchPredictorMaxTExtension() { return branch_predictor_max_t_extension_; }
    
private:
    Parameters() {
        search_radius_ = 25.0f;
        gps_sigma_ = 10.0f;
        gps_max_heading_error_ = 15.0f;
        delta_growing_length_ = 50.0f;
        
        road_sigma_h_ = 10.0f;
        road_sigma_w_ = 2.5f;
        road_vote_grid_size_ = 2.5f;
        road_vote_threshold_ = 0.5f;
        
        branch_predictor_extension_ratio_ = 6.0f;
        branch_predictor_max_t_extension_ = 30.0f;
    }
    
    Parameters(Parameters const&) = delete;
    void operator=(Parameters const&) = delete;
    
    virtual ~Parameters() {};
    
    // Parameters
    float search_radius_;
    float delta_growing_length_;
    float gps_sigma_;
    float gps_max_heading_error_;
    
    float road_sigma_h_;
    float road_sigma_w_;
    float road_vote_grid_size_;
    float road_vote_threshold_;
    
    float branch_predictor_extension_ratio_;
    float branch_predictor_max_t_extension_;
};

class SceneConst
{
public:
    static SceneConst& getInstance() {
        static SceneConst singleton_;
        return singleton_;
    }
   
    const QVector4D& getBoundBox() { return bound_box_; }
    void setBoundBox(QVector4D new_bound_box) {
        bound_box_ = new_bound_box;
        updateAttr();
    }
    
    void updateBoundBox(QVector4D bound_box_to_insert) {
        if (bound_box_[0] > bound_box_to_insert[0]) {
            bound_box_[0] = bound_box_to_insert[0];
        }
        if (bound_box_[1] < bound_box_to_insert[1]) {
            bound_box_[1] = bound_box_to_insert[1];
        }
        if (bound_box_[2] > bound_box_to_insert[2]) {
            bound_box_[2] = bound_box_to_insert[2];
        }
        if (bound_box_[3] < bound_box_to_insert[3]) {
            bound_box_[3] = bound_box_to_insert[3];
        }
        updateAttr();
    }
    
    void updateAttr(); // Update
    
    // Convert x,y coordinates into drawing coordinates
    Vertex normalize(float x, float y, float z = 0.0f);
    
private:
    SceneConst() { bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10); };
    SceneConst(SceneConst const&);
    SceneConst& operator=(SceneConst const&);
    virtual ~SceneConst() {};
    
    // Scene boundary box, [min_easting, max_easting, min_northing, max_northing]
    QVector4D                           bound_box_;
    float                               scale_factor_;
    float                               delta_x_;
    float                               delta_y_;
    float                               center_x_;
    float                               center_y_;
};

#endif //COMMON_H_

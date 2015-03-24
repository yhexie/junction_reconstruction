#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <vector>
#include <QVector4D>
#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include "pcl_wrapper_types.h"
#include <Eigen/Dense>

#define PI 3.1415926
using namespace std;

// Default directories
const string default_trajectory_dir = "/Users/chenchen/Research/junction_reconstruction/data/trajectory_data/junction_trajectories";
const string default_map_dir = "/Users/chenchen/Research/junction_reconstruction/data/junction_data";
const string default_python_test_dir = "/Users/chenchen/Research/python_test";

typedef RichPoint                       PclPoint;
typedef pcl::PointCloud<PclPoint>       PclPointCloud;
typedef pcl::search::Search<PclPoint>   PclSearchTree;

static const float POSITIVE_INFINITY = 1e6;

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

namespace Common{
    std::string int2String(int i, int width);
    void randomK(std::vector<int>& random_k, int k, int N);
}

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
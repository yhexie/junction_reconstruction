#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <vector>
#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include "pcl_wrapper_types.h"

#define PI 3.1415926

typedef RichPoint                       PclPoint;
typedef pcl::PointCloud<PclPoint>       PclPointCloud;
typedef pcl::search::Search<PclPoint>   PclSearchTree;

static const double POSITIVE_INFINITY = 1e6;

// Visualization Z values
    // Traj
static const float Z_TRAJECTORIES           = -1.0f;
static const float Z_PATHLETS               = -0.4f;
static const float Z_SELECTED_TRAJ          = -0.4f;
static const float Z_TRAJECTORY_SPEED       = -0.3f;
static const float Z_SAMPLES                = -0.5f;
static const float Z_SELECTED_SAMPLES       = 0.0f;
static const float Z_SEGMENTS               = -0.4f;
static const float Z_DISTANCE_GRAPH         = -0.3f;
    // OSM
static const float Z_OSM = -0.35f;
    // Graph
static const float Z_GRAPH = -0.3f;
static const float Z_PATH = -0.2f;

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

namespace Common{
    std::string int2String(int i, int width);
    void randomK(std::vector<int>& random_k, int k, int N);
}



#endif //COMMON_H_
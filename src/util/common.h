#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <vector>
#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include "pcl_wrapper_types.h"

typedef RichPoint                       PclPoint;
typedef pcl::PointCloud<PclPoint>       PclPointCloud;
typedef pcl::search::Search<PclPoint>   PclSearchTree;

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
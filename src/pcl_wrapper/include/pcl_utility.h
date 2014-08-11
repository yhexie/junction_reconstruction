#ifndef PCL_UTILITY_H
#define PCL_UTILITY_H

#include <string>

#include "pcl_wrapper_types.h"
#include "pcl_wrapper_exports.h"

namespace PCLUtility
{
  PCL_WRAPPER_EXPORTS float transformationToRotationAngle(const Eigen::Matrix4d& transformation);
};

#endif // PCL_UTILITY_H

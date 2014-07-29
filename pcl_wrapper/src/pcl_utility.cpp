#include "pcl_utility.h"

namespace PCLUtility
{
  float transformationToRotationAngle(const Eigen::Matrix4d& transformation)
  {
	  Eigen::Vector3d up_vector(0.0f, 1.0f, 0.0f);
	  Eigen::Vector3d horizontal(1.0f, 0.0f, 0.0f);

	  Eigen::Matrix3d rotation = transformation.topLeftCorner(3, 3);
	  Eigen::Vector3d horizontal_rotated = rotation*horizontal;
	  Eigen::Vector3d projection(horizontal_rotated.x(), 0.0f, horizontal_rotated.z());

	  Eigen::Vector3d vertical = projection.cross(horizontal);
	  double magnitude = vertical.norm();
	  if (vertical.dot(up_vector) < 0)
		  magnitude = -magnitude;
	  float theta = std::atan2(magnitude, projection.dot(horizontal));

	  return theta;
  }
};


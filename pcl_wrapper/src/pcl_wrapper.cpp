#include "pcl_wrapper_types.h"
#include "pcl_wrapper_exports.h"

#include <pcl/impl/pcl_base.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/impl/organized.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/surface/impl/convex_hull.hpp>
#include <pcl/surface/impl/concave_hull.hpp>
#include <pcl/features/impl/integral_image_normal.hpp>

#include <pcl/features/impl/ppf.hpp>

template class PCL_WRAPPER_EXPORTS pcl::VoxelGrid<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::ConvexHull<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::ConcaveHull<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::KdTreeFLANN<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::search::KdTree<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::search::Search<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::search::FlannSearch<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::search::OrganizedNeighbor<RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::NormalEstimation<RichPoint, RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::IntegralImageNormalEstimation<RichPoint, RichPoint>;
//template class PCL_WRAPPER_EXPORTS pcl::PPFRegistration<RichPoint, RichPoint>;
template class PCL_WRAPPER_EXPORTS pcl::PPFEstimation<RichPoint, RichPoint, pcl::PPFSignature>;
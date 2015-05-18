#include "pcl_types.h"

#include <pcl/impl/pcl_base.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/impl/organized.hpp>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

template class pcl::KdTreeFLANN<RichPoint>;
template class pcl::search::KdTree<RichPoint>;
template class pcl::search::Search<RichPoint>;
template class pcl::search::FlannSearch<RichPoint>;
template class pcl::search::OrganizedNeighbor<RichPoint>;

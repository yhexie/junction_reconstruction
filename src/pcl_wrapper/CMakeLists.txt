set(Boost_USE_STATIC_LIBS  ON)
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

find_package(PCL REQUIRED common io kdtree search features filters registration segmentation surface sample_consensus)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_common_LIBRARY_DIRS})
link_directories(${PCL_io_LIBRARY_DIRS})
link_directories(${PCL_kdtree_LIBRARY_DIRS})
link_directories(${PCL_search_LIBRARY_DIRS})
link_directories(${PCL_surface_LIBRARY_DIRS})
link_directories(${PCL_features_LIBRARY_DIRS})
link_directories(${PCL_filters_LIBRARY_DIRS})
link_directories(${PCL_registration_LIBRARY_DIRS})
link_directories(${PCL_segmentation_LIBRARY_DIRS})
link_directories(${PCL_sample_consensus_LIBRARY_DIRS})
link_directories(${PCL_filters_LIBRARY_DIRS})
add_definitions(${PCL_io_DEFINITIONS})
add_definitions(${PCL_kdtree_DEFINITIONS})
add_definitions(${PCL_search_DEFINITIONS})
add_definitions(${PCL_surface_DEFINITIONS})
add_definitions(${PCL_features_DEFINITIONS})
add_definitions(${PCL_filters_DEFINITIONS})
add_definitions(${PCL_registration_DEFINITIONS})
add_definitions(${PCL_segmentation_DEFINITIONS})
add_definitions(${PCL_sample_consensus_DEFINITIONS})

include_directories(${FLANN_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(incs      include/pcl_wrapper_types.h
              include/pcl_wrapper_exports.h
              include/pcl_utility.h
              )

set(srcs      src/pcl_wrapper.cpp
              src/pcl_utility.cpp
              )

set(impl_incs)

set(lib_name pcl_wrapper)
add_library(${lib_name} SHARED ${srcs} ${incs} ${impl_incs})
target_link_libraries(${lib_name} 
  ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
  debug ${PCL_COMMON_LIBRARY_DEBUG} optimized ${PCL_COMMON_LIBRARY}
  debug ${PCL_IO_LIBRARY_DEBUG} optimized ${PCL_IO_LIBRARY}
  debug ${PCL_KDTREE_LIBRARY_DEBUG} optimized ${PCL_KDTREE_LIBRARY}
  debug ${PCL_SEARCH_LIBRARY_DEBUG} optimized ${PCL_SEARCH_LIBRARY}
  debug ${PCL_FEATURES_LIBRARY_DEBUG} optimized ${PCL_FEATURES_LIBRARY}
  debug ${PCL_FILTERS_LIBRARY_DEBUG} optimized ${PCL_FILTERS_LIBRARY}
  debug ${PCL_REGISTRATION_LIBRARY_DEBUG} optimized ${PCL_REGISTRATION_LIBRARY}
  debug ${PCL_SAMPLE_CONSENSUS_LIBRARY_DEBUG} optimized ${PCL_SAMPLE_CONSENSUS_LIBRARY}
  debug ${PCL_SEGMENTATION_LIBRARY_DEBUG} optimized ${PCL_SEGMENTATION_LIBRARY}
  debug ${PCL_SURFACE_LIBRARY_DEBUG} optimized ${PCL_SURFACE_LIBRARY}
  ${QHULL_LIBRARY}
  )

if(WIN32 AND MSVC)
  set_target_properties(${lib_name} PROPERTIES LINK_FLAGS_RELEASE /OPT:REF)
elseif(CMAKE_SYSTEMname STREQUAL "Darwin")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(${lib_name} PROPERTIES LINK_FLAGS -Wl)
  endif()
elseif(__COMPILER_PATHSCALE)
  set_target_properties(${lib_name} PROPERTIES LINK_FLAGS -mp)
else()
  set_target_properties(${lib_name} PROPERTIES LINK_FLAGS -Wl)
endif()

set_target_properties(${lib_name} PROPERTIES DEFINE_SYMBOL "PCL_WRAPPER_API_EXPORTS")

set_target_properties(${lib_name} PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(${lib_name} PROPERTIES RELEASE_POSTFIX _release)

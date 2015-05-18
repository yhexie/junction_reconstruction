cmake_minimum_required(VERSION 2.8.11)

# OpenCV
#find_package(OpenCV REQUIRED)

# Boost
set(Boost_USE_STATIC_LIBS  ON)
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# PCL
find_package(PCL REQUIRED common kdtree search)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${FLANN_INCLUDE_DIRS})

# CGAL
find_package(CGAL QUIET COMPONENTS Core ) # CGAL
if ( CGAL_FOUND )
  include( ${CGAL_USE_FILE} )
  include( CGAL_CreateSingleSourceCGALProgram )
else()
    message(STATUS "This program requires the CGAL library, and will not be compiled.")
endif()

# Osmium for parsing openstreetmap
find_package(EXPAT REQUIRED)
include_directories(${EXPAT_INCLUDE_DIRS})
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
SET(OSMPBF_INCLUDE_DIR "/usr/local/include/osmpbf")
SET(OSMPBF_LIBRARIES "/usr/local/lib/libosmpbf.a")
set(PROJ_INCLUDE_DIR "/usr/local/include/")
set(PROJ_LIBRARIES "/usr/local/lib/libproj.dylib")
find_package(BZIP2 REQUIRED)
include_directories(${BZIP_INCLUDE_DIRS})
include_directories("osmium/include")

# Qt 
set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Xml REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(OpenGL REQUIRED)
    # Set Qt files and resources here
file(GLOB uis resources/*.ui)
file(GLOB resources resources/*.qrc)
#set(uis       resources/main_window.ui)
#set(resources  resources/main_window.qrc)
QT5_WRAP_UI(ui_srcs ${uis})
QT5_ADD_RESOURCES(resource_srcs ${resources})

# Put the ui in the windows project file
IF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
  SET (SOURCE_DIR ${SOURCE_DIR} ${uis})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
IF (${CMAKE_BUILD_TOOL} MATCHES "devenv")
  SET (SOURCE_DIR ${SOURCE_DIR} ${uis})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "devenv")

file(GLOB core_source core/*.cpp core/*.cc)
file(GLOB core_header core/*.h)
file(GLOB util_source util/*.cpp)
file(GLOB util_header util/*.h)
file(GLOB display_source display/*.cpp)
file(GLOB display_header display/*.h)
file(GLOB exec_source exec/*.cpp)

set(srcs ${core_source}
         ${util_source}
         ${display_source}
         ${exec_source})

set(incs ${core_header}
         ${util_header}
         ${display_header})

include_directories(core display util)

# Organize files
SOURCE_GROUP("Resources" FILES ${uis} ${resources})
SOURCE_GROUP("Generated" FILES ${ui_srcs} ${moc_srcs} ${resource_srcs})
SET_SOURCE_FILES_PROPERTIES(${srcs} PROPERTIES OBJECT_DEPENDS "${ui_srcs}")

SOURCE_GROUP("Core" FILES ${core_source} ${core_header})
SOURCE_GROUP("Display" FILES ${display_source} ${display_header})
SOURCE_GROUP("Exec" FILES ${exec_source})
SOURCE_GROUP("Util" FILES ${util_source} ${util_header})

# Put the ui in the windows project file
IF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
  SET (srcs ${srcs} ${uis})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
IF (${CMAKE_BUILD_TOOL} MATCHES "devenv")
  SET (srcs ${srcs} ${uis})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "devenv")

set(exe_name junction_reconstruction)
add_executable(${exe_name} ${ui_srcs} ${moc_srcs} ${resource_srcs} ${srcs} ${incs})
target_link_libraries(${exe_name} 
    ${CGAL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${Boost_SYSTEM_LIBRARY} 
    ${Boost_FILESYSTEM_LIBRARY}
    ${PCL_COMMON_LIBRARY} 
    ${PCL_KDTREE_LIBRARY}
    ${PCL_SEARCH_LIBRARY}
    Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Xml Qt5::Concurrent
    ${PROTOBUF_LIBRARIES}
    ${EXPAT_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${OSMPBF_LIBRARIES}
    ${PROJ_LIBRARIES}
)

if(WIN32 AND MSVC)
  set_target_properties(${exe_name} PROPERTIES LINK_FLAGS /FORCE:MULTIPLE)
  set_target_properties(${exe_name} PROPERTIES LINK_FLAGS_RELEASE /OPT:REF)
elseif(CMAKE_SYSTEMname STREQUAL "Darwin")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(${exe_name} PROPERTIES LINK_FLAGS -Wl)
  endif()
elseif(__COMPILER_PATHSCALE)
  set_target_properties(${exe_name} PROPERTIES LINK_FLAGS -mp)
else()
  set_target_properties(${exe_name} PROPERTIES LINK_FLAGS -Wl)
endif()

set_target_properties(${exe_name} PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(${exe_name} PROPERTIES RELEASE_POSTFIX _release)


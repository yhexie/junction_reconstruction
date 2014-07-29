cmake_minimum_required(VERSION 2.8.11)

project(my_project)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(MSVC)
  add_definitions ("-DBOOST_ALL_NO_LIB -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj /EHsc /fp:fast /wd4800 /wd4521 /wd4251 /wd4275 /wd4305 /wd4355 /wd4819 /wd4250 /wd4267")
  # no global code optimization to speed up compile
  #SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /wd4800")
  #SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
  #SET(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
  # /MANIFEST:NO") # please, don't disable manifest generation, otherwise crash at start for vs2008
else(MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth-1024 -stdlib=libc++")
endif()

# the gcc-4.2.1 coming with MacOS X is not compatible with the OpenMP pragmas we use, so disabling OpenMP for it
if((NOT APPLE) OR (NOT CMAKE_COMPILER_IS_GNUCXX) OR (GCC_VERSION VERSION_GREATER 4.2.1) OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  find_package(OpenMP)
endif()

if (APPLE)
  set(CMAKE_PREFIX_PATH "/usr/local/Cellar/qt5/5.3.1")
endif()

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message (STATUS "Found OpenMP")
  if(MSVC90 OR MSVC10)
    if(MSVC90)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC10)
      set(OPENMP_DLL VCOMP100)
    endif(MSVC90)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
  endif(MSVC90 OR MSVC10)
else(OPENMP_FOUND)
  message (STATUS "Not found OpenMP")
endif()

set(GPS_Map_Construction_OUTPUT_LIB_DIR ${PROJECT_BINARY_DIR}/lib)
set(GPS_Map_Construction_OUTPUT_BIN_DIR ${PROJECT_BINARY_DIR}/bin)
make_directory(${GPS_Map_Construction_OUTPUT_LIB_DIR})
make_directory(${GPS_Map_Construction_OUTPUT_BIN_DIR})
if(WIN32)
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} CONFIG)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG} "${GPS_Map_Construction_OUTPUT_LIB_DIR}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${GPS_Map_Construction_OUTPUT_BIN_DIR}")
    # ---[ Windows requires DLLs (shared libraries) to be installed in the same directory as executables
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG} "${GPS_Map_Construction_OUTPUT_BIN_DIR}")
  endforeach(config)
else(WIN32)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${GPS_Map_Construction_OUTPUT_LIB_DIR}")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${GPS_Map_Construction_OUTPUT_BIN_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${GPS_Map_Construction_OUTPUT_LIB_DIR}")
endif(WIN32)

add_subdirectory(pcl_wrapper EXCLUDE_FROM_ALL)
#add_subdirectory(pcl_wrapper)

find_package(PCL REQUIRED common io)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${FLANN_INCLUDE_DIRS})

find_package(CGAL QUIET COMPONENTS Core )
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
# End Osmium

include_directories(${PROJECT_SOURCE_DIR}/pcl_wrapper/include)
include_directories("src")

# Qt stuff
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5Xml REQUIRED)
find_package(Qt5Concurrent REQUIRED)
find_package(OpenGL REQUIRED)

# Set Qt files and resources here
set(uis       resources/main_window.ui)
set(resources  resources/main_window.qrc)

# Qt stuff
QT5_WRAP_UI(ui_srcs ${uis})
QT5_ADD_RESOURCES(resource_srcs ${resources})

set(incs    src/main_window.h
            src/common.h
            src/color_map.h
            src/trajectory.h
            src/trajectories.h
            src/scene_widget.h
            src/customized_shader_program.h
            src/renderable.h
            src/openstreetmap.h
            src/cgal_types.h
            )

set(srcs    src/main.cpp
            src/main_window.cpp
            src/color_map.cpp
            src/common.cpp
            src/trajectory.cpp
            src/trajectories.cpp
            src/scene_widget.cpp
            src/customized_shader_program.cpp
            src/renderable.cpp
            src/openstreetmap.cpp
           )

# Osmium code
include("osmium/include")

# Organize files
SOURCE_GROUP("Resources" FILES ${uis} ${resources})
SOURCE_GROUP("Generated" FILES ${ui_srcs} ${moc_srcs} ${resource_srcs})
SET_SOURCE_FILES_PROPERTIES(${srcs} PROPERTIES OBJECT_DEPENDS "${ui_srcs}")

# Put the ui in the windows project file
IF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
  SET (srcs ${srcs} ${uis})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "msdev")
IF (${CMAKE_BUILD_TOOL} MATCHES "devenv")
  SET (srcs ${srcs} ${uis})
ENDIF (${CMAKE_BUILD_TOOL} MATCHES "devenv")

set(exe_name my_project)
add_executable(${exe_name} ${ui_srcs} ${moc_srcs} ${resource_srcs} ${srcs} ${incs} ${impl_incs})
target_link_libraries(${exe_name} 
    pcl_wrapper
    ${Boost_THREAD_LIBRARY} 
    ${Boost_SYSTEM_LIBRARY} 
    ${CGAL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    Qt5::Core Qt5::Widgets Qt5::OpenGL Qt5::Xml Qt5::Concurrent
    ${PROTOBUF_LIBRARIES}
    ${EXPAT_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${OSMPBF_LIBRARIES}
)

set_target_properties(${exe_name} PROPERTIES DEBUG_POSTFIX _debug)
set_target_properties(${exe_name} PROPERTIES RELEASE_POSTFIX _release)

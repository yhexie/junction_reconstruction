junction_reconstruction
=======================

This is the code repository for GPS based junction reconstruction.

### Dependencies

* [PCL](http://pointclouds.org) (i.e., Point Cloud Library) >= 1.7.1
* [CGAL](https://www.cgal.org) >= 4.4
* [osmium](https://github.com/joto/osmium) and its related prerequisites
* [Protobuf](https://code.google.com/p/protobuf/) >= 2.5.0
* [libproj](http://trac.osgeo.org/proj/) (NOTICE: need to manually change src/CMakeLists.txt for the directory of libproj)
* Qt5

### Compile

1. Compile `proto/gps_trajectory.proto`
```
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/gps_trajectory.proto
```

Then move the output `gps_trajectory.pb.h` and `gps_trajectory.pb.cc` to the `core/` directory.

2. Follow standard CMake procedure to compile the rest of the code.

For more details, please refer to the [wiki page](https://github.com/cchen1986/junction_reconstruction/wiki).

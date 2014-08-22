#include "latlon_converter.h"

Projector::Projector(){
    if (!(src_proj_ = pj_init_plus(BJ_SRC_PROJ))) {
        fprintf(stderr, "Error! Cannot initialize latlon to XY projector!\n");
        exit(1);
    }
    if (!(dst_proj_ = pj_init_plus(BJ_DST_PROJ))) {
        fprintf(stderr, "Error! Cannot initialize latlon to XY projector!\n");
        exit(1);
    }
}

Projector::~Projector(){
    
}

void Projector::convertLatlonToXY(float lat, float lon, float &x, float &y){
    double tmp_x = lon * DEG_TO_RAD;
    double tmp_y = lat * DEG_TO_RAD;
    pj_transform(src_proj_, dst_proj_, 1, 1, &tmp_x, &tmp_y, NULL);
    x = static_cast<float>(tmp_x - BJ_X_OFFSET);
    y = static_cast<float>(tmp_y - BJ_Y_OFFSET);
}
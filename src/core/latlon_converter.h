#ifndef LATLON_CONVERTER_H_
#define LATLON_CONVERTER_H_

#include <proj_api.h>
#include <cstdio>
using namespace std;

static const char *BJ_DST_PROJ = "+proj=utm +zone=50 +south=False +ellps=WGS84";
static const char *BJ_SRC_PROJ = "+proj=latlon +ellps=WGS84";

class Projector{
public:
    Projector();
    ~Projector();
    void convertLatlonToXY(float, float, float&, float&);
private:
    projPJ              src_proj_;
    projPJ              dst_proj_;
};

#endif //LATLON_CONVERTER_H_
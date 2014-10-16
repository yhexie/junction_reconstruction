#ifndef LATLON_CONVERTER_H_
#define LATLON_CONVERTER_H_

#include <proj_api.h>
#include <cstdio>
using namespace std;

static const double UTC_OFFSET = 1241100000;

static const char *BJ_DST_PROJ = "+proj=utm +zone=50 +south=False +ellps=WGS84";
static const char *BJ_SRC_PROJ = "+proj=latlon +ellps=WGS84";
static const double BJ_X_OFFSET = 440000;
static const double BJ_Y_OFFSET = 4400000;

static const char *SF_DST_PROJ = "+proj=utm +zone=10 +south=False +ellps=WGS84";
static const char *SF_SRC_PROJ = "+proj=latlon +ellps=WGS84";
static const double SF_X_OFFSET = 550000;
static const double SF_Y_OFFSET = 4160000;

class Projector{
public:
    static Projector& getInstance() {
        static Projector singleton_;
        return singleton_;
    }
    
    typedef enum {BEIJING, SAN_FRANCISCO} NamedUTMZone;
    
    Projector();
    ~Projector();
    
    void convertLatlonToXY(float, float, float&, float&);
    
    void setUTMZone(NamedUTMZone zone);
    
private:
    projPJ              src_proj_;
    projPJ              dst_proj_;
    double              x_offset_;
    double              y_offset_;
};

#endif //LATLON_CONVERTER_H_
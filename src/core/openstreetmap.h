#ifndef OPENSTREETMAP_H_
#define OPENSTREETMAP_H_

#define OSMIUM_WITH_PBF_INPUT
#define OSMIUM_WITH_XML_INPUT

#include <osmium.hpp>
#include <osmium/handler/coordinates_for_ways.hpp>
#include <osmium/storage/byid/sparse_table.hpp>
#include <osmium/storage/byid/mmap_file.hpp>
#include <osmium/geometry/point.hpp>

#include "renderable.h"
#include "common.h"
#include "color_map.h"
using namespace std;

typedef Osmium::Storage::ById::SparseTable<Osmium::OSM::Position> storage_sparsetable_t;
typedef Osmium::Storage::ById::MmapFile<Osmium::OSM::Position> storage_mmap_t;
typedef Osmium::Handler::CoordinatesForWays<storage_sparsetable_t, storage_mmap_t> cfw_handler_t;

enum WAYTYPE{
    MOTORWAY = 0,
    MOTORWAY_LINK = 1,
    TRUNK = 2,
    TRUNK_LINK = 3,
    PRIMARY = 4,
    PRIMARY_LINK = 5,
    SECONDARY = 6,
    SECONDARY_LINK = 7,
    TERTIARY = 8,
    TERTIARY_LINK = 9,
    OTHER = 10
};

class OsmWay {
public:
    OsmWay(void);
    ~OsmWay(void);
    vector<double> &eastings() {return eastings_;}
    vector<double> &northings() {return northings_;}
    void setWayType(WAYTYPE type) {way_type_ = type;}
    bool isOneway() {return is_oneway_;}
    void setOneway(bool value) {is_oneway_ = value;}
    const WAYTYPE wayType() const {return way_type_;}
private:
    vector<double> eastings_; // Node indicies in node list
    vector<double> northings_;
    WAYTYPE way_type_;
    bool is_oneway_;
};

class OpenStreetMap : public Renderable{
public:
    OpenStreetMap(QObject *parent);
    ~OpenStreetMap();
    
    void prepareForVisualization(QVector4D bound_box);
    Color getWayColor(OsmWay &aWay);
    int getWayWidth(OsmWay &aWay);
    bool isEmpty() const {return is_empty_;}
    void draw();
    bool loadOSM(const string &filename);
    
    void pushAWay(OsmWay &aWay);
    void updateBoundBox();
    const QVector4D&    BoundBox(void) const {return bound_box_;}
    void clearData(void);
    
private:
    QVector4D                   bound_box_;
    bool                        is_empty_;
    vector<Vertex>              normalized_vertices_;
    vector<Color>               vertex_colors_;
    vector<Vertex>              direction_vertices_;
    vector<Color>               direction_colors_;
    vector<vector<unsigned> >   direction_idxs_;
    
    vector<vector<unsigned> >   way_idxs_;
    vector<int>            way_widths_;
    
    vector<OsmWay>              ways_;
};

class MyShapeHandler : public Osmium::Handler::Base{
    storage_sparsetable_t store_pos;
    storage_mmap_t store_neg;
    cfw_handler_t *handler_cfw;
public:
    MyShapeHandler(OpenStreetMap *osm_ptr){
        handler_cfw = new cfw_handler_t(store_pos, store_neg);
        osm_ptr_ = osm_ptr;
    }
    
    ~MyShapeHandler(){
        delete handler_cfw;
    }
    
    void init(Osmium::OSM::Meta &meta){
        handler_cfw->init(meta);
    }
    
    void node(const boost::shared_ptr<Osmium::OSM::Node const> &node);
    
    void after_nodes(){
        handler_cfw->after_nodes();
    }
    
    void way(const boost::shared_ptr<Osmium::OSM::Way> &way);
private:
    OpenStreetMap *osm_ptr_;
};

#endif // OPENSTREETMAP_H_
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

typedef Osmium::Storage::ById::SparseTable<Osmium::OSM::Position> storage_sparsetable_t;
typedef Osmium::Storage::ById::MmapFile<Osmium::OSM::Position> storage_mmap_t;
typedef Osmium::Handler::CoordinatesForWays<storage_sparsetable_t, storage_mmap_t> cfw_handler_t;

class MyShapeHandler : public Osmium::Handler::Base{
    storage_sparsetable_t store_pos;
    storage_mmap_t store_neg;
    cfw_handler_t *handler_cfw;
public:
    MyShapeHandler(){
        handler_cfw = new cfw_handler_t(store_pos, store_neg);
    }
    
    ~MyShapeHandler(){
        delete handler_cfw;
    }
    
    void init(Osmium::OSM::Meta &meta){
        handler_cfw->init(meta);
    }
    
    void node(const shared_ptr<Osmium::OSM::Node const> &node){
        handler_cfw->node(node);
        const char *amenity = node->tags().get_value_by_key("amenity");
        if (amenity && !strcmp(amenity, "post_box")) {
            try {
                Osmium::Geometry::Point point(*node);
            } catch (Osmium::Geometry::IllegalGeometry) {
                std::cerr << "Ignoring ilegal geometry for node " << node->id() <<".\n";
            }
        }
    }
    
    void after_nodes(){
        handler_cfw->after_nodes();
    }
    
    void way(const shared_ptr<Osmium::OSM::Way> &way){
        handler_cfw->way(way);
        Osmium::OSM::TagList::iterator it;
        for (it = way->tags().begin(); it != way->tags().end(); ++it) {
            std::cout << it->key() << ".\n";
        }
    }
};

class OsmNode{
public:
    OsmNode(unsigned node_id, easting, northing) : node_id_(node_id), easting_(easting), northing_(northing);
private:
    unsigned node_id_;
    float easting_;
    float northing_;
};

class OsmWay {
public:
    OsmWay(void);
private:
    
};

class OpenStreetMap : public Renderable{
public:
    OpenStreetMap(void);
    OpenStreetMap(const char *fileName);
};

#endif // OPENSTREETMAP_H_
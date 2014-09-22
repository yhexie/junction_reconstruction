#include "openstreetmap.h"
#include "latlon_converter.h"

OsmWay::OsmWay(void) {
    is_oneway_ = false;
}

OsmWay::~OsmWay(void){
    
}

OpenStreetMap::OpenStreetMap(QObject *parent) : Renderable(parent)
{
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    is_empty_ = true;
    normalized_vertices_.clear();
    vertex_colors_.clear();
    way_idxs_.clear();
    way_widths_.clear();
    ways_.clear();
}

OpenStreetMap::~OpenStreetMap(){
}

bool OpenStreetMap::loadOSM(const string &filename){
    Osmium::OSMFile infile(filename.c_str());
    MyShapeHandler handler(this);
    ways_.clear();
    Osmium::Input::read(infile, handler);
    google::protobuf::ShutdownProtobufLibrary();
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    updateBoundBox();
    printf("bound box: min_e=%.2f, max_e=%.2f, min_n=%.2f, max_n=%.2f\n", bound_box_[0], bound_box_[1], bound_box_[2], bound_box_[3]);
    is_empty_ = false;
    return true;
}

void OpenStreetMap::pushAWay(OsmWay &aWay){
    ways_.push_back(aWay);
}

void OpenStreetMap::updateBoundBox(){
    for (size_t i = 0; i < ways_.size(); ++i) {
        OsmWay &aWay = ways_[i];
        for (size_t j = 0; j < aWay.eastings().size(); ++j) {
            if (aWay.eastings()[j] < bound_box_[0]) {
                bound_box_[0] = aWay.eastings()[j];
            }
            if (aWay.eastings()[j] > bound_box_[1]){
                bound_box_[1] = aWay.eastings()[j];
            }
            if (aWay.northings()[j] < bound_box_[2]) {
                bound_box_[2] = aWay.northings()[j];
            }
            if (aWay.northings()[j] > bound_box_[3]){
                bound_box_[3] = aWay.northings()[j];
            }
        }
    }
}

void OpenStreetMap::prepareForVisualization(QVector4D bound_box){
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (delta_x < 0 || delta_y < 0) {
        fprintf(stderr, "Trajectory bounding box error! Min greater than Max!\n");
    }
    
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    float scale_factor_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
    float arrow_length = 2.5f / scale_factor_; // in meters
    
    way_idxs_.clear();
    way_idxs_.resize(ways_.size());
    way_widths_.clear();
    way_widths_.resize(ways_.size(), 2);
    normalized_vertices_.clear();
    vertex_colors_.clear();
    direction_colors_.clear();
    direction_vertices_.clear();
    direction_idxs_.clear();
    
    for (size_t i = 0; i < ways_.size(); ++i) {
        OsmWay &aWay = ways_[i];
        Color vertex_color = getWayColor(aWay);
        way_widths_[i] = getWayWidth(aWay);
        vector<unsigned> &way_idx = way_idxs_[i];
        way_idx.resize(aWay.eastings().size());
        for (size_t j = 0; j < aWay.eastings().size(); ++j) {
            float n_x = (aWay.eastings()[j] - center_x) / scale_factor_;
            float n_y = (aWay.northings()[j] - center_y) / scale_factor_;
            way_idx[j] = normalized_vertices_.size();
            normalized_vertices_.push_back(Vertex(n_x, n_y, Z_OSM));
            vertex_colors_.push_back(vertex_color);
            if (!aWay.isOneway()) {
                continue;
            }
            if (j != 0 && j != aWay.eastings().size()-1) {
                float dx = (aWay.eastings()[j] - aWay.eastings()[j-1]) / scale_factor_;
                float dy = (aWay.northings()[j] - aWay.northings()[j-1]) / scale_factor_;
                float length = sqrt(dx*dx + dy*dy);
                dx = dx / length * arrow_length;
                dy = dy / length * arrow_length;
                QVector3D vec(dx, dy, 0.0);
                QMatrix4x4 m1;
                m1.setToIdentity();
                m1.rotate(30, 0.0f, 0.0f, 1.0f);
                QMatrix4x4 m2;
                m2.setToIdentity();
                m2.rotate(-30, 0.0f, 0.0f, 1.0f);
                QVector3D vec1 = m1.map(vec);
                QVector3D vec2 = m2.map(vec);
                Vertex &prevV = normalized_vertices_[normalized_vertices_.size()-2];
                float center_x = 0.5*n_x + 0.5*prevV.x;
                float center_y = 0.5*n_y + 0.5*prevV.y;
                
                vector<unsigned> direction_idx(3, 0);
                direction_idx[0] = direction_vertices_.size();
                direction_vertices_.push_back(Vertex(center_x - vec1.x(), center_y - vec1.y(), Z_OSM));
                direction_idx[1] = direction_vertices_.size();
                direction_vertices_.push_back(Vertex(center_x, center_y, Z_OSM));
                direction_idx[2] = direction_vertices_.size();
                direction_idxs_.push_back(direction_idx);
                direction_vertices_.push_back(Vertex(center_x - vec2.x(), center_y - vec2.y(), Z_OSM));
                direction_colors_.push_back(vertex_color);
                direction_colors_.push_back(vertex_color);
                direction_colors_.push_back(vertex_color);
            }
        }
    }
}

Color OpenStreetMap::getWayColor(OsmWay &aWay){
    switch (aWay.wayType()) {
        case MOTORWAY:
        case MOTORWAY_LINK:
            return Color(0.0f, 0.4f, 0.8f, 1.0f);
        case TRUNK:
        case TRUNK_LINK:
            return Color(0.0f, 0.8f, 0.4f, 1.0f);
        case PRIMARY:
        case PRIMARY_LINK:
            return Color(1.0f, 0.4f, 0.4f, 1.0f);
        case SECONDARY:
        case SECONDARY_LINK:
            return Color(1.0f, 0.7f, 0.4f, 1.0f);
        case TERTIARY:
        case TERTIARY_LINK:
            return Color(1.0f, 0.4f, 0.4f, 1.0f);
        default:
            return Color(0.7f, 0.7f, 0.7f, 1.0f);
    }
}

int OpenStreetMap::getWayWidth(OsmWay &aWay){
    switch (aWay.wayType()) {
        case MOTORWAY:
        case MOTORWAY_LINK:
            return 5;
        case TRUNK:
        case TRUNK_LINK:
            return 4;
        case PRIMARY:
        case PRIMARY_LINK:
            return 3;
        case SECONDARY:
        case SECONDARY_LINK:
            return 3;
        case TERTIARY:
        case TERTIARY_LINK:
            return 3;
        default:
            return 3;
    }
}

void OpenStreetMap::draw(){
    if (way_idxs_.size() == 0) {
        return;
    }
    
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&vertex_colors_[0], 4*vertex_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    for (size_t i=0; i < way_idxs_.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(way_idxs_[i][0]), way_idxs_[i].size()*sizeof(unsigned));
        glLineWidth(way_widths_[i]);
        glDrawElements(GL_LINE_STRIP, way_idxs_[i].size(), GL_UNSIGNED_INT, 0);
        glPointSize(way_widths_[i]*2);
        glDrawElements(GL_POINTS, way_idxs_[i].size(), GL_UNSIGNED_INT, 0);
    }
   
    // draw arrows
    QOpenGLBuffer vertex_buffer;
    vertex_buffer.create();
    vertex_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertex_buffer.bind();
    vertex_buffer.allocate(&direction_vertices_[0], 3*direction_vertices_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    QOpenGLBuffer color_buffer;
    color_buffer.create();
    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    color_buffer.bind();
    color_buffer.allocate(&direction_colors_[0], 4*direction_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    for (size_t i=0; i < direction_idxs_.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(direction_idxs_[i][0]), direction_idxs_[i].size()*sizeof(unsigned));
        glLineWidth(3);
        glDrawElements(GL_LINE_STRIP, direction_idxs_[i].size(), GL_UNSIGNED_INT, 0);
    }
}

void MyShapeHandler::node(const boost::shared_ptr<const Osmium::OSM::Node> &node){
    handler_cfw->node(node);
    const char *amenity = node->tags().get_value_by_key("amenity");
    if (amenity && !strcmp(amenity, "post_box")) {
        try {
            Osmium::Geometry::Point point(*node);
        } catch (Osmium::Geometry::IllegalGeometry) {
            cerr << "Ignoring ilegal geometry for node " << node->id() <<".\n";
        }
    }
}

void MyShapeHandler::way(const boost::shared_ptr<Osmium::OSM::Way> &way){
    handler_cfw->way(way);
    Osmium::OSM::WayNodeList way_nodes = way->nodes();
    Osmium::OSM::WayNodeList::iterator it;
    OsmWay new_way;
    
    // Process Way Tags
    const char *highway_type = way->tags().get_value_by_key("highway");
    if (highway_type) {
        if (strcmp(highway_type, "motorway") == 0) {
            new_way.setOneway(true);
            new_way.setWayType(MOTORWAY);
        }
        else if (strcmp(highway_type, "motorway_link") == 0){
            new_way.setWayType(MOTORWAY_LINK);
        }
        else if (strcmp(highway_type, "trunk") == 0){
            new_way.setWayType(TRUNK);
        }
        else if (strcmp(highway_type, "trunk_link") == 0){
            new_way.setWayType(TRUNK_LINK);
        }
        else if (strcmp(highway_type, "primary") == 0){
            new_way.setWayType(PRIMARY);
        }
        else if (strcmp(highway_type, "primary_link") == 0){
            new_way.setWayType(PRIMARY_LINK);
        }
        else if (strcmp(highway_type, "secondary") == 0){
            new_way.setWayType(SECONDARY);
        }
        else if (strcmp(highway_type, "secondary_link") == 0){
            new_way.setWayType(SECONDARY_LINK);
        }
        else if (strcmp(highway_type, "tertiary") == 0){
            new_way.setWayType(TERTIARY);
        }
        else if (strcmp(highway_type, "tertiary_link") == 0){
            new_way.setWayType(TERTIARY_LINK);
        }
        else{
            new_way.setWayType(OTHER);
        }
    }
    else{
        return;
    }
    const char *oneway_tag = way->tags().get_value_by_key("oneway");
    if (oneway_tag && strcmp(oneway_tag, "yes") == 0) {
        new_way.setOneway(true);
    }
    
    // Process Way Geometry
    Projector converter;
    new_way.northings().resize(way_nodes.size());
    new_way.eastings().resize(way_nodes.size());
    int node_idx = 0;
    for (it = way_nodes.begin(); it != way_nodes.end(); ++it) {
        float x;
        float y;
        converter.convertLatlonToXY(it->lat(), it->lon(), x, y);
        new_way.eastings()[node_idx] = x;
        new_way.northings()[node_idx] = y;
        ++node_idx;
    }
    
    osm_ptr_->pushAWay(new_way);
}

void OpenStreetMap::clearData(){
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    is_empty_ = true;
    normalized_vertices_.clear();
    vertex_colors_.clear();
    way_idxs_.clear();
    way_widths_.clear();
    ways_.clear();
}
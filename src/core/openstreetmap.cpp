#include "openstreetmap.h"
#include "latlon_converter.h"
#include <pcl/common/centroid.h>
#include <pcl/search/impl/flann_search.hpp>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;

OsmNode::OsmNode(void){
    degree_ = 0;
}

OsmNode::~OsmNode(void){
    
}

OsmWay::OsmWay(void) {
    is_oneway_ = false;
}

OsmWay::~OsmWay(void){
    
}

OpenStreetMap::OpenStreetMap(QObject *parent) : Renderable(parent),
map_point_cloud_(new PclPointCloud), map_search_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    is_empty_ = true;
    normalized_vertices_.clear();
    vertex_colors_.clear();
    way_idxs_.clear();
    way_widths_.clear();
    ways_.clear();
    nodes_.clear();
    node_idx_map_.clear();
    map_point_cloud_->clear();
}

OpenStreetMap::~OpenStreetMap(){
}

bool OpenStreetMap::loadOSM(const string &filename){
    Osmium::OSMFile infile(filename.c_str());
    MyShapeHandler handler(this);
    ways_.clear();
    
    Osmium::Input::read(infile, handler);
    google::protobuf::ShutdownProtobufLibrary();
    
    QVector4D bound_box = QVector4D(1e10, -1e10, 1e10, -1e10);
    for (size_t i = 0; i < ways_.size(); ++i) {
        OsmWay &aWay = ways_[i];
        for (size_t j = 0; j < aWay.eastings().size(); ++j) {
            if (aWay.eastings()[j] < bound_box[0]) {
                bound_box[0] = aWay.eastings()[j];
            }
            if (aWay.eastings()[j] > bound_box[1]){
                bound_box[1] = aWay.eastings()[j];
            }
            if (aWay.northings()[j] < bound_box[2]) {
                bound_box[2] = aWay.northings()[j];
            }
            if (aWay.northings()[j] > bound_box[3]){
                bound_box[3] = aWay.northings()[j];
            }
        }
    }
    
    SceneConst::getInstance().updateBoundBox(bound_box);
   
    is_empty_ = false;
    checkEquivalency();
    
    return true;
}

bool OpenStreetMap::extractMapBranchingPoints(const string &filename){
    int n_branching_pts = 0;
    vector<double> lat;
    vector<double> lon;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        OsmNode &node = nodes_[i];
        if (node.degree() >= 3) {
            n_branching_pts += 1;
            lat.push_back(node.lat());
            lon.push_back(node.lon());
        }
    }
    printf("There are %d branching points.\n", n_branching_pts);
    ofstream output;
    output.open(filename);
    output << n_branching_pts << endl;
    for(int i = 0; i < n_branching_pts; ++i){
        output.precision(6);
        output.setf( std::ios::fixed, std:: ios::floatfield );
        output << lat[i] << ", " << lon[i] <<endl;
    }
    
    output << nodes_.size() << endl;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        OsmNode &node = nodes_[i];
        output.setf( std::ios::fixed, std:: ios::floatfield );
        output.precision(6);
        output << node.lat() << ", " << node.lon() << endl;
    }
    output.close();
    return true;
}

int OpenStreetMap::findNodeId(uint64_t ref_id){
    map<uint64_t, int>::iterator it;
    it = node_idx_map_.find(ref_id);
    if (it == node_idx_map_.end()) {
        return -1;
    }
    else{
        return it->second;
    }
}

bool OpenStreetMap::insertNode(uint64_t ref_id, double lat, double lon){
    OsmNode newNode;
    newNode.setLatLon(lat, lon);
    
    int node_id = nodes_.size();
    bool result = node_idx_map_.emplace(ref_id, node_id).second;
    if (result){
        nodes_.push_back(newNode);
        return true;
    }
    else{
        return false;
    }
}

void OpenStreetMap::pushAWay(OsmWay &aWay){
    ways_.push_back(aWay);
}

void OpenStreetMap::prepareForVisualization(){
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
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
            return Color(1.0f, 0.7f, 0.4f, 1.0f);
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
    
    if(way_nodes.size() == 0){
        return;
    }
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
    
    int current_way_id = osm_ptr_->ways().size();
    
    // Process Way Geometry
    Projector &converter = Projector::getInstance();
    
    new_way.northings().resize(way_nodes.size());
    new_way.eastings().resize(way_nodes.size());
    new_way.node_ids().resize(way_nodes.size());
    int node_idx = 0;
    for (it = way_nodes.begin(); it != way_nodes.end(); ++it) {
        float x;
        float y;
        int tmp_node_idx = osm_ptr_->findNodeId(it->ref());
        if (tmp_node_idx != -1){
            // Add to node degree
            int delta_degree = 2;
            if (node_idx == 0 || node_idx == way_nodes.size() - 1){
                delta_degree = 1;
            }
            osm_ptr_->nodes()[tmp_node_idx].degree() += delta_degree;
            osm_ptr_->nodes()[tmp_node_idx].way_ids().push_back(current_way_id);
        }
        else{
            // Add a new nodes
            osm_ptr_->insertNode(it->ref(), it->lat(), it->lon());
            int delta_degree = 2;
            if (node_idx == 0 || node_idx == way_nodes.size() - 1){
                delta_degree = 1;
            }
            tmp_node_idx = osm_ptr_->nodes().size() - 1;
            osm_ptr_->nodes().back().degree() += delta_degree;
            osm_ptr_->nodes().back().way_ids().push_back(current_way_id);
        }
        converter.convertLatlonToXY(it->lat(), it->lon(), x, y);
        new_way.eastings()[node_idx] = x;
        new_way.northings()[node_idx] = y;
        new_way.node_ids()[node_idx] = tmp_node_idx;
        ++node_idx;
    }
    
    osm_ptr_->pushAWay(new_way);
}

void OpenStreetMap::clearData(){
    is_empty_ = true;
    normalized_vertices_.clear();
    vertex_colors_.clear();
    way_idxs_.clear();
    way_widths_.clear();
    ways_.clear();
}

void OpenStreetMap::updateMapSearchTree(float grid_size){
    map_point_cloud_->clear();
    
    PclPoint point;
    point.setNormal(0, 0, 1);
    for (size_t i = 0; i < ways_.size(); ++i) {
        OsmWay &aWay = ways_[i];
        if (aWay.eastings().size() == 0)
            continue;
        for (int j = 0; j < aWay.eastings().size()-1; ++j) {
            Vector2d start(aWay.eastings()[j], aWay.northings()[j]);
            Vector2d end(aWay.eastings()[j+1], aWay.northings()[j+1]);
            Vector2d vec = end - start;
            float length = vec.norm();
            vec.normalize();
            float heading = acos(vec[0]) * 180.0f / PI;
            if (vec[1] < 0) {
                heading = 360.0f - heading;
            }
            
            int n_pt = ceil(length / grid_size);
            float delta_length = length / n_pt;
            for (int k = 0; k < n_pt; ++k) {
                Vector2d pt = start + vec * k * delta_length;
                point.setCoordinate(pt.x(), pt.y(), 0.0f);
                point.id_trajectory = i; // This will record the osmWay id
                
                int node_degree = 2;
                if(k == 0){
                    node_degree = nodes_[aWay.node_ids()[j]].degree();
                }
                point.id_sample = node_degree;
                point.head = floor(heading);
                
                map_point_cloud_->push_back(point);
            }
        }
        // Insert the last point
        int last_pt_id = aWay.eastings().size() - 1;
        Vector2d start(aWay.eastings()[last_pt_id-1], aWay.northings()[last_pt_id-1]);
        Vector2d end(aWay.eastings()[last_pt_id], aWay.northings()[last_pt_id]);
        Vector2d vec = end - start;
        vec.normalize();
        float heading = acos(vec[0]) * 180.0f / PI;
        if (vec[1] < 0) {
            heading = 360.0f - heading;
        }
        
        point.setCoordinate(aWay.eastings()[last_pt_id], aWay.northings()[last_pt_id], 0.0f);
        point.id_trajectory = i; // This will record the osmWay id
        point.id_sample = nodes_[aWay.node_ids()[last_pt_id]].degree();
        point.head = floor(heading);
        
        map_point_cloud_->push_back(point);
    }
    map_search_tree_->setInputCloud(map_point_cloud_);
}

bool OpenStreetMap::twoWaysEquivalent(int i, int j){
    if(equivalent_ways_.find(pair<int, int>(i, j)) != equivalent_ways_.end()){
        return true;
    }
    if(equivalent_ways_.find(pair<int, int>(j, i)) != equivalent_ways_.end()){
        return true;
    }
    
    return false;
}

void OpenStreetMap::checkEquivalency(){
    equivalent_ways_.clear();
    for(int i = 0; i < nodes_.size(); ++i){
        OsmNode& this_node = nodes_[i];
        if(this_node.degree() == 2){
            if(this_node.way_ids().size() == 2){
                equivalent_ways_.insert(pair<int, int>(this_node.way_ids()[0], this_node.way_ids()[1]));
                equivalent_ways_.insert(pair<int, int>(this_node.way_ids()[1], this_node.way_ids()[0]));
            }
        }
    }
}
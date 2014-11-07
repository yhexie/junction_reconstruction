#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <boost/filesystem.hpp>
#include "color_map.h"
#include "trajectories.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <queue>
#include "graph.h"
#include "gps_trajectory.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <pcl/common/centroid.h>
#include <pcl/search/impl/flann_search.hpp>

#include "latlon_converter.h"

using namespace Eigen;

static const float search_distance = 250.0f; // in meters
static const float arrow_size = 50.0f;   // in meters
static const int arrow_angle = 15; // in degrees

static const float SPEED_FILTER = 2.5; // meters per second

struct QueueValue{
    int index;
    double dist;
};

class MyComparison{
public:
    MyComparison(){}
    bool operator()(const QueueValue &lhs, const QueueValue& rhs) const{
        return (lhs.dist > rhs.dist);
    }
};

Trajectories::Trajectories(QObject *parent) : Renderable(parent), data_(new PclPointCloud), tree_(new pcl::search::FlannSearch<PclPoint>(false)), distance_graph_vertices_(new PclPointCloud), distance_graph_search_tree_(new pcl::search::FlannSearch<PclPoint>(false)), sample_point_size_(10.0f), sample_color_(Color(0.0f, 0.3f, 0.6f, 1.0f)),samples_(new PclPointCloud), sample_tree_(new pcl::search::FlannSearch<PclPoint>(false)), cluster_centers_(new PclPointCloud), cluster_center_search_tree_(new pcl::search::FlannSearch<PclPoint>(false)), pathlet_data_(new PclPointCloud), pathlet_tree_(new pcl::search::FlannSearch<PclPoint>(false)), single_pathlet_data_(new PclPointCloud), single_pathlet_tree_(new pcl::search::FlannSearch<PclPoint>(false)), map_data_(new PclPointCloud), map_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    point_size_ = 5.0f;
    line_width_ = 3.0f;
    render_mode_ = false;
    selection_mode_ = false;
    show_direction_ = false;
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    show_interpolated_trajectories_ = true;
    
    graph_ = NULL;
    
    // Distance graph
    distance_graph_ = NULL;
    show_distance_graph_ = true;
    
    // Samples
    show_samples_ = true;
    samples_->clear();
    picked_sample_idxs_.clear();
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    cluster_descriptors_ = NULL;
    normalized_cluster_center_locs_.clear();
    cluster_center_colors_.clear();
    
    // Segments
    segments_.clear();
    graph_interpolated_segments_.clear();
    segments_to_draw_.clear();
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    search_range_ = 10.0f;
    
    // Pathlets
    show_pathlets_ = true;
    show_pathlet_direction_ = false;
    pathlet_threshold_ = 0;
    ith_pathlet_to_show_ = -1;
}

Trajectories::~Trajectories(){
    if (distance_graph_ != NULL){
        delete distance_graph_;
    }
    if (cluster_descriptors_ != NULL){
        delete cluster_descriptors_;
    }
}

void Trajectories::draw(){
    if(!render_mode_){
        // Draw selected clusters
        if (selected_cluster_idxs_.size() == 0){
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
            shadder_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&vertex_colors_const_[0], 4*vertex_colors_const_.size()*sizeof(float));
            shadder_program_->setupColorAttributes();
            
            glPointSize(point_size_);
            glDrawArrays(GL_POINTS, 0, normalized_vertices_.size());
        }
        else{
            vector<int> &cluster = point_clusters_[selected_cluster_idxs_[0]];
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
            shadder_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&vertex_colors_const_[0], 4*vertex_colors_const_.size()*sizeof(float));
            shadder_program_->setupColorAttributes();
            
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(cluster[0]), cluster.size()*sizeof(int));
            glDrawElements(GL_POINTS, cluster.size(), GL_UNSIGNED_INT, 0);
        }
        
        // Draw speed and heading
        if (show_direction_){
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&vertex_speed_[0], 3*vertex_speed_.size()*sizeof(float));
            shadder_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&vertex_speed_colors_[0], 4*vertex_speed_colors_.size()*sizeof(float));
            shadder_program_->setupColorAttributes();
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(vertex_speed_indices_[0]), vertex_speed_indices_.size()*sizeof(int));
            glLineWidth(line_width_);
            glDrawElements(GL_LINES, vertex_speed_indices_.size(), GL_UNSIGNED_INT, 0);
        }
    }
    else{
        if(trajectories_.size() == 0)
            return;
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_vertices_[0], 3*normalized_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&vertex_colors_individual_[0], 4*vertex_colors_individual_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        for (int i=0; i < trajectories_.size(); ++i) {
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(trajectories_[i][0]), trajectories_[i].size()*sizeof(size_t));
            glLineWidth(line_width_);
            glDrawElements(GL_LINE_STRIP, trajectories_[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_);
            glDrawElements(GL_POINTS, trajectories_[i].size(), GL_UNSIGNED_INT, 0);
        }
    }
    
    // Draw selected trajectories
    if (selection_mode_){
        drawSelectedTrajectories(selected_trajectories_);
    }else{
        selected_trajectories_.clear();
    }
    
    // Draw Distance Graph
    if (show_distance_graph_ && distance_graph_ != NULL) {
        // Draw Edges
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_distance_graph_vertices_[0], 3*normalized_distance_graph_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&distance_graph_vertex_colors_[0], 4*distance_graph_vertex_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        vector<int> &distance_graph_edge_idxs = distance_graph_->edge_idxs();
        element_buffer.allocate(&(distance_graph_edge_idxs[0]), distance_graph_edge_idxs.size()*sizeof(int));
        glLineWidth(line_width_);
        glDrawElements(GL_LINES, distance_graph_edge_idxs.size(), GL_UNSIGNED_INT, 0);
        
        // Draw Vertices
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&normalized_distance_graph_vertices_[0], 3*normalized_distance_graph_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&distance_graph_vertex_colors_[0], 4*distance_graph_vertex_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        glPointSize(point_size_);
        glDrawArrays(GL_POINTS, 0, normalized_distance_graph_vertices_.size());
    }
    
    // Draw samples
    if (show_samples_){
        if (samples_->size() > 0) {
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&normalized_sample_locs_[0], 3*normalized_sample_locs_.size()*sizeof(float));
            shadder_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&sample_vertex_colors_[0], 4*sample_vertex_colors_.size()*sizeof(float));
            shadder_program_->setupColorAttributes();
            
            glPointSize(sample_point_size_);
            glDrawArrays(GL_POINTS, 0, normalized_sample_locs_.size());
            
            // Draw picked samples
            if (picked_sample_idxs_.size() > 0) {
                vector<Vertex> picked_sample_locs;
                vector<Color> picked_sample_colors;
                for (set<int>::iterator it = picked_sample_idxs_.begin(); it != picked_sample_idxs_.end(); ++it) {
                    int sample_id = *it;
                    Vertex &n_sample_loc = normalized_sample_locs_[sample_id];
                    picked_sample_locs.push_back(Vertex(n_sample_loc.x, n_sample_loc.y, Z_SELECTED_SAMPLES));
                    picked_sample_colors.push_back(Color(1.0, 0.0, 0.0, 1.0));
                }
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&picked_sample_locs[0], 3*picked_sample_locs.size()*sizeof(float));
                shadder_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&picked_sample_colors[0], 4*picked_sample_colors.size()*sizeof(float));
                shadder_program_->setupColorAttributes();
                
                glPointSize(1.5*sample_point_size_);
                glDrawArrays(GL_POINTS, 0, picked_sample_locs.size());
            }
        }
    }
    
    // Draw segments_to_draw_
    show_segments_ = false;
    if (show_segments_){
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&segment_vertices_[0], 3*segment_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&segment_colors_[0], 4*segment_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        for (int i=0; i < segment_idxs_.size(); ++i) {
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(segment_idxs_[i][0]), segment_idxs_[i].size()*sizeof(size_t));
            glLineWidth(line_width_);
            glDrawElements(GL_LINE_STRIP, segment_idxs_[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_);
            glDrawElements(GL_POINTS, segment_idxs_[i].size(), GL_UNSIGNED_INT, 0);
        }
        
        // Draw segment speed and heading
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&segment_speed_[0], 3*segment_speed_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&segment_speed_colors_[0], 4*segment_speed_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(segment_speed_indices_[0]), segment_speed_indices_.size()*sizeof(int));
        glLineWidth(line_width_);
        glDrawElements(GL_LINES, segment_speed_indices_.size(), GL_UNSIGNED_INT, 0);
    }
    
    // Draw pathlet_to_draw_
    if (show_pathlets_){
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&pathlet_vertices_[0], 3*pathlet_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&pathlet_colors_[0], 4*pathlet_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        for (int i = 0; i < pathlet_to_draw_.size(); ++i) {
            if (ith_pathlet_to_show_ != -1 && ith_pathlet_to_show_ != i){
                continue;
            }
            int pathlet_id = pathlet_to_draw_[i];
            vector<int> &pathlet_idx = pathlet_idxs_[pathlet_id];
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(pathlet_idx[0]), pathlet_idx.size()*sizeof(size_t));
            glPointSize(point_size_ * 2.0);
            glDrawElements(GL_POINTS, pathlet_idx.size(), GL_UNSIGNED_INT, 0);
        }
        
        // Draw pathlet speed and heading
        if (show_pathlet_direction_){
            vertexPositionBuffer_.create();
            vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexPositionBuffer_.bind();
            vertexPositionBuffer_.allocate(&pathlet_speed_[0], 3*pathlet_speed_.size()*sizeof(float));
            shadder_program_->setupPositionAttributes();
            
            vertexColorBuffer_.create();
            vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
            vertexColorBuffer_.bind();
            vertexColorBuffer_.allocate(&pathlet_speed_colors_[0], 4*pathlet_speed_colors_.size()*sizeof(float));
            shadder_program_->setupColorAttributes();
            vector<int> all_speed_idxs;
            for (int i = 0; i < pathlet_to_draw_.size(); ++i) {
                if (ith_pathlet_to_show_ != -1 && ith_pathlet_to_show_ != i){
                    continue;
                }
                int pathlet_id = pathlet_to_draw_[i];
                vector<int> &speed_idx = pathlet_speed_indices_[pathlet_id];
                for (int j = 0; j < speed_idx.size(); ++j) {
                    all_speed_idxs.push_back(speed_idx[j]);
                }
            }
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(all_speed_idxs[0]), all_speed_idxs.size()*sizeof(int));
            glLineWidth(line_width_);
            glDrawElements(GL_LINES, all_speed_idxs.size(), GL_UNSIGNED_INT, 0);
        }
    }
}

void Trajectories::prepareForVisualization(QVector4D bound_box){
    display_bound_box_ = bound_box;
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    normalized_sample_locs_.clear();
    normalized_cluster_center_locs_.clear();
    cluster_center_colors_.clear();
    vertex_speed_.clear();
    vertex_speed_indices_.clear();
    vertex_speed_colors_.clear();
    
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (delta_x < 0 || delta_y < 0) {
        fprintf(stderr, "Trajectory bounding box error! Min greater than Max!\n");
    }
    
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    scale_factor_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
    
    for (size_t i=0; i<data_->size(); ++i) {
        const PclPoint &point = data_->at(i);
        float n_x = (point.x - center_x) / scale_factor_;
        float n_y = (point.y - center_y) / scale_factor_;
        normalized_vertices_.push_back(Vertex(n_x, n_y, Z_TRAJECTORIES));
        
        // Speed and heading
        float speed = static_cast<float>(data_->at(i).speed) / 100.0f;
        float heading = static_cast<float>(data_->at(i).head) *PI / 180.0f;
        
        Color tmp_color(0.8, 0.8, 0.8, 1.0);
        if (data_->at(i).head < 45 || data_->at(i).head >= 315 ) {
            tmp_color.r = 1.0f;
            tmp_color.g = 0.0f;
            tmp_color.b = 0.0f;
            tmp_color.alpha = 1.0f;
        }
        else if(data_->at(i).head >= 45 && data_->at(i).head < 135){
            tmp_color.r = 0.0f;
            tmp_color.g = 1.0f;
            tmp_color.b = 0.0f;
            tmp_color.alpha = 1.0f;
        }
        else if(data_->at(i).head >= 135 && data_->at(i).head < 225){
            tmp_color.r = 0.0f;
            tmp_color.g = 0.0f;
            tmp_color.b = 1.0f;
            tmp_color.alpha = 1.0f;
        }
        else if(data_->at(i).head >= 225 && data_->at(i).head < 315){
            tmp_color.r = 1.0f;
            tmp_color.g = 1.0f;
            tmp_color.b = 0.0f;
            tmp_color.alpha = 1.0f;
        }
        
        float speed_line_length = speed / scale_factor_;
        float speed_delta_x = speed_line_length * sin(heading);
        float speed_delta_y = speed_line_length * cos(heading);
        vertex_speed_indices_.push_back(vertex_speed_.size());
        vertex_speed_.push_back(Vertex(n_x, n_y, Z_TRAJECTORY_SPEED));
        vertex_speed_indices_.push_back(vertex_speed_.size());
        vertex_speed_.push_back(Vertex(n_x+speed_delta_x, n_y+speed_delta_y, Z_TRAJECTORY_SPEED));
        
        vertex_speed_colors_.push_back(tmp_color);
        vertex_speed_colors_.push_back(tmp_color);
    }
    
    Color light_gray = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_GRAY);
    vertex_colors_const_.resize(normalized_vertices_.size(), light_gray);
    
    //Color orange = ColorMap::getInstance().getNamedColor(ColorMap::ORANGE);
    //vertex_colors_const_.resize(normalized_vertices_.size(), orange);
    
    vertex_colors_individual_.resize(data_->size(), Color(0,0,0,1));
    for (int i=0; i < trajectories_.size(); ++i) {
        Color traj_color = ColorMap::getInstance().getDiscreteColor(i);
        for (int j = 0; j < trajectories_[i].size(); ++j) {
            int vertex_idx = trajectories_[i][j];
            vertex_colors_individual_[vertex_idx].r = traj_color.r;
            vertex_colors_individual_[vertex_idx].g = traj_color.g;
            vertex_colors_individual_[vertex_idx].b = traj_color.b;
        }
    }
    
    // Prepare for Pathlets
    if (pathlets_.size() > 0) {
        pathlet_vertices_.clear();
        pathlet_colors_.clear();
        pathlet_idxs_.clear();
        pathlet_speed_.clear();
        pathlet_speed_colors_.clear();
        pathlet_speed_indices_.clear();
        pathlet_idxs_.resize(pathlets_.size(), vector<int>());
        for (int i = 0; i < pathlets_.size(); ++i) {
            Color pathlet_color = ColorMap::getInstance().getDiscreteColor(i);
            vector<int> speed_idxs;
            for (int j = 0; j < pathlets_[i].size(); ++j) {
                int seg_id = pathlets_[i][j];
                // Collect segment points
                Segment &seg = segments_[seg_id];
                for (int k = 0; k < seg.points().size(); ++k) {
                    SegPoint &point = seg.points()[k];
                    float n_x = (point.x - center_x) / scale_factor_;
                    float n_y = (point.y - center_y) / scale_factor_;
                    pathlet_idxs_[i].push_back(pathlet_vertices_.size());
                    pathlet_vertices_.push_back(Vertex(n_x, n_y, Z_PATHLETS));
                    pathlet_colors_.push_back(pathlet_color);
                    
                    // Speed and heading
                    float speed = static_cast<float>(point.speed) / 100.0f;
                    float heading = static_cast<float>(point.head) *PI / 180.0f;
                    
                    Color tmp_color(0.8, 0.8, 0.8, 1.0);
                    if (point.head < 45 || point.head >= 315 ) {
                        tmp_color.r = 1.0f;
                        tmp_color.g = 0.0f;
                        tmp_color.b = 0.0f;
                        tmp_color.alpha = 1.0f;
                    }
                    else if(point.head >= 45 && point.head < 135){
                        tmp_color.r = 0.0f;
                        tmp_color.g = 1.0f;
                        tmp_color.b = 0.0f;
                        tmp_color.alpha = 1.0f;
                    }
                    else if(point.head >= 135 && point.head < 225){
                        tmp_color.r = 0.0f;
                        tmp_color.g = 0.0f;
                        tmp_color.b = 1.0f;
                        tmp_color.alpha = 1.0f;
                    }
                    else if(point.head >= 225 && point.head < 315){
                        tmp_color.r = 1.0f;
                        tmp_color.g = 1.0f;
                        tmp_color.b = 0.0f;
                        tmp_color.alpha = 1.0f;
                    }
                    
                    float speed_line_length = speed / scale_factor_;
                    float speed_delta_x = speed_line_length * sin(heading);
                    float speed_delta_y = speed_line_length * cos(heading);
                    
                    speed_idxs.push_back(pathlet_speed_.size());
                    pathlet_speed_.push_back(Vertex(n_x, n_y, Z_TRAJECTORY_SPEED));
                    
                    speed_idxs.push_back(pathlet_speed_.size());
                    pathlet_speed_.push_back(Vertex(n_x+speed_delta_x, n_y+speed_delta_y, Z_TRAJECTORY_SPEED));
                    pathlet_speed_colors_.push_back(tmp_color);
                    pathlet_speed_colors_.push_back(tmp_color);
                }
            }
            pathlet_speed_indices_.push_back(speed_idxs);
        }
    }
    
    printf("totally %lu pathlet points\n", pathlet_vertices_.size());
    
    // Prepare for distance graph
    if (!show_distance_graph_) {
        return;
    }
    else{
        if (distance_graph_ != NULL) {
            normalized_distance_graph_vertices_.clear();
            distance_graph_vertex_colors_.clear();
            normalized_distance_graph_vertices_.resize(distance_graph_vertices_->size());
            Color dark_gray = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
            distance_graph_vertex_colors_.resize(distance_graph_vertices_->size(), dark_gray);
            
            for (size_t i = 0; i < distance_graph_vertices_->size(); ++i){
                PclPoint &loc = distance_graph_vertices_->at(i);
                normalized_distance_graph_vertices_[i].x = (loc.x - center_x) / scale_factor_;
                normalized_distance_graph_vertices_[i].y = (loc.y - center_y) / scale_factor_;
                normalized_distance_graph_vertices_[i].z = Z_DISTANCE_GRAPH;
            }
        }
    }
    
    // Prepare for sample visualization
    if (samples_->size() == 0)
        return;
    
    normalized_sample_locs_.resize(samples_->size());
    
    for (size_t i = 0; i < samples_->size(); ++i){
        PclPoint &loc = samples_->at(i);
        normalized_sample_locs_[i].x = (loc.x - center_x) / scale_factor_;
        normalized_sample_locs_[i].y = (loc.y - center_y) / scale_factor_;
        normalized_sample_locs_[i].z = Z_SAMPLES;
    }
    
    // Prepare for graph visualization
    if (cluster_centers_->size() == 0)
        return;
    
    normalized_cluster_center_locs_.resize(cluster_centers_->size());
    cluster_center_colors_.resize(cluster_centers_->size());
    
    for (size_t i = 0; i < cluster_centers_->size(); ++i){
        PclPoint &loc = cluster_centers_->at(i);
        normalized_cluster_center_locs_[i].x = (loc.x - center_x) / scale_factor_;
        normalized_cluster_center_locs_[i].y = (loc.y - center_y) / scale_factor_;
        normalized_cluster_center_locs_[i].z = Z_DISTANCE_GRAPH;
        cluster_center_colors_[i].r = sample_color_.r;
        cluster_center_colors_[i].g = sample_color_.g;
        cluster_center_colors_[i].b = sample_color_.b;
        cluster_center_colors_[i].alpha = sample_color_.alpha;
    }
}

bool Trajectories::loadTXT(const string& filename)
{
    ifstream fin(filename);
    if (!fin.good())
        return false;
    
    PclPoint point;
    point.setNormal(0, 0, 1);
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    
    int num_trajectory = 0;
    fin >> num_trajectory;
    trajectories_.resize(num_trajectory, vector<int>());
    for (int id_trajectory = 0; id_trajectory < num_trajectory; ++ id_trajectory)
    {
        int num_samples;
        fin >> num_samples;
        vector<int>& trajectory = trajectories_[id_trajectory];
        trajectory.resize(num_samples);
        for (int id_sample = 0; id_sample < num_samples; ++ id_sample)
        {
            double x, y, t;
            int heavy;
            fin >> x >> y >> t >> heavy;
            x -= BJ_X_OFFSET;
            y -= BJ_Y_OFFSET;
            t -= UTC_OFFSET;
            
            if (x < bound_box_[0]) {
                bound_box_[0] = x;
            }
            if (x > bound_box_[1]){
                bound_box_[1] = x;
            }
            if (y < bound_box_[2]) {
                bound_box_[2] = y;
            }
            if (y > bound_box_[3]){
                bound_box_[3] = y;
            }
            
            point.setCoordinate(x, y, 0.0);
            point.t = static_cast<float>(t);
            point.id_trajectory = id_trajectory;
            point.id_sample = id_sample;
            
            trajectory[id_sample] = data_->size();
            data_->push_back(point);
        }
    }
    
    return true;
}

bool Trajectories::loadPBF(const string &filename){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // Prepare the coordinate projector
    Projector &converter = Projector::getInstance();
    bool has_correct_projection = false;
    
    // Read the trajectory collection from file.
    int fid = open(filename.c_str(), O_RDONLY);
    if (fid == -1) {
        fprintf(stderr, "ERROR! Cannot open protobuf trajectory file!\n");
        return false;
    }
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    
    google::protobuf::io::ZeroCopyInputStream *raw_input = new google::protobuf::io::FileInputStream(fid);
    google::protobuf::io::CodedInputStream *coded_input = new google::protobuf::io::CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(1e9, 9e8);
    
    PclPoint point;
    point.setNormal(0, 0, 1);
    
    uint32_t num_trajectory;
    if (!coded_input->ReadLittleEndian32(&num_trajectory)) {
        return false; // possibly end of file
    }
    
    trajectories_.resize(num_trajectory, vector<int>());
    for (size_t id_traj = 0; id_traj < num_trajectory; ++id_traj) {
        uint32_t length;
        if (!coded_input->ReadLittleEndian32(&length)) {
            break; // possibly end of file
        }
        auto limit = coded_input->PushLimit(length);
        GpsTraj new_traj;
        if(!new_traj.MergePartialFromCodedStream(coded_input)){
            fprintf(stderr, "ERROR! Protobuf trajectory file contaminated!\n");
            return false;
        }
        coded_input->PopLimit(limit);
        
        // Add the points to point cloud and trajectory
        vector<int>& trajectory=trajectories_[id_traj];
        trajectory.resize(new_traj.point_size());
        
        if (id_traj == 0) {
            // Check if the x, y field of new_traj point is correct according to current projector. If not, we will reporject all x, y field from lat lon coordinate.
            float x;
            float y;
            converter.convertLatlonToXY(new_traj.point(0).lat()/1.0e5, new_traj.point(0).lon()/1.0e5, x, y);
            
            if ( fabs(x - new_traj.point(0).x()) < 1e-4 &&
                fabs(y - new_traj.point(0).y()) < 1e-4) {
                has_correct_projection = true;
            }
        }
        
        if (has_correct_projection) {
            for (int pt_idx=0; pt_idx < new_traj.point_size(); ++pt_idx) {
                if (new_traj.point(pt_idx).x() < bound_box_[0]) {
                    bound_box_[0] = new_traj.point(pt_idx).x();
                }
                if (new_traj.point(pt_idx).x() > bound_box_[1]) {
                    bound_box_[1] = new_traj.point(pt_idx).x();
                }
                if (new_traj.point(pt_idx).y() < bound_box_[2]) {
                    bound_box_[2] = new_traj.point(pt_idx).y();
                }
                if (new_traj.point(pt_idx).y() > bound_box_[3]) {
                    bound_box_[3] = new_traj.point(pt_idx).y();
                }
                
                point.setCoordinate(new_traj.point(pt_idx).x(), new_traj.point(pt_idx).y(), 0.0);
                point.t = new_traj.point(pt_idx).timestamp();
                point.id_trajectory = id_traj;
                point.id_sample = pt_idx;
                point.car_id = new_traj.point(pt_idx).car_id();
                point.lon = new_traj.point(pt_idx).lon();
                point.lat = new_traj.point(pt_idx).lat();
                point.speed = new_traj.point(pt_idx).speed();
                point.head = new_traj.point(pt_idx).head();
                
                trajectory[pt_idx] = data_->size();
                data_->push_back(point);
            }
        }
        else{
            for (int pt_idx=0; pt_idx < new_traj.point_size(); ++pt_idx) {
                // Check if the x, y field is correct. If not, we will redo the projection from lat lon field.
                float x;
                float y;
                converter.convertLatlonToXY(new_traj.point(pt_idx).lat()/1.0e5, new_traj.point(pt_idx).lon()/1.0e5, x, y);
                
                double t = new_traj.point(pt_idx).timestamp();
                
                if (x < bound_box_[0]) {
                    bound_box_[0] = x;
                }
                if (x > bound_box_[1]){
                    bound_box_[1] = x;
                }
                if (y < bound_box_[2]) {
                    bound_box_[2] = y;
                }
                if (y > bound_box_[3]){
                    bound_box_[3] = y;
                }
                
                point.setCoordinate(x, y, 0.0);
                point.t = t - UTC_OFFSET;
                point.id_trajectory = id_traj;
                point.id_sample = pt_idx;
                point.car_id = new_traj.point(pt_idx).car_id();
                point.lon = new_traj.point(pt_idx).lon();
                point.lat = new_traj.point(pt_idx).lat();
                point.speed = new_traj.point(pt_idx).speed();
                point.head = new_traj.point(pt_idx).head();
                
                trajectory[pt_idx] = data_->size();
                data_->push_back(point);
            }
        }
    }
    
    close(fid);
    delete raw_input;
    delete coded_input;
    printf("Loading completed. %d trajectories loaded. Totally %zu points.\n", num_trajectory, data_->size());
    return true;
}

bool Trajectories::loadPathlets(const string &filename, int &n_selected){
    ifstream fin(filename);
    if (!fin.good())
        return false;
    
    pathlets_.clear();
    segments_.clear();
    selected_pathlets_.clear();
    trajectory_explanations_.clear();
    ith_pathlet_to_show_ = -1;
    
    // Load all pathlets
    int n_pathlets = 0;
    fin >> n_pathlets;
    pathlets_.resize(n_pathlets, vector<int>());
    for (int id_pathlet = 0; id_pathlet < n_pathlets; ++ id_pathlet)
    {
        int n_segments;
        fin >> n_segments;
        
        for (int i_segment = 0; i_segment < n_segments; ++i_segment) {
            int n_points;
            fin >> n_points;
            
            Segment new_seg;
            for (int i_pt = 0; i_pt < n_points; ++i_pt) {
                double x, y, t, head, speed, orig_idx;
                
                fin >> x >> y >> t >> head >> speed >> orig_idx;
                int trunked_head = static_cast<int>(head);
                int trunked_speed = static_cast<int>(speed);
                int trunked_idx = static_cast<int>(orig_idx);
                
                new_seg.points().push_back(SegPoint(x, y, t, trunked_head, trunked_speed, trunked_idx));
            }
            pathlets_[id_pathlet].push_back(segments_.size());
            segments_.push_back(new_seg);
        }
    }
    
    // Load selected pathlet ids
    fin >> n_pathlets;
    pathlet_selected_count_.resize(n_pathlets, 0);
    for (int i = 0; i < n_pathlets; ++i) {
        int count = -1;
        fin >> count;
        pathlet_selected_count_[i] = count;
    }
    
    // Load trajectory explanations
    int n_traj = 0;
    fin >> n_traj;
    trajectory_explanations_.resize(n_traj, vector<int>());
    for (int i = 0; i < n_traj; ++i) {
        int n_pathlet = 0;
        vector<int> &explain = trajectory_explanations_[i];
        fin >> n_pathlet;
        for (int j = 0; j < n_pathlet; ++j) {
            int p_id = 0;
            fin >> p_id;
            explain.push_back(p_id);
        }
    }
    
    // Update selected pathlets
    vector<int> picked_pathlets;
    for (int i = 0; i < n_pathlets; ++i) {
        if (pathlet_selected_count_[i] < pathlet_threshold_) {
            continue;
        }
        picked_pathlets.push_back(i);
    }
    
    updateSelectedPathlets(picked_pathlets);
    updatePathletSearchTree();
    showAllPathlets();
    
    return true;
}

int Trajectories::setPathletThreshold(int new_threshold){
    if (pathlet_selected_count_.size() == 0) {
        return 0;
    }
    pathlet_threshold_ = new_threshold;
    selected_pathlets_.clear();
    pathlet_to_draw_.clear();
    vector<int> picked_pathlets;
    for (int i = 0; i < pathlets_.size(); ++i){
        if (pathlet_selected_count_[i] < pathlet_threshold_){
            continue;
        }
        picked_pathlets.push_back(i);
        pathlet_to_draw_.push_back(i);
    }
    updateSelectedPathlets(picked_pathlets);
    updatePathletSearchTree();
    return selected_pathlets_.size();
}

bool myFunction(pair<int, pair<int, int>> p1, pair<int,pair<int, int>> p2) {
    if (p1.second.first > p2.second.first) {
        return true;
    }
    else if (p1.second.first == p2.second.first){
        if (p1.second.second > p2.second.second) {
            return true;
        }
        else{
            return false;
        }
    }
    else{
        return false;
    }
}

void Trajectories::updateSelectedPathlets(vector<int> &picked_pathlets){
    vector<pair<int, pair<int,int>>> pairs;
    for (int i = 0; i < picked_pathlets.size(); ++i) {
        pairs.push_back(pair<int, pair<int, int>>(picked_pathlets[i], pair<int, int>(pathlet_selected_count_[picked_pathlets[i]], pathlets_[picked_pathlets[i]].size())));
    }
    
    // Sort
    sort(pairs.begin(), pairs.end(), myFunction);
   
    selected_pathlets_.clear();
    for (vector<pair<int, pair<int, int>>>::iterator it = pairs.begin(); it != pairs.end(); ++it) {
        selected_pathlets_.push_back(it->first);
    }
}

void Trajectories::showAllPathlets(){
    pathlet_to_draw_.clear();
    for (int i = 0; i < selected_pathlets_.size(); ++i) {
        pathlet_to_draw_.push_back(selected_pathlets_[i]);
    }
}

void Trajectories::showPathletAt(int idx){
    pathlet_to_draw_.clear();
    pathlet_to_draw_.push_back(selected_pathlets_[idx]);
    printf("Pathlet %d, selected %d times.\n", selected_pathlets_[idx], pathlet_selected_count_[selected_pathlets_[idx]]);
}

void Trajectories::updatePathletSearchTree(){
    pathlet_data_->clear();
    PclPoint point;
    for (int i = 0; i < selected_pathlets_.size(); ++i) {
        int pathlet_id = selected_pathlets_[i];
        vector<int> &pathlet_segment_idxs = pathlets_[pathlet_id];
        for (int j = 0; j < pathlet_segment_idxs.size(); ++j) {
            Segment &seg = segments_[pathlet_segment_idxs[j]];
            for (int k = 0; k < seg.points().size(); ++k) {
                SegPoint &pt = seg.points()[k];
                point.setCoordinate(pt.x, pt.y, 0.0);
                point.id_trajectory = pathlet_id;
                point.speed = pt.speed;
                point.head = pt.head;
                
                pathlet_data_->push_back(point);
            }
        }
    }
    pathlet_tree_->setInputCloud(pathlet_data_);
}

bool Trajectories::savePathlets(const string &filename){
    ofstream output;
    output.open(filename.c_str());
    
    // Output all pathlets
    output << pathlets_.size() << endl;
    for (int i = 0; i < pathlets_.size(); ++i) {
        vector<int> &this_cluster = pathlets_[i];
        output << this_cluster.size() << endl;
        for (int j = 0; j < this_cluster.size(); ++j) {
            Segment &this_seg = segments_[this_cluster[j]];
            output << this_seg.points().size() << endl;
            for (int k = 0; k < this_seg.points().size(); ++k) {
                SegPoint &this_pt = this_seg.points()[k];
                output << setiosflags(ios::fixed) << setprecision(2) << this_pt.x << " " << this_pt.y << " " << this_pt.t << " " << this_pt.head << " " << this_pt.speed << " " << this_pt.orig_idx << endl;
            }
        }
    }
    
    // Output selected pathlet ids
    output << pathlets_.size() << endl;
    for (int i = 0; i < pathlets_.size(); ++i) {
        output << pathlet_selected_count_[i] << endl;
    }
    
    // Output trajectory explanations
    output << trajectories_.size() << endl;
    for (int i = 0; i < trajectories_.size(); ++i){
        output << trajectory_explanations_[i].size() << endl;
        int n = trajectory_explanations_[i].size();
        for (int j = 0; j < n - 1; ++j) {
            output << trajectory_explanations_[i][j] << " ";
        }
        output << trajectory_explanations_[i][n-1] << endl;
    }
    
    output.close();
    return true;
}

bool Trajectories::load(const string& filename)
{
    bool success = false;
    cout << "Loading Trajectories from " << filename << "..."<<endl;
    string extension = boost::filesystem::path(filename).extension().string();
    if (extension == ".txt")
        success = loadTXT(filename);
    else if (extension == ".pcd"){
        //success = loadPCD(filename);
    }
    else if (extension == ".pbf"){
        success = loadPBF(filename);
    }
    
    if (success)
    {
        tree_->setInputCloud(data_);
    }
    
    return true;
}

bool Trajectories::save(const string& filename){
    if (savePBF(filename)){
        printf("%s saved.\n", filename.c_str());
    }
    else{
        printf("File cannot be saved.");
    }
    return true;
}

bool Trajectories::isEmpty(){
    if (data_->size() > 0){
        return false;
    }
    else{
        return true;
    }
}

bool Trajectories::savePBF(const string &filename){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    // Read the trajectory collection from file.
    int fid = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
    
    if (fid == -1) {
        fprintf(stderr, "ERROR! Cannot create protobuf trajectory file!\n");
        return false;
    }
    
    google::protobuf::io::ZeroCopyOutputStream *raw_output = new google::protobuf::io::FileOutputStream(fid);
    google::protobuf::io::CodedOutputStream *coded_output = new google::protobuf::io::CodedOutputStream(raw_output);
    
    uint32_t num_trajectory = trajectories_.size();
    
    coded_output->WriteLittleEndian32(num_trajectory);
    
    for (size_t id_traj = 0; id_traj < num_trajectory; ++id_traj) {
        GpsTraj new_traj;
        for (size_t pt_idx = 0; pt_idx < trajectories_[id_traj].size(); ++pt_idx) {
            TrajPoint* new_pt = new_traj.add_point();
            int pt_idx_in_data = trajectories_[id_traj][pt_idx];
            new_pt->set_car_id(data_->at(pt_idx_in_data).car_id);
            new_pt->set_speed(data_->at(pt_idx_in_data).speed);
            new_pt->set_head(data_->at(pt_idx_in_data).head);
            new_pt->set_lon(data_->at(pt_idx_in_data).lon);
            new_pt->set_lat(data_->at(pt_idx_in_data).lat);
            new_pt->set_x(data_->at(pt_idx_in_data).x);
            new_pt->set_y(data_->at(pt_idx_in_data).y);
            new_pt->set_timestamp(data_->at(pt_idx_in_data).t);
        }
        string s;
        new_traj.SerializeToString(&s);
        coded_output->WriteLittleEndian32(s.size());
        coded_output->WriteString(s);
    }
    delete coded_output;
    delete raw_output;
    close(fid);
    
    return true;
}

bool Trajectories::extractFromFiles(const QStringList &filenames, QVector4D bound_box, PclSearchTree::Ptr &map_search_tree, int nmin_inside_pts){
    data_->clear();
    trajectories_.clear();
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    // Check bound box
    if (bound_box[1] < bound_box[0] || bound_box[3] < bound_box[2]) {
        fprintf(stderr, "Bound Box ERROR when extractng trajectories.\n");
        return false;
    }
    
    for (size_t i = 0; i < filenames.size(); ++i) {
        std::string filename(filenames.at(i).toLocal8Bit().constData());
        printf("\tExtracting from %s\n", filename.c_str());
        Trajectories *new_trajectories = new Trajectories(NULL);
        new_trajectories->load(filename);
        new_trajectories->extractTrajectoriesFromRegion(bound_box, this, map_search_tree, nmin_inside_pts);
        delete new_trajectories;
    }
    
    tree_->setInputCloud(data_);
    
    printf("Totally %lu trajectories, %lu points.\n", trajectories_.size(), data_->size());
    
    return true;
}

bool Trajectories::extractTrajectoriesFromRegion(QVector4D bound_box, Trajectories *container,  PclSearchTree::Ptr &map_search_tree, int min_inside_pts){
    // Find all potential trajectories indices inside using kdtree search
    PclPoint center_point;
    float x_length = bound_box[1] - bound_box[0];
    float y_length = bound_box[3] - bound_box[2];
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    center_point.setCoordinate(center_x, center_y, 0.0f);
    double radius = 0.5*sqrt(x_length*x_length + y_length*y_length);
    
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(center_point, radius, k_indices, k_distances);
    
    set<int> inside_trajectory_idxs;
    for (size_t i = 0; i < k_indices.size(); ++i)
    {
        const PclPoint &point = data_->at(k_indices[i]);
        inside_trajectory_idxs.insert(point.id_trajectory);
    }
    
    for (set<int>::iterator it = inside_trajectory_idxs.begin(); it != inside_trajectory_idxs.end(); ++it) {
        vector<int> &trajectory = trajectories_[*it];
        vector<PclPoint> extracted_trajectory;
        bool recording = false;
        
        // Traverse trajectory to chop segments.
        for (size_t pt_idx = 0; pt_idx < trajectory.size(); ++pt_idx) {
            const PclPoint &point = data_->at(trajectory[pt_idx]);
            if (point.x < bound_box[1] && point.x > bound_box[0] && point.y < bound_box[3] && point.y > bound_box[2]) {
                // Point is inside query bound_box
                // Check if point is near the map
                vector<int> k_idxs;
                vector<float> k_dist_sqrs;
                map_search_tree->nearestKSearch(point, 1, k_idxs, k_dist_sqrs);
                bool to_skip = false;
                if (k_idxs.size() != 0) {
                    if (k_dist_sqrs[0] > 625.0f) {
                        to_skip = true;
                    }
                }
                if (to_skip) {
                    // If it is recording, then this trajectory is not qualified
                    recording = false;
                    extracted_trajectory.clear();
                    continue;
                }
                
                if (!recording) {
                    recording = true;
                    extracted_trajectory.clear();
                    if (pt_idx > 0) {
                        // Insert previous point
                        PclPoint &prev_point = data_->at(trajectory[pt_idx-1]);
                        extracted_trajectory.push_back(prev_point);
                    }
                    // Insert this point
                    extracted_trajectory.push_back(point);
                }
                else{
                    extracted_trajectory.push_back(point);
                }
            }
            else{
                // Point is outside query bound_box
                if (recording) {
                    recording = false;
                    // Insert this point
                    extracted_trajectory.push_back(point);
                    if (extracted_trajectory.size() > min_inside_pts + 2) {
                        if(isQualifiedTrajectory(extracted_trajectory))
                            container->insertNewTrajectory(extracted_trajectory);
                    }
                }
            }
        }
        if (recording){
            if (extracted_trajectory.size() > min_inside_pts + 2) {
                if (isQualifiedTrajectory(extracted_trajectory))
                    container->insertNewTrajectory(extracted_trajectory);
            }
        }
    }
    return true;
}

bool Trajectories::isQualifiedTrajectory(vector<PclPoint> &trajectory){
    // Filter bad trajectories by average speed
    if (trajectory.size() < 2) {
        return false;
    }
    float length = 0.0f;
    for (size_t pt_idx = 0; pt_idx < trajectory.size() - 1; ++pt_idx){
        float delta_x = trajectory[pt_idx+1].x - trajectory[pt_idx].x;
        float delta_y = trajectory[pt_idx+1].y - trajectory[pt_idx].y;
        length += sqrt(delta_x*delta_x + delta_y*delta_y);
    }
    float delta_t = trajectory[trajectory.size()-1].t - trajectory[0].t;
    
    float avg_velocity = length / delta_t;
    if (avg_velocity < SPEED_FILTER) {
        return false;
    }
    return true;
}

bool Trajectories::insertNewTrajectory(vector<PclPoint> &pt_container){
    if (pt_container.empty()) {
        return true;
    }
    
    vector<int> trajectory;
    trajectory.resize(pt_container.size());
    
    for (size_t i = 0; i < pt_container.size(); ++i) {
        PclPoint &point = pt_container[i];
        if (point.x < bound_box_[0]) {
            bound_box_[0] = point.x;
        }
        if (point.x > bound_box_[1]){
            bound_box_[1] = point.x;
        }
        if (point.y < bound_box_[2]) {
            bound_box_[2] = point.y;
        }
        if (point.y > bound_box_[3]){
            bound_box_[3] = point.y;
        }
        
        point.id_trajectory = trajectories_.size();
        point.id_sample = i;
        trajectory[i] = data_->size();
        data_->push_back(point);
    }
    trajectories_.push_back(trajectory);
    return true;
}

void Trajectories::selectNearLocation(float x, float y, float max_distance){
    selected_trajectories_.clear();
    PclPoint center_point;
    center_point.setCoordinate(x, y, 0.0f);
    vector<int> k_indices;
    vector<float> k_sqr_distances;
    
    tree_->nearestKSearch(center_point, 1, k_indices, k_sqr_distances);
    
    if (k_indices.empty())
        return;
    
    PclPoint &selected_point = data_->at(k_indices[0]);
    
    if (sqrt(k_sqr_distances[0]) < search_distance){
        int traj_id = selected_point.id_trajectory;
        selected_trajectories_.push_back(selected_point.id_trajectory);
        printf("Trajectory %d selected.\n", traj_id);
        //for (size_t pt_idx = 0; pt_idx < trajectories_[traj_id].size(); ++pt_idx) {
        //int pt_idx_in_data = trajectories_[traj_id][pt_idx];
        //printf("t = %.2f, x=%.2f, y=%.2f, head=%d, speed=%d\n", data_->at(pt_idx_in_data).t, data_->at(pt_idx_in_data).x, data_->at(pt_idx_in_data).y, data_->at(pt_idx_in_data).head, data_->at(pt_idx_in_data).speed);
        //}
    }
}

void Trajectories::drawSelectedTrajectories(vector<int> &traj_indices){
    if (traj_indices.empty()){
        return;
    }
    
    vector<Vertex>          selected_vertices;
    vector<Color>           selected_vertex_colors;
    vector<vector<int> >    selected_trajectories;
    vector<Vertex>              direction_vertices;
    vector<Color>               direction_colors;
    vector<vector<unsigned> >   direction_idxs;
    
    float arrow_length = arrow_size / scale_factor_;
    
    selected_trajectories.resize(traj_indices.size());
    
    for(size_t idx = 0; idx < traj_indices.size(); ++idx){
        int selected_traj_idx = traj_indices[idx];
        vector<int> &orig_trajectory = trajectories_[selected_traj_idx];
        vector<int> &new_trajectory_indices = selected_trajectories[idx];
        for(size_t pt_idx = 0; pt_idx < orig_trajectory.size(); ++pt_idx){
            new_trajectory_indices.push_back(selected_vertices.size());
            
            Vertex &origVertex = normalized_vertices_[orig_trajectory[pt_idx]];
            Color &origColor = vertex_colors_individual_[orig_trajectory[pt_idx]];
            selected_vertices.push_back(Vertex(origVertex.x, origVertex.y, Z_SELECTED_TRAJ));
            selected_vertex_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
            
            // Prepare for drawing arrow for trajectory
            if (pt_idx != 0) {
                int last_pt_idx = selected_vertices.size() - 1;
                float dx = selected_vertices[last_pt_idx].x - selected_vertices[last_pt_idx-1].x;
                float dy = selected_vertices[last_pt_idx].y - selected_vertices[last_pt_idx-1].y;
                float length = sqrt(dx*dx + dy*dy);
                if (length*scale_factor_ < 3*arrow_size) {
                    continue;
                }
                dx = dx / length * arrow_length;
                dy = dy / length * arrow_length;
                QVector3D vec(dx, dy, 0.0);
                QMatrix4x4 m1;
                m1.setToIdentity();
                m1.rotate(arrow_angle, 0.0f, 0.0f, 1.0f);
                QMatrix4x4 m2;
                m2.setToIdentity();
                m2.rotate(-arrow_angle, 0.0f, 0.0f, 1.0f);
                QVector3D vec1 = m1.map(vec);
                QVector3D vec2 = m2.map(vec);
                float center_x = 0.5*selected_vertices[last_pt_idx].x + 0.5*selected_vertices[last_pt_idx-1].x;
                float center_y = 0.5*selected_vertices[last_pt_idx].y + 0.5*selected_vertices[last_pt_idx-1].y;
                vector<unsigned> direction_idx(3, 0);
                direction_idx[0] = direction_vertices.size();
                direction_vertices.push_back(Vertex(center_x - vec1.x(), center_y - vec1.y(), Z_SELECTED_TRAJ));
                direction_idx[1] = direction_vertices.size();
                direction_vertices.push_back(Vertex(center_x, center_y, Z_SELECTED_TRAJ));
                direction_idx[2] = direction_vertices.size();
                direction_idxs.push_back(direction_idx);
                direction_vertices.push_back(Vertex(center_x - vec2.x(), center_y - vec2.y(), Z_SELECTED_TRAJ));
                direction_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
                direction_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
                direction_colors.push_back(Color(origColor.r, origColor.g, origColor.b, origColor.alpha));
            }
        }
    }
    
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&selected_vertices[0], 3*selected_vertices.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&selected_vertex_colors[0], 4*selected_vertex_colors.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    for (int i=0; i < selected_trajectories.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(selected_trajectories[i][0]), selected_trajectories[i].size()*sizeof(size_t));
        glLineWidth(line_width_*1.5);
        glDrawElements(GL_LINE_STRIP, selected_trajectories[i].size(), GL_UNSIGNED_INT, 0);
        glPointSize(point_size_*1.5);
        glDrawElements(GL_POINTS, selected_trajectories[i].size(), GL_UNSIGNED_INT, 0);
    }
    
    // Draw arrows
    QOpenGLBuffer vertex_buffer;
    vertex_buffer.create();
    vertex_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertex_buffer.bind();
    vertex_buffer.allocate(&direction_vertices[0], 3*direction_vertices.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    QOpenGLBuffer color_buffer;
    color_buffer.create();
    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    color_buffer.bind();
    color_buffer.allocate(&direction_colors[0], 4*direction_colors.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    for (size_t i=0; i < direction_idxs.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(direction_idxs[i][0]), direction_idxs[i].size()*sizeof(unsigned));
        glLineWidth(line_width_*2.5);
        glDrawElements(GL_TRIANGLES, direction_idxs[i].size(), GL_UNSIGNED_INT, 0);
    }
    
    // Draw speed and heading
    vector<int> speed_heading_idxs;
    for (size_t i = 0; i < traj_indices.size(); ++i) {
        vector<int> &traj = trajectories_[traj_indices[i]];
        for (size_t j = 0; j < traj.size(); ++j) {
            int pt_idx = traj[j];
            speed_heading_idxs.push_back(2*pt_idx);
            speed_heading_idxs.push_back(2*pt_idx+1);
        }
    }
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&vertex_speed_[0], 3*vertex_speed_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&vertex_speed_colors_[0], 4*vertex_speed_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
    element_buffer.create();
    element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    element_buffer.bind();
    element_buffer.allocate(&(speed_heading_idxs[0]), speed_heading_idxs.size()*sizeof(int));
    glLineWidth(line_width_);
    glDrawElements(GL_LINES, speed_heading_idxs.size(), GL_UNSIGNED_INT, 0);
    
    // Draw Interpolated trajectories
    if (show_interpolated_trajectories_) {
        if (graph_ == NULL) {
            return;
        }
        
        // Compute interpolation using graph_.
        vector<vector<int> >    interpolated_trajectories(traj_indices.size());
        for (size_t i = 0; i < traj_indices.size(); ++i) {
            int traj_id = traj_indices[i];
            vector<int> &orig_trajectory = trajectories_[traj_id];
            vector<int> &result_traj = interpolated_trajectories[i];
            graph_->shortestPathInterpolation(orig_trajectory, result_traj);
        }
        vector<Vertex>          interpolated_trajectory_vertices;
        vector<Color>           interpolated_trajectory_vertex_colors;
        vector<vector<int>>             interpolated_trajectory_indices;
        
        for (size_t i = 0; i < interpolated_trajectories.size(); ++i) {
            vector<int> &traj = interpolated_trajectories[i];
            vector<int> new_trajectory_indices;
            //Color &origColor = vertex_colors_individual_[orig_trajectory[0]];
            Color origColor = Color(1.0, 0.0, 0, 1);
            for(size_t pt_idx = 0; pt_idx < traj.size(); ++pt_idx){
                float alpha = 1.0 - 0.5 * static_cast<float>(pt_idx) / traj.size();
                new_trajectory_indices.push_back(interpolated_trajectory_vertices.size());
                Vertex &sampleVertex = normalized_cluster_center_locs_[traj[pt_idx]];
                interpolated_trajectory_vertices.push_back(Vertex(sampleVertex.x, sampleVertex.y, 0));
                interpolated_trajectory_vertex_colors.push_back(Color(alpha*origColor.r, alpha*origColor.g, alpha*origColor.b, 1.0));
            }
            interpolated_trajectory_indices.push_back(new_trajectory_indices);
        }
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&interpolated_trajectory_vertices[0], 3*interpolated_trajectory_vertices.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&interpolated_trajectory_vertex_colors[0], 4*interpolated_trajectory_vertex_colors.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        
        for (int i=0; i < interpolated_trajectory_indices.size(); ++i) {
            QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
            element_buffer.create();
            element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
            element_buffer.bind();
            element_buffer.allocate(&(interpolated_trajectory_indices[i][0]), interpolated_trajectory_indices[i].size()*sizeof(size_t));
            glLineWidth(line_width_*2);
            glDrawElements(GL_LINE_STRIP, interpolated_trajectory_indices[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_*2);
            glDrawElements(GL_POINTS, interpolated_trajectory_indices[i].size(), GL_UNSIGNED_INT, 0);
        }
    }
}

void Trajectories::setSelectedTrajectories(vector<int> &selectedIdxs){
    selected_trajectories_.clear();
    selected_trajectories_.resize(selectedIdxs.size());
    for (size_t i = 0; i < selectedIdxs.size(); ++i) {
        selected_trajectories_[i] = selectedIdxs[i];
    }
}

void Trajectories::clearData(void){
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    point_size_ = 5.0f;
    line_width_ = 3.0f;
    render_mode_ = false;
    selection_mode_ = false;
    
    data_->clear();
    trajectories_.clear();
    selected_trajectories_.clear();
    normalized_vertices_.clear();
    vertex_colors_const_.clear();
    vertex_colors_individual_.clear();
    
    // Clear samples
    samples_->clear();
    picked_sample_idxs_.clear();
    ith_pathlet_to_show_ = -1;
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    
    // Clear segments
    segments_.clear();
    segments_to_draw_.clear();
    search_range_ = 0.0f;
    segment_idxs_.clear();
    segment_vertices_.clear();
    segment_colors_.clear();
}

void Trajectories::computeWeightAtScale(int neighborhood_size){
    if (samples_->size() == 0){
        return;
    }
    
    sample_weight_.clear();
    sample_weight_.resize(samples_->size(), 0.0);
    
    clock_t begin = clock();
    
    int start = 10;
    int step = 10;
    int stop = 100;
    
    double max = 0.0;
    
    double sigma_max = 0.2;
    
    for(size_t i = 0; i < samples_->size(); ++i){
        vector<int> k_indices;
        vector<float> k_distances;
        sample_tree_->nearestKSearch(samples_->at(i), stop, k_indices, k_distances);
        
        MatrixXf m(stop, 2);
        for(size_t j = 0; j < stop; ++j){
            m(j, 0) = samples_->at(k_indices[j]).x;
            m(j, 1) = samples_->at(k_indices[j]).y;
        }
        MatrixXf centered = m.rowwise() - m.colwise().mean();
        
        double prev_weight = 0.0f;
        double prev_slop = 1.0f;
        int count = 0;
        for (int neighborhood_size = start; neighborhood_size <= stop; neighborhood_size += step) {
            MatrixXf block = centered.block(0, 0, neighborhood_size, 2);
            MatrixXf cov = (block.adjoint() * block) / double(m.rows());
            
            SelfAdjointEigenSolver<MatrixXf> es(cov);
            double sum_values = es.eigenvalues().sum();
            double tmp_weight;
            if (sum_values < 1e-6) {
                tmp_weight = 0.0f;
            }
            else{
                tmp_weight = es.eigenvalues()(0) / sum_values;
            }
            
            double cur_slop = (tmp_weight - prev_weight) / step;
            if (cur_slop > 4*prev_slop) {
                //printf("i=%lu, neighborhood size: %d\n", i, neighborhood_size);
                break;
            }
            
            if (tmp_weight > sigma_max) {
                sample_weight_[i] += 1.0f;
            }
            
            count += 1;
            prev_weight = tmp_weight;
            prev_slop = cur_slop;
        }
        
        if (sample_weight_[i] > max) {
            max = sample_weight_[i];
        }
    }
    
    
    double w_max = 0.6;
    double w_min = 0.5;
    for (size_t i = 0; i < samples_->size(); ++i) {
        double w = sample_weight_[i] / 9;
        Color pt_color;
        if (w > w_max) {
            pt_color = ColorMap::getInstance().getContinusColor(0.9);
        }
        else if (w < w_min){
            pt_color = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_GRAY);
        }
        else{
            pt_color = ColorMap::getInstance().getContinusColor(0.5);
        }
        
        sample_vertex_colors_[i].r = pt_color.r;
        sample_vertex_colors_[i].g = pt_color.g;
        sample_vertex_colors_[i].b = pt_color.b;
    }
    
    clock_t end = clock();
    
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    printf("time elapsed = %.1f sec.", elapsed_time);
}

void Trajectories::DBSCAN(double eps, int minPts){
    point_clusters_.clear();
    vector<int> noise_cluster;
    point_clusters_.push_back(noise_cluster);
    vector<int> assigned_cluster(data_->size(), -1); // -1: means unassigned; 0: noise; others: regular cluster
    
    set<int> unvisited_pts;
    for (int i = 0; i < data_->size(); ++i) {
        unvisited_pts.insert(i);
    }
    printf("hi, entered DBSCAN. eps = %.2f, minPts = %d\n", eps, minPts);
    clock_t begin = clock();
    while (!unvisited_pts.empty()) {
        int pt_idx = *unvisited_pts.begin();
        // Mark P as visited
        unvisited_pts.erase(unvisited_pts.begin());
        
        // Query neighbor points
        vector<int> neighborhood;
        DBSCANRegionQuery(data_->at(pt_idx), eps, neighborhood);
        
        if (neighborhood.size() < minPts) {
            // Mark P as noise
            noise_cluster.push_back(pt_idx);
        }
        else{
            vector<int> new_cluster;
            int current_cluster_id = point_clusters_.size();
            DBSCANExpandCluster(pt_idx, neighborhood, new_cluster, eps, minPts, unvisited_pts, assigned_cluster, current_cluster_id);
            if (new_cluster.size() >= minPts){
                printf("new cluster size: %lu \n", new_cluster.size());
                point_clusters_.push_back(new_cluster);
                //            if (clusters.size() >= 10) {
                //                break;
                //            }
            }
            else{
                for (size_t i = 0; i < new_cluster.size(); ++i) {
                    noise_cluster.push_back(new_cluster[i]);
                }
            }
        }
    }
    
    clock_t end = clock();
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    printf("time elapsed for DBSCAN = %.1f sec.\n", elapsed_time);
    printf("Totally %lu clusters generated.\n", point_clusters_.size());
    
    // Update point colors with clusters
    for (int i = 0; i < point_clusters_.size(); ++i) {
        vector<int> &cluster = point_clusters_[i];
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(i);
        if (i == 0) {
            cluster_color.r = 0.8f;
            cluster_color.g = 0.8f;
            cluster_color.b = 0.8f;
        }
        for (size_t j = 0; j < cluster.size(); ++j) {
            int pt_idx = cluster[j];
            vertex_colors_const_[pt_idx].r = cluster_color.r;
            vertex_colors_const_[pt_idx].g = cluster_color.g;
            vertex_colors_const_[pt_idx].b = cluster_color.b;
        }
    }
}

void Trajectories::DBSCANExpandCluster(int pt_idx, vector<int> &neighborhood, vector<int> &cluster, double eps, int minPts, set<int> &unvisited_pts, vector<int> &assigned_cluster, int current_cluster_id){
    cluster.clear();
    // Add P to cluster C
    cluster.push_back(pt_idx);
    assigned_cluster[pt_idx] = current_cluster_id;
    
    // Initialize front&rear directions, as well as the front&rear point sets.
    Vector2d front_dir(0.0f, 0.0f);
    Vector2d rear_dir(0.0f, 0.0f);
    Vector2d front_center(0.0f, 0.0f);
    Vector2d rear_center(0.0f, 0.0f);
    set<int> front_pts;
    set<int> rear_pts;
    double angle = static_cast<double>(450 - data_->at(pt_idx).head) * PI / 180.0f;
    Vector2d starting_dir(cos(angle), sin(angle));
    vector<double> dot_values(neighborhood.size(), 0.0f);
    int count = 0;
    double min_proj = INFINITY;
    double max_proj = -1 * INFINITY;
    for (size_t i = 0; i < neighborhood.size(); ++i){
        int this_pt_idx = neighborhood[i];
        Vector2d vec(data_->at(this_pt_idx).x - data_->at(pt_idx).x,
                     data_->at(this_pt_idx).y - data_->at(pt_idx).y);
        dot_values[count] = vec.dot(starting_dir);
        if (dot_values[count] < min_proj) {
            min_proj = dot_values[count];
        }
        if (dot_values[count] > max_proj)
        {
            max_proj = dot_values[count];
        }
        count += 1;
    }
    // Determine the front and rear directions and loc
    double ratio = 0.8;
    double delta = ratio * 0.5 *(max_proj - min_proj);
    double min_threshold = min_proj + delta;
    double max_threshold = max_proj - delta;
    for (size_t i = 0; i < neighborhood.size(); ++i) {
        if (dot_values[i] <= min_threshold) {
            // Insert the corresponding point to rear
            rear_pts.insert(neighborhood[i]);
        }
        if (dot_values[i] >= max_threshold){
            // Insert the corresponding point to front
            front_pts.insert(neighborhood[i]);
        }
    }
    if (min_proj >= -1e-6) {
        rear_pts.clear();
    }
    if (max_proj <= 1e-6) {
        front_pts.clear();
    }
    
    // Compute frond direction and center as average of front and end point sets.
    if (!front_pts.empty()) {
        for (set<int>::iterator it = front_pts.begin(); it != front_pts.end(); ++it) {
            int front_pt_idx = *it;
            double front_pt_angle = static_cast<double>(450 - data_->at(front_pt_idx).head) * PI / 180.0f;
            front_dir.x() += cos(front_pt_angle);
            front_dir.y() += sin(front_pt_angle);
            front_center.x() += data_->at(front_pt_idx).x;
            front_center.y() += data_->at(front_pt_idx).y;
        }
        front_dir.normalize();
        front_center.x() /= front_pts.size();
        front_center.y() /= front_pts.size();
    }
    
    if (!rear_pts.empty()) {
        for (set<int>::iterator it = rear_pts.begin(); it != rear_pts.end(); ++it) {
            int rear_pt_idx = *it;
            double rear_pt_angle = static_cast<double>(450 - data_->at(rear_pt_idx).head) * PI / 180.0f;
            rear_dir.x() += cos(rear_pt_angle);
            rear_dir.y() += sin(rear_pt_angle);
            rear_center.x() += data_->at(rear_pt_idx).x;
            rear_center.y() += data_->at(rear_pt_idx).y;
        }
        rear_dir.normalize();
        rear_center.x() /= front_pts.size();
        rear_center.y() /= front_pts.size();
    }
    
    while (!front_pts.empty() || !rear_pts.empty()){
        if (!front_pts.empty()) {
            // Extend from front point sets
            set<int> front_neighbors;
            for (set<int>::iterator pt_iterator = front_pts.begin(); pt_iterator != front_pts.end(); ++pt_iterator) {
                int another_pt_idx = *pt_iterator;
                if (another_pt_idx != pt_idx){
                    set<int>::iterator it = unvisited_pts.find(another_pt_idx);
                    if (it != unvisited_pts.end()) {
                        // This point P' is unvisited, mark it as visited
                        unvisited_pts.erase(it);
                        vector<int> neighbor_pts;
                        DBSCANRegionQuery(data_->at(another_pt_idx), eps, neighbor_pts);
                        if (neighbor_pts.size() >= minPts) {
                            // Insert it to neightor point sets
                            for (size_t i = 0; i < neighbor_pts.size(); ++i) {
                                front_neighbors.insert(neighbor_pts[i]);
                            }
                        }
                    }
                }
            }
            
            // Foreach P' in P's neighborhood
            for (set<int>::iterator pt_iterator = front_neighbors.begin(); pt_iterator != front_neighbors.end(); ++pt_iterator) {
                int another_pt_idx = *pt_iterator;
                // If P' is not yet member of any cluster, add P' to cluster C
                if (assigned_cluster[another_pt_idx] == -1){
                    cluster.push_back(another_pt_idx);
                    assigned_cluster[another_pt_idx] = current_cluster_id;
                }
            }
            
            // Recompute front point sets
            front_pts.clear();
            vector<double> front_dot_values(front_neighbors.size(), 0.0f);
            int count = 0;
            double front_min_proj = INFINITY;
            double front_max_proj = -1 * INFINITY;
            for (set<int>::iterator it = front_neighbors.begin(); it != front_neighbors.end(); ++it) {
                int this_pt_idx = *it;
                Vector2d vec(data_->at(this_pt_idx).x - front_center.x(),
                             data_->at(this_pt_idx).y - front_center.y());
                front_dot_values[count] = vec.dot(front_dir);
                if (dot_values[count] < front_min_proj) {
                    front_min_proj = front_dot_values[count];
                }
                if (front_dot_values[count] > front_max_proj)
                {
                    front_max_proj = front_dot_values[count];
                }
                count += 1;
            }
            
            // Determine the front and rear directions and loc
            double front_delta = ratio * 0.5 *(front_max_proj - front_min_proj);
            double front_max_threshold = front_max_proj - front_delta;
            count = 0;
            for (set<int>::iterator it = front_neighbors.begin(); it != front_neighbors.end(); ++it) {
                if (front_dot_values[count] >= front_max_threshold) {
                    front_pts.insert(*it);
                }
                count += 1;
            }
            if (front_max_proj <= 1e-6) {
                front_pts.clear();
            }
            
            // Compute front direction and center as average of front and end point sets.
            front_dir.x() = 0.0f;
            front_dir.y() = 0.0f;
            front_center.x() = 0.0f;
            front_center.y() = 0.0f;
            if (!front_pts.empty()) {
                for (set<int>::iterator it = front_pts.begin(); it != front_pts.end(); ++it) {
                    int front_pt_idx = *it;
                    double front_pt_angle = static_cast<double>(450 - data_->at(front_pt_idx).head) * PI / 180.0f;
                    front_dir.x() += cos(front_pt_angle);
                    front_dir.y() += sin(front_pt_angle);
                    front_center.x() += data_->at(front_pt_idx).x;
                    front_center.y() += data_->at(front_pt_idx).y;
                }
                front_dir.normalize();
                front_center.x() /= front_pts.size();
                front_center.y() /= front_pts.size();
            }
        }
        if (!rear_pts.empty()){
            // Extend from rear point sets
            set<int> rear_neighbors;
            for (set<int>::iterator pt_iterator = rear_pts.begin(); pt_iterator != rear_pts.end(); ++pt_iterator) {
                int another_pt_idx = *pt_iterator;
                if (another_pt_idx != pt_idx){
                    set<int>::iterator it = unvisited_pts.find(another_pt_idx);
                    if (it != unvisited_pts.end()) {
                        // This point P' is unvisited, mark it as visited
                        unvisited_pts.erase(it);
                        vector<int> neighbor_pts;
                        DBSCANRegionQuery(data_->at(another_pt_idx), eps, neighbor_pts);
                        if (neighbor_pts.size() >= minPts) {
                            // Insert it to neightor point sets
                            for (size_t i = 0; i < neighbor_pts.size(); ++i) {
                                rear_neighbors.insert(neighbor_pts[i]);
                            }
                        }
                    }
                }
            }
            
            // Foreach P' in P's neighborhood
            for (set<int>::iterator pt_iterator = rear_neighbors.begin(); pt_iterator != rear_neighbors.end(); ++pt_iterator) {
                int another_pt_idx = *pt_iterator;
                // If P' is not yet member of any cluster, add P' to cluster C
                if (assigned_cluster[another_pt_idx] == -1){
                    cluster.push_back(another_pt_idx);
                    assigned_cluster[another_pt_idx] = current_cluster_id;
                }
            }
            
            // Recompute rear point sets
            rear_pts.clear();
            vector<double> rear_dot_values(rear_neighbors.size(), 0.0f);
            int count = 0;
            double rear_min_proj = INFINITY;
            double rear_max_proj = -1 * INFINITY;
            for (set<int>::iterator it = rear_neighbors.begin(); it != rear_neighbors.end(); ++it) {
                int this_pt_idx = *it;
                Vector2d vec(data_->at(this_pt_idx).x - rear_center.x(),
                             data_->at(this_pt_idx).y - rear_center.y());
                rear_dot_values[count] = vec.dot(rear_dir);
                if (rear_dot_values[count] < rear_min_proj) {
                    rear_min_proj = rear_dot_values[count];
                }
                if (rear_dot_values[count] > rear_max_proj)
                {
                    rear_max_proj = rear_dot_values[count];
                }
                count += 1;
            }
            
            // Determine the front and rear directions and loc
            double rear_delta = ratio * 0.5 *(rear_max_proj - rear_min_proj);
            double rear_min_threshold = rear_min_proj + rear_delta;
            count = 0;
            for (set<int>::iterator it = rear_neighbors.begin(); it != rear_neighbors.end(); ++it) {
                if (rear_dot_values[count] >= rear_min_threshold) {
                    rear_pts.insert(*it);
                }
                count += 1;
            }
            if (rear_max_proj <= 1e-6) {
                rear_pts.clear();
            }
            
            // Compute rear direction and center as average of front and end point sets.
            rear_dir.x() = 0.0f;
            rear_dir.y() = 0.0f;
            rear_center.x() = 0.0f;
            rear_center.y() = 0.0f;
            if (!rear_pts.empty()) {
                for (set<int>::iterator it = rear_pts.begin(); it != rear_pts.end(); ++it) {
                    int rear_pt_idx = *it;
                    double rear_pt_angle = static_cast<double>(450 - data_->at(rear_pt_idx).head) * PI / 180.0f;
                    rear_dir.x() += cos(rear_pt_angle);
                    rear_dir.y() += sin(rear_pt_angle);
                    rear_center.x() += data_->at(rear_pt_idx).x;
                    rear_center.y() += data_->at(rear_pt_idx).y;
                }
                rear_dir.normalize();
                rear_center.x() /= rear_pts.size();
                rear_center.y() /= rear_pts.size();
            }
        }
    }
}

void Trajectories::DBSCANRegionQuery(PclPoint &p, double eps, vector<int> &neighborhood){
    neighborhood.clear();
    //    if (p.speed < 10)
    //        return;
    double angle = static_cast<double>(450 - p.head) * PI / 180.0f;
    Vector2d dir(cos(angle), sin(angle));
    vector<int> nearby_pt_idxs;
    vector<float> pt_dist_squares;
    //int K = 50;
    //double dist_threshold_square = 100.0; // in m^2
    //tree_->nearestKSearch(p, K, nearby_pt_idxs, pt_dist_squares);
    tree_->radiusSearch(p, 5.0, nearby_pt_idxs, pt_dist_squares);
    
    double eps_to_rad = eps * PI / 180.f;
    double threshold = cos(eps_to_rad);
    
    // Only add direction compatible points to neighborhood
    for (size_t i = 0; i < nearby_pt_idxs.size(); ++i) {
        //        if (pt_dist_squares[i] > dist_threshold_square) {
        //            continue;
        //        }
        int pt_idx = nearby_pt_idxs[i];
        //        if (data_->at(pt_idx).speed < 10)
        //            continue;
        double pt_angle = static_cast<double>(450 - data_->at(pt_idx).head) * PI / 180.0f;
        Vector2d pt_dir(cos(pt_angle), sin(pt_angle));
        double dot_value = pt_dir.dot(dir);
        if (dot_value > threshold) {
            neighborhood.push_back(pt_idx);
        }
    }
}

void Trajectories::sampleDBSCANClusters(float neighborhood_size){
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    
    // Clear samples
    PclPoint point;
    samples_->clear();
    picked_sample_idxs_.clear();
    
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    cluster_samples_.clear();
    
    // Compute samples for each clusters
    PclPointCloud::Ptr cluster_points(new PclPointCloud);
    PclSearchTree::Ptr cluster_points_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    for (size_t i = 1; i < point_clusters_.size(); ++i) {
        vector<int> &cluster = point_clusters_[i];
        vector<int> this_cluster_sample_idxs;
        Color cluster_sample_color = ColorMap::getInstance().getDiscreteColor(i);
        // Find max_x, min_x, max_y, min_y
        float min_x = INFINITY;
        float min_y = INFINITY;
        float max_x = -1 * INFINITY;
        float max_y = -1 * INFINITY;
        for (size_t j = 0; j < cluster.size(); ++j) {
            int pt_idx = cluster[j];
            if (data_->at(pt_idx).x < min_x) {
                min_x = data_->at(pt_idx).x;
            }
            if (data_->at(pt_idx).y < min_y) {
                min_y = data_->at(pt_idx).y;
            }
            if (data_->at(pt_idx).x > max_x) {
                max_x = data_->at(pt_idx).x;
            }
            if (data_->at(pt_idx).y > max_y) {
                max_y = data_->at(pt_idx).y;
            }
        }
        
        delta_x = max_x - min_x;
        delta_y = max_y - min_y;
        // Initialize the samples using a unified grid
        vector<Vertex> seed_samples;
        set<int> grid_idxs;
        int n_x = static_cast<int>(delta_x / neighborhood_size) + 1;
        int n_y = static_cast<int>(delta_y / neighborhood_size) + 1;
        float step_x = delta_x / n_x;
        float step_y = delta_y / n_y;
        
        cluster_points->clear();
        for (size_t j = 0; j < cluster.size(); ++j) {
            int pt_idx = cluster[j];
            PclPoint &point = data_->at(pt_idx);
            cluster_points->push_back(point);
            int pt_i = floor((point.x - min_x) / step_x);
            int pt_j = floor((point.y - min_y) / step_y);
            if (pt_i == n_x){
                pt_i -= 1;
            }
            if (pt_j == n_y) {
                pt_j -= 1;
            }
            int idx = pt_i + pt_j*n_x;
            grid_idxs.insert(idx);
        }
        
        cluster_points_search_tree->setInputCloud(cluster_points);
        for (set<int>::iterator it = grid_idxs.begin(); it != grid_idxs.end(); ++it) {
            int idx = *it;
            int pt_i = idx % n_x;
            int pt_j = idx / n_x;
            float pt_x = min_x + (pt_i+0.5)*step_x;
            float pt_y = min_y + (pt_j+0.5)*step_y;
            seed_samples.push_back(Vertex(pt_x, pt_y, 0.0));
        }
        
        int n_iter = 1;
        float search_radius = (step_x > step_y) ? step_x : step_y;
        search_radius /= sqrt(2.0);
        search_radius += 1.0;
        for (size_t i_iter = 0; i_iter < n_iter; ++i_iter) {
            // For each seed, search its neighbors and do an average
            for (vector<Vertex>::iterator it = seed_samples.begin(); it != seed_samples.end(); ++it) {
                PclPoint search_point;
                search_point.setCoordinate((*it).x, (*it).y, 0.0f);
                // Search nearby and mark nearby
                vector<int> k_indices;
                vector<float> k_distances;
                cluster_points_search_tree->radiusSearch(search_point, search_radius, k_indices, k_distances);
                
                if (k_indices.size() == 0) {
                    printf("WARNING!!! Empty sampling cell!\n");
                    continue;
                }
                float sum_x = 0.0;
                float sum_y = 0.0;
                int sum_angle = 0;
                int sum_speed = 0;
                for (vector<int>::iterator neighbor_it = k_indices.begin(); neighbor_it != k_indices.end(); ++neighbor_it) {
                    PclPoint &nb_pt = cluster_points->at(*neighbor_it);
                    sum_x += nb_pt.x;
                    sum_y += nb_pt.y;
                    sum_angle += nb_pt.head;
                    sum_speed += nb_pt.speed;
                }
                
                sum_x /= k_indices.size();
                sum_y /= k_indices.size();
                sum_angle /= k_indices.size();
                sum_speed /= k_indices.size();
                
                point.setCoordinate(sum_x, sum_y, 0.0f);
                point.t = 0;
                point.id_trajectory = 0;
                point.id_sample = i;
                point.head = cluster_points->at(k_indices[0]).head;
                point.speed = sum_speed;
                
                this_cluster_sample_idxs.push_back(samples_->size());
                samples_->push_back(point);
                sample_vertex_colors_.push_back(cluster_sample_color);
            }
        }
        cluster_samples_.push_back(this_cluster_sample_idxs);
    }
    
    sample_tree_->setInputCloud(samples_);
    printf("Sampling Done. Totally %zu samples.\n", samples_->size());
    cluster_centers_ = samples_;
    cluster_center_search_tree_ = sample_tree_;
}

void Trajectories::selectDBSCANClusterAt(int clusterId){
    selected_cluster_idxs_.clear();
    selected_cluster_idxs_.push_back(clusterId);
}

void Trajectories::showAllDBSCANClusters(){
    selected_cluster_idxs_.clear();
}

void Trajectories::cutTraj(){
    ofstream test;
    test.open("test_cut_traj.txt");
    test << trajectories_.size()<<endl;
    float MAX_SEGMENT_LENGTH = 1000.0f; // in meters
    segments_.clear();
    segment_id_lookup_.resize(data_->size(), vector<int>());
    clock_t begin = clock();
    for (size_t s = 0; s < trajectories_.size(); ++s) {
        vector<int> &selected_traj = trajectories_[s];
        vector<float> cut_strengths(selected_traj.size(), 0.0f);
        float sigma1 = 10.0f; // Distance sigma, in meters
        float sigma2 = 500.0f; // Speed sigma, e.g., 500 cm/sec
        // Compute cut strength at each trajectory point
        for (size_t i = 0; i < selected_traj.size(); ++i) {
            int pt_idx = selected_traj[i];
            vector<int> nearby_pt_idxs;
            vector<float> nearby_pt_dist_sqrts;
            tree_->radiusSearch(data_->at(pt_idx), 10.0f, nearby_pt_idxs, nearby_pt_dist_sqrts);
            for (size_t j = 0; j < nearby_pt_idxs.size(); ++j) {
                int nearby_pt_idx = nearby_pt_idxs[j];
                if (data_->at(nearby_pt_idx).id_trajectory == data_->at(pt_idx).id_trajectory){
                    continue;
                }
                
                float nearby_pt_dist_sqrt = nearby_pt_dist_sqrts[j];
                int delta_theta = abs(data_->at(pt_idx).head - data_->at(nearby_pt_idx).head);
                if (delta_theta > 180) {
                    delta_theta = 360 - delta_theta;
                }
                float angle = delta_theta / 180.0f * PI;
                float this_strength = (1 - cos(angle)) * expf(-1.0f * nearby_pt_dist_sqrt / sigma1 / sigma1) * expf(-1.0f * data_->at(nearby_pt_idx).speed / sigma2);
                
                if (this_strength < 1e-4)
                    this_strength = 0.0f;
                cut_strengths[i] += this_strength;
            }
        }
        
        // Look for cut point
        vector<int> peaks;
        peakDetector(cut_strengths, 5, 5, 1.2, peaks);
        
        // Cut the trajectory into pieces
        for (int i = 0; i < peaks.size() - 1; ++i) {
            for (int j = i+1; j < peaks.size(); ++j) {
                // The new segment will have points from (i to j)
                // Add points from prev_idx to nxt_idx as a new segment
                Segment new_segment;
                float cummulated_length = 0.0f;
                int start = peaks[i];
                int end = peaks[j];
                
                for (int s = start; s <= end; ++s) {
                    int orig_pt_idx = selected_traj[s];
                    PclPoint &this_point = data_->at(orig_pt_idx);
                    if (s > i) {
                        float delta_x = this_point.x - data_->at(selected_traj[s-1]).x;
                        float delta_y = this_point.y - data_->at(selected_traj[s-1]).y;
                        float delta_length = sqrt(delta_x*delta_x + delta_y*delta_y);
                        cummulated_length += delta_length;
                    }
                    new_segment.points().push_back(SegPoint(this_point.x, this_point.y, this_point.t, this_point.head, this_point.speed, orig_pt_idx));
                }
                if (cummulated_length > MAX_SEGMENT_LENGTH) {
                    continue;
                }
                new_segment.segLength() = cummulated_length;
                int new_segment_id = segments_.size();
                segments_.push_back(new_segment);
                // Update reverse segment id lookup
                for (int s = 0; s < new_segment.points().size(); ++s){
                    int orig_pt_idx = new_segment.points()[s].orig_idx;
                    vector<int> &pt_segment_lists = segment_id_lookup_[orig_pt_idx];
                    pt_segment_lists.push_back(new_segment_id);
                }
            }
        }
        
        test<<selected_traj.size()<<endl;
        test.precision(6);
        test.setf( std::ios::fixed, std:: ios::floatfield );
        for (size_t i = 0; i < selected_traj.size(); ++i) {
            int pt_idx = selected_traj[i];
            test << data_->at(pt_idx).x << ", " << data_->at(pt_idx).y << ", " <<cut_strengths[i] <<endl;;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout <<"Time elapsed: " << elapsed_secs <<endl;
    cout <<"Totally there are " << segments_.size() << " segments." << endl;
    
    // Write generated segments to file
    ofstream output;
    output.open("test_cut_segments.txt");
    output<< segments_.size()<<endl;
    for (int i = 0; i < segments_.size(); ++i) {
        Segment &this_seg = segments_[i];
        output << this_seg.points().size() << endl;
        for (int j = 0; j < this_seg.points().size(); ++j) {
            SegPoint &this_pt = this_seg.points()[j];
            output << this_pt.x << ", " << this_pt.y << endl;
        }
    }
    output.close();
    test.close();
}

void Trajectories::mergePathlet(){
    pathlets_.clear();
    
    // Simplify segments using Douglas-Peucker algorithm
    simplified_segments_.clear();
    //    ofstream out;
    //    out.open("test_simplify_segments.txt");
    //    out<< segments_.size()<<endl;
    //    for (int s = 0; s < segments_.size(); ++s){
    //        Segment &this_seg = segments_[s];
    //        out << this_seg.points().size() << endl;
    //        for (int j = 0; j < this_seg.points().size(); ++j) {
    //            SegPoint &this_pt = this_seg.points()[j];
    //            out << this_pt.x <<", " << this_pt.y << endl;
    //        }
    //    }
    //    out<< segments_.size()<<endl;
    for (int s = 0; s < segments_.size(); ++s) {
        Segment selected_seg = segments_[s];
        
        vector<int> results;
        douglasPeucker(0, selected_seg.points().size()-1, 7.5, selected_seg, results);
        
        Segment new_segment;
        for (int i = 0; i < results.size(); ++i) {
            SegPoint &pt = selected_seg.points()[results[i]];
            new_segment.points().push_back(pt);
        }
        new_segment.segLength() = selected_seg.segLength();
        simplified_segments_.push_back(new_segment);
    }
    
    //        Segment &this_seg = segments_[s];
    //        out << results.size() << endl;
    //        for (int j = 0; j < results.size(); ++j) {
    //            int pt_idx = results[j];
    //            SegPoint &this_pt = this_seg.points()[pt_idx];
    //            out << this_pt.x << ", " << this_pt.y << endl;
    //        }
    //    out.close();
    
    float DISTANCE_THRESHOLD = 15.0f; // meter
    
    set<int> unmerged_pathlets;
    for (int i = 0; i < segments_.size(); ++i) {
        unmerged_pathlets.insert(i);
    }
    
    while (unmerged_pathlets.size() != 0){
        // Pick
        int selected_segment_idx = *unmerged_pathlets.begin();
        
        vector<int> new_cluster;
        
        // Expand
        expandPathletCluster(selected_segment_idx, DISTANCE_THRESHOLD, new_cluster);
        
        // Store
        pathlets_.push_back(new_cluster);
        
        // Remove
        for (int i = 0; i < new_cluster.size(); ++i) {
            unmerged_pathlets.erase(new_cluster[i]);
        }
        
        cout << "\t" << unmerged_pathlets.size() << " remining." << endl;
    }
    cout << "Merged clusters: " << pathlets_.size() << endl;
    
    // Output results
    ofstream output;
    output.open("test_merging_segments.txt");
    output<< pathlets_.size()<<endl;
    for (int i = 0; i < pathlets_.size(); ++i) {
        vector<int> &this_cluster = pathlets_[i];
        output << this_cluster.size() << endl;
        for (int j = 0; j < this_cluster.size(); ++j) {
            Segment &this_seg = segments_[this_cluster[j]];
            output << this_seg.points().size() << endl;
            for (int k = 0; k < this_seg.points().size(); ++k) {
                SegPoint &this_pt = this_seg.points()[k];
                output << this_pt.x << ", " << this_pt.y << endl;
            }
        }
    }
    output.close();
    
    // Update pathlet score, and hashing
    pathlet_scores_.resize(pathlets_.size(), 0.0f);
    traj_pathlets_.clear();
    for (int i = 0; i < trajectories_.size(); ++i) {
        traj_pathlets_.insert(pair<int, set<int>>(i, set<int>()));
    }
    pathlet_trajs_.clear();
    for (int i = 0; i < pathlets_.size(); ++i) {
        pathlet_trajs_.insert(pair<int, set<int>>(i, set<int>()));
    }
    
    pathlet_explained_.clear();
    pathlet_explained_.resize(pathlets_.size(), map<int, pair<int, int>>());
    for (int i = 0; i < pathlets_.size(); ++i){
        vector<int> &current_pathlet = pathlets_[i];
        set<int> &pathlet_traj_hook = pathlet_trajs_[i];
        map<int, pair<int, int>> &this_pathlet_can_explain = pathlet_explained_[i];
        map<int, pair<int, int>>::iterator it;
        for (int j = 0; j < current_pathlet.size(); ++j) {
            Segment &current_segment = segments_[current_pathlet[j]];
            int segment_size = current_segment.points().size();
            if (segment_size < 2) {
                continue;
            }
            int orig_start_idx_in_data = current_segment.points()[0].orig_idx;
            int orig_end_idx_in_data = current_segment.points()[segment_size-1].orig_idx;
            
            if (data_->at(orig_start_idx_in_data).id_trajectory != data_->at(orig_end_idx_in_data).id_trajectory) {
                printf("ERROR! Segment doesn't belong to the same trajectory!\n");
                exit(1);
            }
            int orig_traj_id = data_->at(orig_start_idx_in_data).id_trajectory;
            set<int> &traj_pathlet_hook = traj_pathlets_[orig_traj_id];
            traj_pathlet_hook.insert(i);
            pathlet_traj_hook.insert(orig_traj_id);
            
            // Record the part the pathlet can explain for the trajectory
            int corresponding_start = -1;
            int corresponding_end = -1;
            vector<int> &current_traj = trajectories_[orig_traj_id];
            for (int s = 0; s < current_traj.size(); ++s) {
                if (current_traj[s] == orig_start_idx_in_data) {
                    corresponding_start = s;
                }
            }
            for (int s = 0; s < current_traj.size(); ++s) {
                if (current_traj[s] == orig_end_idx_in_data) {
                    corresponding_end = s;
                }
            }
            
            if (corresponding_start == -1 || corresponding_end == -1) {
                printf("ERROR! Cannot find corresponding start or end!\n");
                exit(1);
            }
            
            if (corresponding_start >= corresponding_end) {
                printf("ERROR! Corresponding start greater than end!\n");
                exit(1);
            }
            
            it = this_pathlet_can_explain.find(orig_traj_id);
            if (it == this_pathlet_can_explain.end()) {
                // We will insert new
                pair<int, int> new_pair(corresponding_start, corresponding_end);
                this_pathlet_can_explain.insert(pair<int, pair<int, int>>(orig_traj_id, new_pair));
            }
            else{
                // We will join
                pair<int, int> &portion = it->second;
                if (portion.first > corresponding_start) {
                    portion.first = corresponding_start;
                }
                if (portion.second < corresponding_end) {
                    portion.second = corresponding_end;
                }
            }
        }
    }
    
    for (int i = 0; i < pathlets_.size(); ++i) {
        pathlet_scores_[i] = static_cast<float>(pathlet_trajs_[i].size());
    }
    
    // Debug
    //for (int i = 0; i < trajectories_.size(); ++i){
    //    printf("trajectory %d has %lu points\n", i, trajectories_[i].size());
    //    map<int, set<int>>::iterator it = traj_pathlets_.find(i);
    //    set<int> &this_pathlets = it->second;
    //    for(set<int>::iterator tmp = this_pathlets.begin(); tmp != this_pathlets.end(); ++tmp){
    //        int pathlet_id = *tmp;
    //        map<int, pair<int, int>> &explained = pathlet_explained_[pathlet_id];
    //
    //        map<int, pair<int, int>>::iterator fit = explained.find(i);
    //        pair<int, int> &range = fit->second;
    //
    //        printf("\tpathlet %d, weight = %.6f, covered %d to %d\n", pathlet_id, pathlet_scores_[pathlet_id], range.first, range.second);
    //    }
    //}
}

int Trajectories::selectPathlet(){
    selected_pathlets_.clear();
    if (pathlets_.size() == 0){
        printf("Please generate segments and merge them first.\n");
        return 0;
    }
    
    // Build graph for each trajectory
    vector<vector<vector<int>>>     G_edges;
    vector<vector<vector<int>>>     G_edge_weight_idxs;
    vector<float>                   pathlet_weights(pathlets_.size(), 0.0f);
    
    float LAMBDA = 0.0f;
    for (int i = 0; i < trajectories_.size(); ++i) {
        vector<int> &current_traj = trajectories_[i];
        vector<vector<int>> edges(current_traj.size(), vector<int>());
        vector<vector<int>> edge_weight_idxs(current_traj.size(), vector<int>());
        
        map<int, set<int>>::iterator it = traj_pathlets_.find(i);
        if (it == traj_pathlets_.end()) {
            printf("WARNING: cannot find trajectory pathlet list.\n");
            continue;
        }
        set<int> &p_lists= it->second;
        // Each pathlet of this trajectory corresponds to an edge
        for (set<int>::iterator p_it = p_lists.begin(); p_it != p_lists.end(); ++p_it) {
            int pathlet_id = *p_it;
            map<int, pair<int, int>> &p_explained = pathlet_explained_[pathlet_id];
            map<int, pair<int, int>>::iterator region_it = p_explained.find(i);
            if (region_it == p_explained.end()) {
                printf("ERROR! Cannot find the desired trajectory that this pathlet should be able to explain!\n");
                exit(1);
            }
            pair<int, int> &region = region_it->second;
            
            edges[region.first].push_back(region.second);
            edge_weight_idxs[region.first].push_back(pathlet_id);
            
            float score;
            if (pathlet_scores_[pathlet_id] > 2) {
                score = pathlet_scores_[pathlet_id];
            }
            else{
                score = 0.1;
            }
            
            //pathlet_weights[pathlet_id] = LAMBDA + 1.0 / pathlet_scores_[pathlet_id];
            pathlet_weights[pathlet_id] = LAMBDA + 1.0 / score;
        }
        G_edges.push_back(edges);
        G_edge_weight_idxs.push_back(edge_weight_idxs);
    }
    
    pathlet_selected_count_.resize(pathlets_.size(), 0);
    trajectory_explanations_.clear();
    trajectory_explanations_.resize(trajectories_.size(), vector<int>());
    for (int i = 0; i < trajectories_.size(); ++i) {
        // For each trajectory, run shortest path
        vector<int> chosen_pathlets;
        shortestPathSelection(G_edges[i], G_edge_weight_idxs[i], pathlet_weights, chosen_pathlets);
        vector<int> &explain = trajectory_explanations_[i];
        
        for (int s = chosen_pathlets.size() - 1; s>=0; --s) {
            int pathlet_id = chosen_pathlets[s];
            explain.push_back(pathlet_id);
            pathlet_selected_count_[pathlet_id] += 1;
        }
    }
    
    // Insert into selected_pathlet
    vector<int> picked_pathlets;
    for (int i = 0; i < pathlets_.size(); ++i){
        if (pathlet_selected_count_[i] <= pathlet_threshold_) {
            continue;
        }
        picked_pathlets.push_back(i);
    }
    
    updateSelectedPathlets(picked_pathlets);
    updatePathletSearchTree();
    showAllPathlets();
    return selected_pathlets_.size();
    
    //ofstream output;
    //output.open("test_selected_pathlets.txt");
    //output<< n_recorded_pathlets <<endl;
    //for (int i = 0; i < pathlets_.size(); ++i) {
    //    if (pathlet_selected_count[i] <= 1) {
    //        continue;
    //    }
    //    vector<int> &this_cluster = pathlets_[i];
    //    output << this_cluster.size() << endl;
    //    for (int j = 0; j < this_cluster.size(); ++j) {
    //        Segment &this_seg = segments_[this_cluster[j]];
    //        output << this_seg.points().size() << endl;
    //        for (int k = 0; k < this_seg.points().size(); ++k) {
    //            SegPoint &this_pt = this_seg.points()[k];
    //            output << this_pt.x << ", " << this_pt.y << endl;
    //        }
    //    }
    //}
    //output.close();
}

bool Trajectories::shortestPathSelection(vector<vector<int>> &edges, vector<vector<int>> &edge_weight_idxs, vector<float> &pathlet_weights, vector<int> &chosen_pathlets){
    // Implement Dijkstra's Algorithm
    chosen_pathlets.clear();
    if (edges.size() == 0){
        return true;
    }
    
    //printf("Traj length %lu\n", edges.size());
    //for (int i = 0; i < edges.size(); ++i) {
    //    for (int j = 0; j < edges[i].size(); ++j) {
    //        int to_vertex = edges[i][j];
    //        int weight_idx = edge_weight_idxs[i][j];
    //        printf("\t %d to %d, weight = %.6f\n", i, to_vertex, pathlet_weights[weight_idx]);
    //    }
    //}
    
    int n_node = edges.size();
    vector<double> dist(n_node, 1e16);
    vector<bool> visited(n_node, false);
    dist[0] = 0.0f;
    vector<QueueValue> element(n_node);
    for (int i = 0; i < n_node; ++i) {
        element[i].index = i;
        element[i].dist = dist[i];
    }
    
    
    priority_queue<QueueValue, vector<QueueValue>, MyComparison> queue(element.begin(), element.end());
    vector<int> previous(n_node, -1);
    
    while (!queue.empty()){
        QueueValue u = queue.top();
        if (u.index == n_node - 1) {
            break;
        }
        
        if (visited[u.index]) {
            queue.pop();
            continue;
        }
        
        visited[u.index] = true;
        
        if (dist[u.index] > 1e15) {
            return false;
        }
        
        vector<int> &to_vertices = edges[u.index];
        vector<int> &to_edge_weight_idx = edge_weight_idxs[u.index];
        for (int i = 0; i < to_vertices.size(); ++i) {
            int pathlet_id = to_edge_weight_idx[i];
            int to_vertex_id = to_vertices[i];
            float edge_weight = pathlet_weights[pathlet_id];
            double alt = dist[u.index] + edge_weight;
            if (alt < dist[to_vertex_id]) {
                dist[to_vertex_id] = alt;
                previous[to_vertex_id] = u.index;
                QueueValue tmp;
                tmp.index = to_vertex_id;
                tmp.dist = alt;
                queue.push(tmp);
            }
        }
        queue.pop();
    }
    
    // Trace the chosen pathlets
    int u = n_node - 1;
    while (previous[u] != -1){
        int first_node = previous[u];
        int selected_pathlet = -1;
        vector<int> &this_edges = edges[first_node];
        for (int i = 0; i < this_edges.size(); ++i) {
            if (this_edges[i] == u) {
                selected_pathlet = edge_weight_idxs[first_node][i];
            }
        }
        
        if (selected_pathlet == -1) {
            printf("ERROR! Pathlet error!\n");
            exit(1);
        }
        chosen_pathlets.push_back(selected_pathlet);
        
        u = first_node;
    }
    
    return true;
}

void Trajectories::douglasPeucker(int start_idx, int end_idx, float epsilon, Segment &seg, vector<int> &results){
    results.clear();
    float dmax = 0.0f;
    int index = 0;
    
    Vector2d v(seg.points()[end_idx].x - seg.points()[start_idx].x,
               seg.points()[end_idx].y - seg.points()[start_idx].y);
    v.normalize();
    for (int i = start_idx+1; i < end_idx; ++i){
        Vector2d v1(seg.points()[i].x - seg.points()[start_idx].x,
                    seg.points()[i].y - seg.points()[start_idx].y);
        float projection = v1.dot(v);
        float length = v1.norm();
        float dist = sqrt(length*length - projection*projection);
        if (dist > dmax) {
            dmax = dist;
            index = i;
        }
    }
    
    if (dmax > epsilon) {
        // Recursive call
        vector<int> result1;
        douglasPeucker(start_idx, index, epsilon, seg, result1);
        vector<int> result2;
        douglasPeucker(index, end_idx, epsilon, seg, result2);
        
        if (result1.size() > 0) {
            for (int s = 0; s < result1.size()-1; ++s) {
                results.push_back(result1[s]);
            }
        }
        if (result2.size() > 0) {
            for (int s = 0; s < result2.size(); ++s) {
                results.push_back(result2[s]);
            }
        }
    }
    else{
        results.push_back(start_idx);
        results.push_back(end_idx);
    }
}

void Trajectories::expandPathletCluster(int starting_segment_idx, float distance_threshold, vector<int> &result){
    float CONSIDERED_AS_SHORT = 100.0f; // in meter
    float AMPLIFYING_RATIO = 3.0f; // for segments that's shorter than
    
    set<int> cluster_segment_idxs;
    cluster_segment_idxs.insert(starting_segment_idx);
    set<int> candidate_pathlets;
    findNearbyPathlets(starting_segment_idx, candidate_pathlets);
    while (!candidate_pathlets.empty()) {
        int candidate_idx = *candidate_pathlets.begin();
        candidate_pathlets.erase(candidate_idx);
        set<int>::iterator finder = cluster_segment_idxs.find(candidate_idx);
        if (finder != cluster_segment_idxs.end()) {
            continue;
        }
        
        // Check the distance between pathlet candidate_idx and existing ones
        float min_distance = POSITIVE_INFINITY;
        for (set<int>::iterator it = cluster_segment_idxs.begin(); it != cluster_segment_idxs.end(); ++it) {
            float dist = computeHausdorffDistance(*it, candidate_idx);
            if (dist < min_distance) {
                min_distance = dist;
            }
        }
        
        if (segments_[starting_segment_idx].segLength() < CONSIDERED_AS_SHORT) {
            min_distance *= AMPLIFYING_RATIO;
        }
        
        if (min_distance < distance_threshold) {
            // Add this pathlet to the cluster
            cluster_segment_idxs.insert(candidate_idx);
            
            // Add the neighbors of this pathlet to candidate
            set<int> tmp_candidate_pathlets;
            findNearbyPathlets(candidate_idx, tmp_candidate_pathlets);
            for (set<int>::iterator tmp_it = tmp_candidate_pathlets.begin(); tmp_it != tmp_candidate_pathlets.end(); ++tmp_it) {
                candidate_pathlets.insert(*tmp_it);
            }
        }
    }
    
    result.clear();
    for (set<int>::iterator it = cluster_segment_idxs.begin(); it != cluster_segment_idxs.end(); ++it) {
        result.push_back(*it);
    }
}

void Trajectories::findNearbyPathlets(int segment_idx, set<int> &nearby_pathlet_idxs){
    float SEARCH_RADIUS = 10.0f;
    float ANGLE_THRESHOLD = 15.0f;
    float this_segment_length = segments_[segment_idx].segLength();
    Segment &this_seg = segments_[segment_idx];
    PclPoint search_point;
    nearby_pathlet_idxs.clear();
    for (int i = 0; i < this_seg.points().size(); ++i) {
        SegPoint &this_pt = this_seg.points()[i];
        search_point.setCoordinate(this_pt.x, this_pt.y, 0.0f);
        vector<int> nearby_pt_idxs;
        vector<float> nearby_dist_sqrts;
        tree_->radiusSearch(search_point, SEARCH_RADIUS, nearby_pt_idxs, nearby_dist_sqrts);
        
        for (int j = 0; j < nearby_pt_idxs.size(); ++j) {
            int nb_idx = nearby_pt_idxs[j];
            // Check angle difference
            int angle_difference = abs(this_pt.head - data_->at(nb_idx).head);
            if (angle_difference > 180){
                angle_difference = 360 - angle_difference;
            }
            if (angle_difference > ANGLE_THRESHOLD) {
                continue;
            }
            vector<int> &corresponding_segments = segment_id_lookup_[nb_idx];
            for (int k = 0; k < corresponding_segments.size(); ++k) {
                int corresponding_segment_idx = corresponding_segments[k];
                if (corresponding_segment_idx == segment_idx) {
                    continue;
                }
                float corresponding_segment_length = segments_[corresponding_segment_idx].segLength();
                // Check length difference
                float delta_dist = abs(corresponding_segment_length - this_segment_length);
                if (delta_dist > 25.0f) {
                    continue;
                }
                nearby_pathlet_idxs.insert(corresponding_segment_idx);
            }
        }
    }
}

float Trajectories::computeHausdorffDistance(int seg1_idx, int seg2_idx){
    if (simplified_segments_.size() == 0) {
        printf("ERROR! Please compute simplified segments first!\n");
        exit(1);
    }
    float d1 = onewayHausdorffDistance(seg1_idx, seg2_idx);
    float d2 = onewayHausdorffDistance(seg2_idx, seg1_idx);
    
    return (d1 > d2) ? d1 : d2;
}

float Trajectories::onewayHausdorffDistance(int from_idx, int to_idx){
    if (simplified_segments_.size() == 0) {
        printf("ERROR! Please compute simplified segments first!\n");
        exit(1);
    }
    float dist = -1 * POSITIVE_INFINITY;
    
    Segment &from_seg = simplified_segments_[from_idx];
    Segment &to_seg = simplified_segments_[to_idx];
    
    if (to_seg.points().size() <= 1) {
        return 0.0f;
    }
    
    for (int i = 0; i < from_seg.points().size(); ++i) {
        SegPoint &from_pt = from_seg.points()[i];
        float tmp_dist = POSITIVE_INFINITY;
        for(int j = 0; j < to_seg.points().size() - 1; ++j){
            SegPoint &to_v1 = to_seg.points()[j];
            SegPoint &to_v2 = to_seg.points()[j+1];
            Vector2d v(to_v2.x - to_v1.x,
                       to_v2.y - to_v1.y);
            float v_length = v.norm();
            v.normalize();
            Vector2d v1(from_pt.x - to_v1.x,
                        from_pt.y - to_v1.y);
            float projection = v1.dot(v);
            float pt_to_segment_dist = 0.0f;
            if (projection < 0) {
                float delta_x = from_pt.x - to_v1.x;
                float delta_y = from_pt.y - to_v1.y;
                pt_to_segment_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            else if(projection > v_length){
                float delta_x = from_pt.x - to_v2.x;
                float delta_y = from_pt.y - to_v2.y;
                pt_to_segment_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
            }
            else{
                float v1_length = v1.norm();
                pt_to_segment_dist = sqrt(v1_length*v1_length - projection*projection);
            }
            if(tmp_dist > pt_to_segment_dist){
                tmp_dist = pt_to_segment_dist;
            }
        }
        if (tmp_dist > dist) {
            dist = tmp_dist;
        }
    }
    
    return dist;
}

void Trajectories::peakDetector(vector<float> &values, int smooth_factor, int window_size, float ratio, vector<int> &detected_peaks){
    detected_peaks.clear();
    if (values.size() <= 2){
        return;
    }
    vector<float> smoothed_values(values.size(), 0.0f);
    int left = floor(smooth_factor / 2);
    int right = smooth_factor - left;
    
    // Smooth the value using smooth_factor
    for (int i = 0; i < values.size(); ++i){
        int count = 1;
        smoothed_values[i] = values[i];
        int start = i - left;
        int end = i + right;
        for (int j = start; j <= end; ++j){
            if (j < 0 || j > values.size() - 1){
                continue;
            }
            count += 1;
            smoothed_values[i] += values[j];
        }
        smoothed_values[i] /= count;
    }
    
    detected_peaks.push_back(0);
    // Detect peaks
    left = floor(window_size / 2);
    right = window_size - left;
    for (int i = 1; i < values.size() - 1; ++i){
        int start = i - left;
        int end = i + right;
        if (start < 1){
            start = 1;
        }
        if (end >= values.size() - 1){
            end = values.size() - 2;
        }
        float min_value = POSITIVE_INFINITY;
        float avg_value = 0.0f;
        int count = 0;
        bool is_max = true;
        for (int j = start; j <= end; ++j) {
            if (smoothed_values[j] > smoothed_values[i]) {
                is_max = false;
                break;
            }
            if (smoothed_values[j] < min_value) {
                min_value = smoothed_values[j];
            }
            avg_value += smoothed_values[j];
            count += 1;
        }
        
        if (is_max) {
            if (count == 0) {
                continue;
            }
            avg_value /= count;
            float v1 = smoothed_values[i] - min_value;
            float v2 = avg_value - min_value;
            if (abs(v2 - v1) < 1.0)
                continue;
            if (v1 > ratio * v2){
                detected_peaks.push_back(i);
            }
        }
    }
    detected_peaks.push_back(values.size() - 1);
}

void Trajectories::computeSegments(float extension){
    /*  Compute segment for each GPS point by extension to both direction.
     Args:
     - extension: in meters. For example, extension = 50 will result in extending the point in both direction by at least 50 meters, and the resulting segment will be around 100 meters.
     */
    
    // For each GPS point, compute its segment
    
    clock_t begin = clock();
    segments_.resize(data_->size());
    for (vector<vector<int>>::iterator traj_itr = trajectories_.begin(); traj_itr != trajectories().end(); ++traj_itr) {
        vector<int> &traj = *traj_itr;
        int prev_idx = 0;
        int nxt_idx = 0;
        int prev_pt_idx_in_data = -1;
        float cum_prev_to_current = 0.0f;
        float cum_current_to_nxt = 0.0f;
        for (size_t pt_idx = 0; pt_idx < traj.size(); ++pt_idx) {
            int pt_idx_in_data = traj[pt_idx];
            if (pt_idx > 0) {
                float delta_x = data_->at(pt_idx_in_data).x - data_->at(prev_pt_idx_in_data).x;
                float delta_y = data_->at(pt_idx_in_data).y - data_->at(prev_pt_idx_in_data).y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                cum_prev_to_current += delta_dist;
                cum_current_to_nxt -= delta_dist;
            }
            
            // Check if prev_idx needs to move
            while (cum_prev_to_current > extension && prev_idx < traj.size() - 1) {
                float delta_x = data_->at(traj[prev_idx+1]).x - data_->at(traj[prev_idx]).x;
                float delta_y = data_->at(traj[prev_idx+1]).y - data_->at(traj[prev_idx]).y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                if (cum_prev_to_current - delta_dist > extension) {
                    ++prev_idx;
                    cum_prev_to_current -= delta_dist;
                }
                else{
                    break;
                }
            }
            
            // Check if nxt_idx needs to move
            while (cum_current_to_nxt < extension && nxt_idx < traj.size() - 1) {
                float delta_x = data_->at(traj[nxt_idx + 1]).x - data_->at(traj[nxt_idx]).x;
                float delta_y = data_->at(traj[nxt_idx + 1]).y - data_->at(traj[nxt_idx]).y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                cum_current_to_nxt += delta_dist;
                ++nxt_idx;
                if (cum_current_to_nxt > extension) {
                    break;
                }
            }
            
            // Add points from prev_idx to nxt_idx as a new segment
            Segment &this_segment = segments_[pt_idx_in_data];
            this_segment.segLength() = cum_prev_to_current + cum_current_to_nxt;
            //printf("Extracted one segment: from %d to %d, prev_to_cur = %.1f m, cur_to_nxt = %.1f\n", prev_idx, nxt_idx, cum_prev_to_current, cum_current_to_nxt);
            float current_time = data_->at(traj[pt_idx]).t;
            for (size_t croped_idx = prev_idx; croped_idx <= nxt_idx; ++croped_idx) {
                PclPoint &this_point = data_->at(traj[croped_idx]);
                this_segment.points().push_back(SegPoint(this_point.x, this_point.y, this_point.t - current_time, traj[croped_idx]));
            }
            
            if (cum_prev_to_current > 10 * extension || cum_current_to_nxt > 10 * extension){
                this_segment.quality() = false;
            }
            
            prev_pt_idx_in_data = pt_idx_in_data;
        }
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Computing segments Done. Time elapsed: %.1f sec\n", elapsed_secs);
    
    //    auto highres_begin = chrono::high_resolution_clock::now();
    //    int from = 0;
    //    int to = 1;
    //    float distance;
    //    for (int tmp = 0; tmp < 10000; ++tmp) {
    //        distance = segments_[from].distanceTo(segments_[to]);
    //    }
    //    auto highres_end = chrono::high_resolution_clock::now();
    //    auto duration = chrono::duration_cast<chrono::nanoseconds>(highres_end - highres_begin).count();
    //    printf("Distance from %d to %d is %.2f\n", from, to, distance);
    //    cout<<"ns total: "<<duration << " ns."<<endl;
    
    //    ofstream test;
    //    test.open("test.txt");
    //    int selected_traj_idx = 0;
    //    test<<trajectories_[selected_traj_idx].size()<<endl;
    //    for (size_t i = 0; i < trajectories_[selected_traj_idx].size(); ++i) {
    //        vector<int> &traj = trajectories_[selected_traj_idx];
    //        test<<data_->at(traj[i]).x<<" "<<data_->at(traj[i]).y<<endl;
    //    }
    //
    //    for (size_t i = 0; i < trajectories_[selected_traj_idx].size(); ++i) {
    //        vector<int> &traj = trajectories_[selected_traj_idx];
    //        int idx = traj[i];
    //        test<<segments_[idx].points().size()<<endl;
    //        for (size_t j=0; j < segments_[idx].points().size(); ++j ) {
    //            test<<segments_[idx].points()[j].x<<" "<<segments_[idx].points()[j].y<<endl;
    //        }
    //    }
    //
    //    test.close();
}

void Trajectories::selectPathletsWithSearchRange(float range){
    search_range_ = range;
    pathlet_to_draw_.clear();
    if (!show_pathlets_)
        return;
    
    if (pathlet_data_->size() == 0) {
        return;
    }
    
    if (picked_sample_idxs_.size() == 0){
        return;
    }
    
    set<int> nearby_pathlets;
    
    for (set<int>::iterator it = picked_sample_idxs_.begin(); it != picked_sample_idxs_.end(); ++it) {
        
        PclPoint &sample_loc = samples_->at(*it);
        vector<int> k_indices;
        vector<float> k_dist_sqrts;
        
        pathlet_tree_->radiusSearch(sample_loc, search_range_, k_indices, k_dist_sqrts);
        
        for (size_t i = 0; i < k_indices.size(); ++i) {
            PclPoint &pt = pathlet_data_->at(k_indices[i]);
            int pathlet_id = pt.id_trajectory;
            nearby_pathlets.insert(pathlet_id);
        }
    }
    
    for (set<int>::iterator it = nearby_pathlets.begin(); it != nearby_pathlets.end(); ++it) {
        pathlet_to_draw_.push_back(*it);
    }
}

void Trajectories::showIthPathlet(int ith){
    if (ith >= 0 && ith < pathlet_to_draw_.size()){
        ith_pathlet_to_show_ = ith;
    }
    else{
        ith_pathlet_to_show_ = -1;
    }
}

void Trajectories::generateRoads(){
    // Generate road for pathlets
    PclPoint point;
    float SEARCH_RADIUS = 30.0f;
    float INTERPOLATION = 5.0f;
    vector<Segment> road_centers;
    vector<float> half_road_widths;
    
    PclPoint                map_point;
    //map_point.id_trajectory will be the road id!!!
    vector<pair<int, int>>  edge_lists;
    vector<vector<int>>     map_roads;
    vector<float>           map_road_width;
    int                     curr_n_road = 0;
    map_data_->clear();
    
    for (size_t i = 0; i < selected_pathlets_.size(); ++i){
        int selected_pathlet_id = selected_pathlets_[i];
        if (pathlet_selected_count_[selected_pathlet_id] == 1){
            if (pathlets_[selected_pathlet_id].size() < 2) {
                break;
            }
        }
        vector<int> &pathlet_seg_idxs = pathlets_[selected_pathlet_id];
        single_pathlet_data_->clear();
        // Find the longest segments
        float current_max_length = -1.0;
        int current_max_id = -1;
        for (vector<int>::iterator it = pathlet_seg_idxs.begin(); it != pathlet_seg_idxs.end(); ++it){
            Segment &seg = segments_[*it];
            if (seg.segLength() > current_max_length) {
                current_max_id = *it;
            }
          
            for (int k = 0; k < seg.points().size(); ++k) {
                SegPoint &pt = seg.points()[k];
                point.setCoordinate(pt.x, pt.y, 0.0f);
                point.head = pt.head;
                point.speed = pt.speed;
                single_pathlet_data_->push_back(point);
            }
        }
        if (current_max_id == -1) {
            continue;
        }
        single_pathlet_tree_->setInputCloud(single_pathlet_data_);
        
        // Find average paths
        Segment &seg = segments_[current_max_id];
        Segment average_path;
        for (size_t j = 0; j < seg.points().size(); ++j) {
            point.setCoordinate(seg.points()[j].x, seg.points()[j].y, 0.0f);
            // Search nearby
            vector<int> k_indices;
            vector<float> k_dist_sqrts;
            single_pathlet_tree_->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dist_sqrts);
            
            if (k_indices.size() == 0){
                continue;
            }
            
            vector<float> xs;
            vector<float> ys;
          
            int n_points = 0;
            float avg_x = 0.0f;
            float avg_y = 0.0f;
            for (size_t k = 0; k < k_indices.size(); ++k) {
                int query_angle = seg.points()[j].head;
                int pt_angle = single_pathlet_data_->at(k_indices[k]).head;
                
                float delta_angle = abs(query_angle - pt_angle);
                if (delta_angle > 180) {
                    delta_angle -= 180;
                }
              
                if (delta_angle > 30) {
                    continue;
                }
                
                xs.push_back(single_pathlet_data_->at(k_indices[k]).x);
                ys.push_back(single_pathlet_data_->at(k_indices[k]).y);
            }
            
            if(xs.size() < 2 || ys.size() < 2){
                continue;
            }
            sort(xs.begin(), xs.end());
            sort(ys.begin(), ys.end());
            n_points = xs.size();
            int med = n_points / 2;
            avg_x = xs[med];
            avg_y = ys[med];
            
            average_path.points().push_back(SegPoint(avg_x, avg_y, 0, 0, 0, 0));
        }
        
        if (average_path.points().size() < 2){
            continue;
        }
        // Simplify the resulting path
        vector<int> results;
        douglasPeucker(0, average_path.points().size()-1, 2.5, average_path, results);
        Segment simplified_path;
        for (int j = 0; j < results.size(); ++j) {
            SegPoint &pt = average_path.points()[results[j]];
            simplified_path.points().push_back(pt);
        }
        
        // Generate dense roads, and compute road width
        Segment road_center;
        vector<int> half_width_count(30, 0);
        if (simplified_path.points().size() <= 1) {
            continue;
        }
        int n_simplified_pt = simplified_path.points().size();
        int this_head;
        for (int j = 0; j < simplified_path.points().size() - 1; ++j) {
            SegPoint &p1 = simplified_path.points()[j];
            SegPoint &p2 = simplified_path.points()[j+1];
            road_center.points().push_back(p1);
            Vector2d v(p2.x - p1.x,
                       p2.y - p1.y);
            float length = v.norm();
            v.normalize();
            float angle = acos(v[0] / length) / PI * 360;
            if (v[1] < 0) {
                angle = 306 - angle;
            }
            float f_head = 450 - angle;
            if (f_head > 360) {
                f_head -= 360;
            }
            this_head = floor(f_head);
            
            int n_pt_to_insert = floor(length / INTERPOLATION);
            Vector2d delta = v / n_pt_to_insert * length;
            for (int k = 1; k <= n_pt_to_insert; ++k) {
                road_center.points().push_back(SegPoint(p1.x + k*delta[0], p1.y + k*delta[1], 0, this_head, 0, 0));
            }
        }
        // insert the last point
        SegPoint &last_pt = simplified_path.points()[n_simplified_pt-1];
        road_center.points().push_back(SegPoint(last_pt.x, last_pt.y, 0, this_head, 0, 0));
        
        // compute road width
        for (int j = 0; j < single_pathlet_data_->size(); ++j) {
            PclPoint &pt = single_pathlet_data_->at(j);
            SegPoint query_point(pt.x, pt.y, 0, 0, 0, 0);
            float dist = POSITIVE_INFINITY;
            for (int k = 0; k < n_simplified_pt-1; ++k) {
                float tmp_dist = pointToSegmentDistance(query_point, simplified_path.points()[k], simplified_path.points()[k+1]);
                if (tmp_dist < dist) {
                    dist = tmp_dist;
                }
            }
            int i_dist = floor(dist);
            if (i_dist < 20) {
                half_width_count[i_dist] += 1;
            }
        }
        
        float half_road_width = 1.0f;
        int n_total_points = single_pathlet_data_->size();
        int sum = 0;
        for (int j = 0; j < half_width_count.size(); ++j) {
            sum += half_width_count[j];
            if (sum > 0.8*n_total_points) {
                if (j > 1.0f) {
                    half_road_width = j+0.5;
                }
                break;
            }
        }
        half_road_width = (half_road_width > 10) ? half_road_width : 10;
        road_centers.push_back(road_center);
        half_road_widths.push_back(half_road_width);
        
        // Insert this road into existing road network
        vector<set<int>> projection(road_center.points().size(), set<int>());
        for(int j = 0; j < road_center.points().size(); ++j){
            // For each point on the road, find a set of possible nearby roads
            set<int> &pt_projection = projection[j];
           
            point.setCoordinate(road_center.points()[j].x, road_center.points()[j].y, 0.0f);
            // Search nearby
            vector<int> k_indices;
            vector<float> k_dist_sqrts;
            if (map_data_->size() != 0) {
                map_tree_->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dist_sqrts);
            }
            // Prone by direction
            vector<int> true_nearby_road;
            int this_pt_angle = road_center.points()[j].head;
            for (size_t k = 0; k < k_indices.size(); ++k) {
                // Check angle difference
                int angle = map_data_->at(k_indices[k]).head;
                int angle_diff = abs(this_pt_angle - angle);
                float dist_diff = sqrt(k_dist_sqrts[k]);
                if (angle_diff > 180){
                    angle_diff -= 180;
                }
                if (angle_diff > 45) {
                    continue;
                }
                if (dist_diff < map_road_width[k_indices[k]] + half_road_width){
                    true_nearby_road.push_back(k_indices[k]);
                }
            }
            
            if (true_nearby_road.size() == 0) {
                continue;
            }
            
            for(int k = 0; k < true_nearby_road.size(); ++k){
                pt_projection.insert(true_nearby_road[k]);
            }
        }
        
        // Decide what to do according to the projection
        // For each projected road, let's assign the points that on this road projected on to it
        map<int, vector<int>> r_proj; // first is road id, second is a vector containing the points on this road gets projected onto that road
        for (int j = 0; j < road_center.points().size(); ++j) {
            set<int> &pt_projection = projection[j];
            for(set<int>::iterator it = pt_projection.begin(); it != pt_projection.end(); ++it){
                int road_id = map_data_->at(*it).id_trajectory;
                // Check if this is already in r_proj
                map<int, vector<int>>::iterator f_it = r_proj.find(road_id);
                if (f_it != r_proj.end()) {
                    f_it->second.push_back(j);
                }
                else{
                    // Insert
                    r_proj.insert(pair<int, vector<int>>(road_id, vector<int>(1, j)));
                }
            }
        }
       
        int MIN_PT_COUNT = 5;
        int ESCAPE_THRESHOLD = 2; // only allow this number of points to escape
        if (r_proj.size() == 0) {
            // Add new road into map_data_
            vector<int> new_road;
            map_road_width.push_back(half_road_width);
            int n_center_points = road_center.points().size();
            for (int j = 0; j < road_center.points().size(); ++j) {
                if (n_center_points < 2){
                    break;
                }
                SegPoint &pt = road_center.points()[j];
                map_point.setCoordinate(pt.x, pt.y, 0.0f);
                map_point.id_trajectory = curr_n_road;
                map_point.head = pt.head;
                new_road.push_back(map_data_->size());
                map_data_->push_back(map_point);
                if (j > 0) {
                    int prev_pt_id = map_data_->size() - 2;
                    int curr_pt_id = map_data_->size() - 1;
                    // Add edge
                    edge_lists.push_back(pair<int, int>(prev_pt_id, curr_pt_id));
                }
            }
            map_roads.push_back(new_road);
            curr_n_road += 1;
            map_tree_->setInputCloud(map_data_);
        }
        else{
            // Remove unqualified projections
            map<int, vector<pair<int, int>>> road_covers; // first: road id, second: a list of covers: (start_idx, end_idx)
            for (map<int, vector<int>>::iterator it = r_proj.begin(); it != r_proj.end(); ++it) {
                if (it->second.size() < MIN_PT_COUNT) {
                    continue;
                }
                // Check if consecutive points count is greater than MIN_PT_COUNT
                vector<int> &pt_idxs = it->second;
                int start_idx = pt_idxs[0];
                int curr_cumm = 0;
                bool has_enough = false;
                for (int k = 1; k < pt_idxs.size(); ++k) {
                    if (pt_idxs[k] - pt_idxs[k-1] < ESCAPE_THRESHOLD + 1) {
                        curr_cumm += 1;
                    }
                    else{
                        if(curr_cumm > MIN_PT_COUNT){
                            has_enough = true;
                            // Record
                            map<int, vector<pair<int, int>>>::iterator f_it = road_covers.find(it->first);
                            if (f_it != road_covers.end()) {
                                // Append
                                vector<pair<int, int>> &list = f_it->second;
                                list.push_back(pair<int, int>(start_idx, pt_idxs[k-1]));
                            }
                            else{
                                // Add new
                                road_covers.insert(pair<int, vector<pair<int, int>>>(it->first, vector<pair<int, int>>(1, pair<int, int>(start_idx, pt_idxs[k-1]))));
                            }
                        }
                        
                        // break
                        start_idx = pt_idxs[k];
                        curr_cumm = 0;
                    }
                }
                if(curr_cumm > MIN_PT_COUNT){
                    has_enough = true;
                    // Record
                    map<int, vector<pair<int, int>>>::iterator f_it = road_covers.find(it->first);
                    if (f_it != road_covers.end()) {
                        // Append
                        vector<pair<int, int>> &list = f_it->second;
                        list.push_back(pair<int, int>(start_idx, pt_idxs.back()));
                    }
                    else{
                        // Add new
                        road_covers.insert(pair<int, vector<pair<int, int>>>(it->first, vector<pair<int, int>>(1, pair<int, int>(start_idx, pt_idxs.back()))));
                    }
                }
            }
            
            // Merge, and generate new road if necessary
            vector<vector<pair<int, int>>> covered_road_idxs(road_center.points().size()); // for each point, it is a list recording (road_id, covered_size)
            for (map<int, vector<pair<int, int>>>::iterator f_it = road_covers.begin(); f_it != road_covers.end(); ++f_it) {
                vector<pair<int, int>> &tmp = f_it->second;
                for (int k = 0; k < tmp.size(); ++k) {
                    covered_road_idxs[tmp[k].first].push_back(pair<int, int>(f_it->first, tmp[k].second - tmp[k].first + 1));
                }
            }
            // Doing greedy
            int generate_start_idx = 0;
            int generate_end_idx = 0;
            int prev_road_id = -1;
            int k = 0;
            while (k < road_center.points().size()) {
                if (covered_road_idxs[k].size() == 0) {
                    generate_end_idx = k;
                }
                else{
                    int first_pt_idx = -1;
                    int last_pt_idx = -1;
                    if (generate_end_idx - generate_start_idx > MIN_PT_COUNT) {
                        // Generate new road, then merge
                        // Add new road into map_data_
                        vector<int> new_road;
                        map_road_width.push_back(half_road_width);
                        for (int j = generate_start_idx; j < generate_end_idx+1; ++j) {
                            SegPoint &pt = road_center.points()[j];
                            map_point.setCoordinate(pt.x, pt.y, 0.0f);
                            map_point.id_trajectory = curr_n_road;
                            map_point.head = pt.head;
                           
                            if (j == generate_start_idx) {
                                first_pt_idx = map_data_->size();
                            }
                          
                            new_road.push_back(map_data_->size());
                            map_data_->push_back(map_point);
                            if (j > generate_start_idx) {
                                int prev_pt_id = map_data_->size() - 2;
                                int curr_pt_id = map_data_->size() - 1;
                                // Add edge
                                edge_lists.push_back(pair<int, int>(prev_pt_id, curr_pt_id));
                            }
                        }
                        last_pt_idx = map_data_->size() - 1;
                        map_tree_->setInputCloud(map_data_);
                        
                        // If there is previous road, add linking edge
                        if (prev_road_id != -1) {
                            if (generate_start_idx > 0){
                            point.setCoordinate(road_center.points()[generate_start_idx-1].x, road_center.points()[generate_start_idx-1].y, 0.0f);
                            // Search nearby
                            vector<int> k_indices;
                            vector<float> k_dist_sqrts;
                            map_tree_->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dist_sqrts);
                            for (int s = 0; s < k_indices.size(); ++s) {
                                if (map_data_->at(k_indices[s]).id_trajectory == prev_road_id) {
                                    if (first_pt_idx != -1){
                                        float delta_x = map_data_->at(k_indices[s]).x - map_data_->at(first_pt_idx).x;
                                        float delta_y = map_data_->at(k_indices[s]).y - map_data_->at(first_pt_idx).y;
                                        float l = sqrt(delta_x*delta_x + delta_y*delta_y);
                                        if (l < 10) {
                                            edge_lists.push_back(pair<int, int>(k_indices[s], first_pt_idx));
                                        }
                                    }
                                    break;
                                }
                            }
                            }
                        }
                        
                        prev_road_id = curr_n_road;
                        map_roads.push_back(new_road);
                        curr_n_road += 1;
                    }
                    
                    generate_start_idx = generate_end_idx;
                    
                    // Find the largest to cover this portion
                    int max_id = -1;
                    int max_size = 0;
                    vector<pair<int, int>> &tmp = covered_road_idxs[k];
                    for (int s = 0; s < tmp.size(); ++s) {
                        if (tmp[s].second > max_size) {
                            max_size = tmp[s].second;
                            max_id = tmp[s].first;
                        }
                    }
                    
                    // Add linking edge
                    point.setCoordinate(road_center.points()[k].x, road_center.points()[k].y, 0.0f);
                    // Search nearby
                    vector<int> k_indices;
                    vector<float> k_dist_sqrts;
                    map_tree_->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dist_sqrts);
                    int to_pt_idx = -1;
                    for (int s = 0; s < k_indices.size(); ++s) {
                        if (map_data_->at(k_indices[s]).id_trajectory == max_id) {
                            to_pt_idx = k_indices[s];
                            break;
                        }
                    }
                    
                    if (last_pt_idx == -1) {
                        if( k > 0){
                            // We have to locate the points
                            // Search nearby
                            point.setCoordinate(road_center.points()[k-1].x, road_center.points()[k-1].y, 0.0f);
                            vector<int> k_indices;
                            vector<float> k_dist_sqrts;
                            map_tree_->radiusSearch(point, SEARCH_RADIUS, k_indices, k_dist_sqrts);
                            int from_pt_idx = -1;
                            for (int s = 0; s < k_indices.size(); ++s) {
                                if (map_data_->at(k_indices[s]).id_trajectory == prev_road_id) {
                                    from_pt_idx = k_indices[s];
                                    break;
                                }
                            }
                            if (from_pt_idx != -1 && to_pt_idx != -1){
                            float delta_x = map_data_->at(from_pt_idx).x - map_data_->at(to_pt_idx).x;
                            float delta_y = map_data_->at(from_pt_idx).y - map_data_->at(to_pt_idx).y;
                            float l = sqrt(delta_x*delta_x + delta_y*delta_y);
                            if (l < 10) {
                                edge_lists.push_back(pair<int, int>(from_pt_idx, to_pt_idx));
                            }
                            }
                        }
                    }
                    else{
                        if (last_pt_idx != -1 && to_pt_idx != -1){
                        float delta_x = map_data_->at(last_pt_idx).x - map_data_->at(to_pt_idx).x;
                        float delta_y = map_data_->at(last_pt_idx).y - map_data_->at(to_pt_idx).y;
                        float l = sqrt(delta_x*delta_x + delta_y*delta_y);
                        if (l < 10)
                            edge_lists.push_back(pair<int, int>(last_pt_idx, to_pt_idx));
                        }
                    }
                    
                    prev_road_id = max_id;
                    // Jump forward
                    k = k + max_size -1;
                }
                ++k;
            }
        }
        map_tree_->setInputCloud(map_data_);
    }
    
    ofstream test;
    test.open("generated_roads.txt");
    test<<road_centers.size()<<endl;
    for (size_t i = 0; i < road_centers.size(); ++i) {
        Segment &seg = road_centers[i];
        test << seg.points().size() << endl;
        test << half_road_widths[i] << endl;
        for (size_t j = 0; j < seg.points().size(); ++j) {
            SegPoint &pt = seg.points()[j];
            test << pt.x << ", " << pt.y << endl;
        }
    }
    
    test.close();
    
    ofstream output;
    output.open("generated_map.txt");
    output << map_data_->size() << endl;
    for(size_t i = 0; i < map_data_->size(); ++i){
        output << map_data_->at(i).x << ", " << map_data_->at(i).y << endl;
    }
    output << map_roads.size() << endl;
    for (size_t i = 0; i < map_roads.size(); ++i) {
        output << map_road_width[i];
        vector<int> &road = map_roads[i];
        for (size_t j = 0; j < road.size(); ++j) {
            output << ", " << road[j];
        }
        output << endl;
    }
    output << edge_lists.size() << endl;
    for (size_t i = 0; i < edge_lists.size(); ++i) {
        output << edge_lists[i].first << ", " << edge_lists[i].second << endl;
    }
    output.close();
}

float Trajectories::pointToSegmentDistance(SegPoint &query_pt, SegPoint &p1, SegPoint &p2){
    Vector2d v(p2.x - p1.x,
               p2.y - p1.y);
    float v_length = v.norm();
    v.normalize();
    Vector2d v1(query_pt.x - p1.x,
                query_pt.y - p1.y);
    float projection = v1.dot(v);
    float pt_to_segment_dist = 0.0f;
    if (projection < 0) {
        float delta_x = query_pt.x - p1.x;
        float delta_y = query_pt.y - p1.y;
        pt_to_segment_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
    }
    else if(projection > v_length){
        float delta_x = query_pt.x - p2.x;
        float delta_y = query_pt.y - p2.y;
        pt_to_segment_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
    }
    else{
        float v1_length = v1.norm();
        pt_to_segment_dist = sqrt(v1_length*v1_length - projection*projection);
    }
    return pt_to_segment_dist;
}

void Trajectories::recomputeSegmentsForSamples(){
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    for (set<int>::iterator it = picked_sample_idxs_.begin(); it != picked_sample_idxs_.end(); ++it) {
        selectSegmentsNearSample(*it);
    }
}

void Trajectories::updateSelectedSegments(void){
    segment_vertices_.clear();
    segment_colors_.clear();
    segment_idxs_.clear();
    segment_speed_.clear();
    segment_speed_colors_.clear();
    segment_speed_indices_.clear();
    
    // Draw general picked segment in segments_to_draw_
    for (size_t idx = 0; idx < segments_to_draw_.size(); ++idx) {
        Segment &this_seg = segments_[segments_to_draw_[idx]];
        vector<int> idxs;
        Color seg_color = ColorMap::getInstance().getDiscreteColor(idx);
        for (int j = 0; j < this_seg.points().size(); ++j) {
            int orig_idx = this_seg.points()[j].orig_idx;
            idxs.push_back(segment_vertices_.size());
            Vertex &orig_vertex = normalized_vertices_[orig_idx];
            segment_vertices_.push_back(Vertex(orig_vertex.x, orig_vertex.y, Z_SEGMENTS));
            Vertex &orig_speed_v1 = vertex_speed_[2*orig_idx];
            Vertex &orig_speed_v2 = vertex_speed_[2*orig_idx+1];
            segment_speed_colors_.push_back(Color(1-seg_color.r, 1.0-seg_color.g, 1.0-seg_color.b, 1.0));
            segment_speed_colors_.push_back(Color(1-seg_color.r, 1.0-seg_color.g, 1.0-seg_color.b, 1.0));
            segment_speed_indices_.push_back(segment_speed_indices_.size());
            segment_speed_.push_back(Vertex(orig_speed_v1.x, orig_speed_v1.y, Z_SEGMENTS+0.1));
            segment_speed_indices_.push_back(segment_speed_indices_.size());
            segment_speed_.push_back(Vertex(orig_speed_v2.x, orig_speed_v2.y, Z_SEGMENTS+0.1));
            
            float alpha = 1.0 - 0.7 * static_cast<float>(j) / this_seg.points().size();
            Color pt_color(alpha*seg_color.r, alpha*seg_color.g, alpha*seg_color.b, 1.0);
            
            segment_colors_.push_back(pt_color);
            
        }
        segment_idxs_.push_back(idxs);
    }
    
    // Draw segments due to picked samples
    for (size_t idx = 0; idx < segments_to_draw_for_samples_.size(); ++idx) {
        Segment &this_seg = segments_[segments_to_draw_for_samples_[idx]];
        Color &this_color = segments_to_draw_for_samples_colors_[idx];
        vector<int> idxs;
        for (int j = 0; j < this_seg.points().size(); ++j) {
            int orig_idx = this_seg.points()[j].orig_idx;
            idxs.push_back(segment_vertices_.size());
            Vertex &orig_vertex = normalized_vertices_[orig_idx];
            segment_vertices_.push_back(Vertex(orig_vertex.x, orig_vertex.y, Z_SEGMENTS));
            float alpha = 1.0 - 0.7 * static_cast<float>(j) / this_seg.points().size();
            Color pt_color(alpha*this_color.r, alpha*this_color.g, alpha*this_color.b, 1.0);
            segment_colors_.push_back(pt_color);
            
            Vertex &orig_speed_v1 = vertex_speed_[2*orig_idx];
            Vertex &orig_speed_v2 = vertex_speed_[2*orig_idx+1];
            segment_speed_colors_.push_back(Color(1-this_color.r, 1.0-this_color.g, 1.0-this_color.b, 1.0));
            segment_speed_colors_.push_back(Color(1-this_color.r, 1.0-this_color.g, 1.0-this_color.b, 1.0));
            segment_speed_indices_.push_back(segment_speed_indices_.size());
            segment_speed_.push_back(Vertex(orig_speed_v1.x, orig_speed_v1.y, Z_SEGMENTS+0.1));
            segment_speed_indices_.push_back(segment_speed_indices_.size());
            segment_speed_.push_back(Vertex(orig_speed_v2.x, orig_speed_v2.y, Z_SEGMENTS+0.1));
        }
        segment_idxs_.push_back(idxs);
    }
}

void Trajectories::selectSegmentsNearSample(int sample_id){
    if (segments_.size() == 0)
        return;
    
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    printf("Search range is %2f\n", search_range_);
    
    Color seg_color = ColorMap::getInstance().getDiscreteColor(sample_id);
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        segments_to_draw_for_samples_.push_back(k_indices[i]);
        segments_to_draw_for_samples_colors_.push_back(seg_color);
    }
    
    updateSelectedSegments();
}

void Trajectories::drawSegmentAt(int seg_idx){
    if (segments_.size() == 0){
        return;
    }
    //segments_to_draw_.clear();
    segments_to_draw_.push_back(seg_idx);
    updateSelectedSegments();
}

void Trajectories::clearDrawnSegments(){
    segments_to_draw_.clear();
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    updateSelectedSegments();
}

Segment &Trajectories::segmentAt(int seg_idx){
    return segments_[seg_idx];
}

void Trajectories::interpolateSegmentWithGraph(Graph *g){
    if (g == NULL){
        printf("Empty graph!\n");
        return;
    }
    if (segments_.empty()){
        printf("No segment to interpolate.\n");
        return;
    }
    
    graph_interpolated_segments_.clear();
    graph_interpolated_segments_.resize(segments_.size());
    
    for (size_t seg_id = 0; seg_id < segments_.size(); ++seg_id){
        Segment &seg = segmentAt(seg_id);
        vector<int> seg_pt_idxs;
        for(size_t i = 0; i < seg.points().size(); ++i){
            SegPoint &pt = seg.points()[i];
            seg_pt_idxs.push_back(pt.orig_idx);
        }
        
        vector<int> &result_path = graph_interpolated_segments_[seg_id];
        g->shortestPathInterpolation(seg_pt_idxs, result_path);
    }
}

void Trajectories::interpolateSegmentWithGraphAtSample(int sample_id, Graph *g, vector<int> &nearby_segments, vector<vector<int>> &results){
    if (g == NULL){
        printf("Empty graph!\n");
        return;
    }
    if (segments_.empty()){
        printf("No segment to interpolate.\n");
        return;
    }
    
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    
    //Color seg_color = ColorMap::getInstance().getDiscreteColor(picked_sample_idxs_.size());
    nearby_segments.clear();
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        nearby_segments.push_back(k_indices[i]);
    }
    
    results.clear();
    results.resize(nearby_segments.size());
    for (size_t seg_id = 0; seg_id < nearby_segments.size(); ++seg_id){
        Segment &seg = segmentAt(nearby_segments[seg_id]);
        vector<int> seg_pt_idxs;
        for(size_t i = 0; i < seg.points().size(); ++i){
            SegPoint &pt = seg.points()[i];
            seg_pt_idxs.push_back(pt.orig_idx);
        }
        
        vector<int> &result_path = results[seg_id];
        g->shortestPathInterpolation(seg_pt_idxs, result_path);
    }
}

void Trajectories::clusterSegmentsWithGraphAtSample(int sample_id, Graph *g){
    vector<int> nearby_segment_idxs;
    vector<vector<int>> interpolated_segments;
    interpolateSegmentWithGraphAtSample(sample_id, g, nearby_segment_idxs, interpolated_segments);
    
    printf("%lu nearby segments.\n", nearby_segment_idxs.size());
    
    set<int> remaining_segments;
    for (size_t i = 0;  i < nearby_segment_idxs.size(); ++i) {
        remaining_segments.insert(i);
    }
    
    vector<vector<int>> clusters;
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    while (remaining_segments.size() != 0) {
        // Randomly pick a segment
        int idx = rand() % remaining_segments.size();
        set<int>::iterator it = remaining_segments.begin();
        advance(it, idx);
        int selected_seg_idx = *it;
        
        vector<int> &selected_segment = interpolated_segments[*it];
        
        // Check if this segment can be merged to existing segments; Otherwise it will become a new cluster.
        bool absorbed = false;
        for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            vector<int> &cluster = clusters[cluster_id];
            float max_distance = 0.0f;
            for (size_t s_id = 0; s_id < cluster.size(); ++s_id) {
                key_type this_key(*it, cluster[s_id]);
                key_type reverse_key(cluster[s_id], *it);
                
                // Check distance between selected segment and this cluster segment.
                map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                float dist;
                if (m_it != computed_dist.end()) {
                    dist = m_it->second;
                }
                else{
                    // Compute distance
                    dist = g->DTWDistance(selected_segment, interpolated_segments[cluster[s_id]]);
                    computed_dist[this_key] = dist;
                    computed_dist[reverse_key] = dist;
                }
                if (dist > max_distance) {
                    max_distance = dist;
                }
            }
            
            if (max_distance < 50.0f) {
                cluster.push_back(selected_seg_idx);
                absorbed = true;
            }
        }
        
        if (!absorbed){
            vector<int> new_cluster;
            new_cluster.push_back(selected_seg_idx);
            clusters.push_back(new_cluster);
        }
        
        remaining_segments.erase(it);
    }
    
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    int n_valid_cluster = 0;
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        if (cluster.size() < 5) {
            continue;
        }
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(n_valid_cluster);
        ++n_valid_cluster;
        for (size_t seg_id = 0; seg_id < cluster.size(); ++seg_id) {
            segments_to_draw_for_samples_.push_back(nearby_segment_idxs[cluster[seg_id]]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    printf("%d cluster extracted.\n", n_valid_cluster);
    updateSelectedSegments();
}

int Trajectories::clusterSegmentsUsingDistanceAtSample(int sample_id, double sigma, double threshold, int minClusterSize){
    vector<int> nearby_segment_idxs;
    
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    
    nearby_segment_idxs.clear();
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        nearby_segment_idxs.push_back(k_indices[i]);
    }
    
    printf("%lu nearby segments.\n", nearby_segment_idxs.size());
    
    //clock_t begin = clock();
    int first_idx = -1;
    float max_dist = 0.0f;
    for (size_t i = 0; i < nearby_segment_idxs.size(); ++i){
        Segment &seg = segments_[nearby_segment_idxs[i]];
        if (seg.points().size() < 2) {
            continue;
        }
        float delta_x = seg.points()[0].x - samples_->at(sample_id).x;
        float delta_y = seg.points()[0].y - samples_->at(sample_id).y;
        float dist1 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        int last_pt_idx = seg.points().size() - 1;
        delta_x = seg.points()[last_pt_idx].x - samples_->at(sample_id).x;
        delta_y = seg.points()[last_pt_idx].y - samples_->at(sample_id).y;
        float dist2 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        float dist = (dist1 > dist2) ? dist1 : dist2;
        if (max_dist < dist) {
            max_dist = dist;
            first_idx = i;
        }
    }
    
    
    // Randomly pick the starting segment
    set<int> remaining_segments;
    for (size_t i = 0;  i < nearby_segment_idxs.size(); ++i) {
        if (i == first_idx) {
            continue;
        }
        remaining_segments.insert(nearby_segment_idxs[i]);
    }
    
    // Start furthest point clustering
    vector<vector<int>> clusters;
    vector<int> first_cluster;
    first_cluster.push_back(nearby_segment_idxs[first_idx]);
    clusters.push_back(first_cluster);
    
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    while (remaining_segments.size() != 0) {
        // Look over remaining segments, find its distance to each existing clusters.
        float max_distance = -1e6;
        int max_segment_idx = -1;
        
        vector<int> segments_to_merge;
        vector<int> merging_to;
        for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
            int seg_id = *it;
            
            // Check the minimum distance of this segment to each of the existing clusters.
            float min_distance = 1e6;
            int min_cluster_id = -1;
            for (size_t i_cluster = 0; i_cluster < clusters.size(); ++i_cluster) {
                vector<int> &current_cluster = clusters[i_cluster];
                // Look through each segment in the cluster
                float tmp_min_distance = 1e6;
                for (size_t j_seg = 0; j_seg < current_cluster.size(); ++j_seg) {
                    int other_seg_id = current_cluster[j_seg];
                    // Reuse distance if it has been computed before
                    key_type this_key(seg_id, other_seg_id);
                    key_type reverse_key(other_seg_id, seg_id);
                    map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                    float tmp_dist = 0.0f;
                    if (m_it != computed_dist.end()) {
                        tmp_dist = m_it->second;
                    }
                    else{
                        // Compute tmp_dist.
                        tmp_dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                        // Insert the computed dist for reuse.
                        //printf("dist = %.2f\n", tmp_dist);
                        computed_dist[this_key] = tmp_dist;
                        computed_dist[reverse_key] = tmp_dist;
                    }
                    if(tmp_dist < tmp_min_distance){
                        tmp_min_distance = tmp_dist;
                    }
                }
                
                if (tmp_min_distance < min_distance) {
                    min_distance = tmp_min_distance;
                    min_cluster_id = i_cluster;
                }
            }
            
            // If min_distance is small, we will merge this segment into that corresponding cluster
            if (min_distance < threshold) {
                // Add this segment id to merging list
                segments_to_merge.push_back(seg_id);
                merging_to.push_back(min_cluster_id);
            }
            
            if (max_distance < min_distance) {
                max_distance = min_distance;
                max_segment_idx = seg_id;
            }
        }
        
        // Remove merging segments
        for (size_t i = 0; i < segments_to_merge.size(); ++i) {
            clusters[merging_to[i]].push_back(segments_to_merge[i]);
            remaining_segments.erase(segments_to_merge[i]);
        }
        
        if (max_distance < threshold) {
            break;
        }
        
        // reate new cluster
        vector<int> new_cluster;
        new_cluster.push_back(max_segment_idx);
        clusters.push_back(new_cluster);
        remaining_segments.erase(max_segment_idx);
    }
    
    //clock_t end = clock();
    //double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    //printf("Clustering %lu segments took %.1f sec.\n", nearby_segment_idxs.size(), elapsed_time);
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    int n_valid_cluster = 0;
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        if (cluster.size() < minClusterSize) {
            continue;
        }
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(n_valid_cluster);
        ++n_valid_cluster;
        for (size_t i = 0; i < cluster.size(); ++i) {
            segments_to_draw_for_samples_.push_back(cluster[i]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    updateSelectedSegments();
    return n_valid_cluster;
}

void Trajectories::clusterSegmentsAtAllSamples(double sigma, int minClusterSize){
    if (samples_->size() == 0)
        return;
    
    cluster_centers_ = samples_;
    cluster_center_search_tree_ = sample_tree_;
    return;
    
    int total_n_clusters = 0;
    
    PclPoint point;
    vector<vector<vector<int>>> all_sample_clusters(samples_->size());
    for (size_t sample_id = 0; sample_id < samples_->size(); ++sample_id){
        vector<vector<int>> &clusters = all_sample_clusters[sample_id];
        int n_clusters = fpsSegmentSampling(sample_id, sigma, minClusterSize, clusters);
        if (n_clusters == 0) {
            total_n_clusters += 1;
        }
        else{
            total_n_clusters += n_clusters;
        }
    }
    
    cluster_centers_->clear();
    cluster_popularity_.clear();
    cluster_descriptors_ = new cv::Mat(total_n_clusters, 16, cv::DataType<float>::type, 0.0f);
    int current_cluster_id = 0;
    for (size_t sample_id = 0; sample_id < samples_->size(); ++sample_id){
        vector<vector<int>> &clusters = all_sample_clusters[sample_id];
        // Deal with empty clusterd sample
        if (clusters.size() == 0) {
            point.setCoordinate(samples_->at(sample_id).x, samples_->at(sample_id).y, 0.0f);
            cluster_centers_->push_back(point);
            cluster_popularity_.push_back(0);
            current_cluster_id += 1;
            continue;
        }
        // Compute descriptors for each clusters and store them.
        for (size_t i_cluster = 0; i_cluster < clusters.size(); ++i_cluster){
            vector<int> &cluster = clusters[i_cluster];
            point.setCoordinate(samples_->at(sample_id).x, samples_->at(sample_id).y, 0.0f);
            cluster_centers_->push_back(point);
            cluster_popularity_.push_back(cluster.size());
            float cummulated_count = 0.0f;
            for (size_t j = 0; j < cluster.size(); ++j){
                int seg_id = cluster[j];
                Segment &seg = segments_[seg_id];
                for (size_t k = 0; k < seg.points().size(); ++k) {
                    float delta_x = seg.points()[k].x - samples_->at(sample_id).x;
                    float delta_y = seg.points()[k].y - samples_->at(sample_id).y;
                    float r = sqrt(delta_x*delta_x + delta_y*delta_y);
                    int r_bin_id = 0;
                    if (r > 50.0f) {
                        r_bin_id = 1;
                    }
                    //                    else if (r > 100.0f && r <= 250.0f){
                    //                        r_bin_id = 2;
                    //                    }
                    //                    else if (r > 250.0f){
                    //                        r_bin_id = 3;
                    //                    }
                    
                    float cos_value = delta_x / r;
                    float theta = acos(cos_value) * 180.0f / PI;
                    if (delta_y < 0) {
                        theta += 180;
                    }
                    int theta_bin_id;
                    if (theta > 45.0f && theta <= 135.0f) {
                        theta_bin_id = 1;
                    }
                    else if (theta > 135.0f && theta <= 225.0f){
                        theta_bin_id = 2;
                    }
                    else if (theta > 225.0f && theta <= 315.0f){
                        theta_bin_id = 3;
                    }
                    else{
                        theta_bin_id = 0;
                    }
                    
                    int bin_idx = r_bin_id * theta_bin_id;
                    
                    if (seg.points()[k].t >= 0) {
                        bin_idx += 8;
                    }
                    cluster_descriptors_->at<float>(current_cluster_id, bin_idx) += 1.0;
                    cummulated_count += 1.0f;
                }
            }
            
            // Normalize this distribution
            if (cummulated_count > 0.0f) {
                for (int s=0; s < 16; ++s) {
                    cluster_descriptors_->at<float>(current_cluster_id, s) /= cummulated_count;
                }
            }
            current_cluster_id += 1;
        }
    }
    
    //cout << *cluster_descriptors_ << endl;
    
    cluster_center_search_tree_->setInputCloud(cluster_centers_);
    
    //cv::flann::Index flann_index(obj_32f, cv::flann::KDTreeIndexParams());
    
    //cv::flann::Index flann_index(, cv::flann::Kd);
    
    //    segments_to_draw_for_samples_.clear();
    //    segments_to_draw_for_samples_colors_.clear();
    //    for (size_t sample_id = 0; sample_id < samples_->size(); ++sample_id) {
    //        vector<int> &cluster_center = sample_segment_clusters_[sample_id];
    //        Color cluster_color = ColorMap::getInstance().getDiscreteColor(sample_id);
    //        for (size_t i = 0; i < cluster_center.size(); ++i) {
    //            segments_to_draw_for_samples_.push_back(cluster_center[i]);
    //            segments_to_draw_for_samples_colors_.push_back(cluster_color);
    //        }
    //    }
    //
    //    updateSelectedSegments();
}

int Trajectories::fpsSegmentSampling(int sample_id, double sigma, int minClusterSize, vector<vector<int>> &clusters){
    vector<int> nearby_segment_idxs;
    // Search GPS points around the sample point
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
    
    nearby_segment_idxs.clear();
    for (size_t i = 0; i < k_indices.size(); ++i) {
        if (!segments_[k_indices[i]].quality()) {
            continue;
        }
        nearby_segment_idxs.push_back(k_indices[i]);
    }
    
    if (nearby_segment_idxs.size() == 0)
        return 0;
    
    int first_idx = -1;
    float max_dist = 0.0f;
    for (size_t i = 0; i < nearby_segment_idxs.size(); ++i){
        Segment &seg = segments_[nearby_segment_idxs[i]];
        if (seg.points().size() < 2) {
            continue;
        }
        float delta_x = seg.points()[0].x - samples_->at(sample_id).x;
        float delta_y = seg.points()[0].y - samples_->at(sample_id).y;
        float dist1 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        int last_pt_idx = seg.points().size() - 1;
        delta_x = seg.points()[last_pt_idx].x - samples_->at(sample_id).x;
        delta_y = seg.points()[last_pt_idx].y - samples_->at(sample_id).y;
        float dist2 = sqrt(delta_x*delta_x + delta_y*delta_y);
        
        float dist = (dist1 > dist2) ? dist1 : dist2;
        if (max_dist < dist) {
            max_dist = dist;
            first_idx = i;
        }
    }
    
    //clock_t begin = clock();
    set<int> remaining_segments;
    for (size_t i = 0;  i < nearby_segment_idxs.size(); ++i) {
        if (i == first_idx) {
            continue;
        }
        remaining_segments.insert(nearby_segment_idxs[i]);
    }
    
    // Start furthest point clustering
    vector<vector<int>> tmp_clusters;
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    int MAX_CLUSTERS = 20;
    vector<float> max_distances(MAX_CLUSTERS, 0.0f);
    vector<int> selected_centers;
    selected_centers.push_back(nearby_segment_idxs[first_idx]);
    for (int i=0; i < MAX_CLUSTERS-1; ++i){
        float max_distance = -1;
        int max_segment_idx = -1;
        for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
            int seg_id = *it;
            float min_distance = POSITIVE_INFINITY;
            // Compute distance between current segment and the already selected cluster center segments.
            for (size_t ith_selected = 0; ith_selected < selected_centers.size(); ++ith_selected) {
                int other_seg_id = selected_centers[ith_selected];
                // Reuse distance if it has been computed before
                key_type this_key(seg_id, other_seg_id);
                key_type reverse_key(other_seg_id, seg_id);
                map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                float dist = 0.0f;
                if (m_it != computed_dist.end()) {
                    dist = m_it->second;
                }
                else{
                    // Compute dist.
                    dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                    // Insert the computed dist for reuse.
                    computed_dist[this_key] = dist;
                    computed_dist[reverse_key] = dist;
                }
                
                if (dist < min_distance) {
                    min_distance = dist;
                }
            }
            // Record the current max distance segment and its id
            if (max_distance < min_distance) {
                max_distance = min_distance;
                max_segment_idx = seg_id;
            }
        }
        
        // Insert max_segment_idx into selected_centers
        selected_centers.push_back(max_segment_idx);
        max_distances[i] = max_distance;
    }
    
    // Determine the number of clusters we need to use
    int n_clusters = 0;
    float threshold = POSITIVE_INFINITY;
    bool threshold_selected = false;
    for (int i = MAX_CLUSTERS-1; i >= 0; --i) {
        if (max_distances[i] > 1e-3 && !threshold_selected) {
            threshold = 1.5*max_distances[i];
            threshold_selected = true;
            //printf("Min value is: %.2f\n", threshold);
        }
        if (max_distances[i] > threshold) {
            n_clusters = i + 2;
            break;
        }
    }
    
    // Initialiaze tmp_clusters
    for (int i = 0; i < n_clusters; ++i) {
        vector<int> new_cluster;
        new_cluster.push_back(selected_centers[i]);
        tmp_clusters.push_back(new_cluster);
        remaining_segments.erase(selected_centers[i]);
    }
    
    // Put remaining segments to clusters
    for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
        int seg_id = *it;
        float min_distance = POSITIVE_INFINITY;
        int min_cluster_id = -1;
        // Compute distance between current segment and the already selected cluster center segments.
        for (size_t ith_cluster = 0; ith_cluster < tmp_clusters.size(); ++ith_cluster) {
            int other_seg_id = tmp_clusters[ith_cluster][0];
            // Reuse distance if it has been computed before
            key_type this_key(seg_id, other_seg_id);
            key_type reverse_key(other_seg_id, seg_id);
            map<key_type, float>::iterator m_it = computed_dist.find(this_key);
            float dist = 0.0f;
            if (m_it != computed_dist.end()) {
                dist = m_it->second;
            }
            else{
                // Compute dist.
                dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                // Insert the computed dist for reuse.
                computed_dist[this_key] = dist;
                computed_dist[reverse_key] = dist;
            }
            
            if (dist < min_distance) {
                min_distance = dist;
                min_cluster_id = ith_cluster;
            }
        }
        if (min_cluster_id != -1)
            tmp_clusters[min_cluster_id].push_back(seg_id);
    }
    
    // Record GOOD clusters into clusters.
    clusters.clear();
    clusters_.clear();
    for (int i = 0; i < tmp_clusters.size(); ++i) {
        vector<int> &cluster = tmp_clusters[i];
        if (cluster.size() >= minClusterSize) {
            clusters.push_back(cluster);
            clusters_.push_back(cluster);
        }
    }
    
    //clock_t end = clock();
    //double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    //printf("Clustering %lu segments took %.1f sec.\n", nearby_segment_idxs.size(), elapsed_time);
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(cluster_id);
        for (size_t i = 0; i < cluster.size(); ++i) {
            segments_to_draw_for_samples_.push_back(cluster[i]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    updateSelectedSegments();
    
    // Write clustering results to .txt file
    ofstream test;
    test.open("clustering_results.txt");
    test<<clusters.size()<<endl;
    for (size_t i = 0; i < clusters.size(); ++i) {
        vector<int> &cluster = clusters[i];
        test << cluster.size() << endl;
        for (size_t j = 0; j < cluster.size(); ++j) {
            Segment s = segments_[cluster[j]];
            test << s.points().size() <<endl;
            for (size_t k = 0; k < s.points().size(); ++k) {
                test << s.points()[k].x << " " << s.points()[k].y << " " << s.points()[k].t << endl;
            }
        }
    }
    
    test.close();
    
    return clusters.size();
}

int Trajectories::fpsMultiSiteSegmentClustering(vector<int> selected_samples, double sigma, int minClusterSize, vector<vector<int>> &clusters){
    set<int> nearby_segment_idxs;
    // Search GPS points around the each sample point
    for(size_t i = 0; i < selected_samples.size(); ++i){
        int sample_id = selected_samples[i];
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->radiusSearch(samples_->at(sample_id), search_range_, k_indices, k_distances);
        
        for (size_t i = 0; i < k_indices.size(); ++i) {
            if (!segments_[k_indices[i]].quality()) {
                continue;
            }
            nearby_segment_idxs.insert(k_indices[i]);
        }
    }
    
    if (nearby_segment_idxs.size() == 0)
        return 0;
    
    int first_idx = rand() % nearby_segment_idxs.size();
    int first_segment_id = -1;
    
    clock_t begin = clock();
    set<int> remaining_segments;
    int i = 0;
    for(set<int>::iterator it = nearby_segment_idxs.begin(); it != nearby_segment_idxs.end(); ++it){
        if (i == first_idx) {
            first_segment_id = *it;
            continue;
        }
        remaining_segments.insert(*it);
        ++i;
    }
    
    // Start furthest point clustering
    vector<vector<int>> tmp_clusters;
    typedef pair<int, int> key_type;
    map<key_type, float> computed_dist;
    
    int MAX_CLUSTERS = 20;
    vector<float> max_distances(MAX_CLUSTERS, 0.0f);
    vector<int> selected_centers;
    selected_centers.push_back(first_segment_id);
    for (int i=0; i < MAX_CLUSTERS-1; ++i){
        float max_distance = -1;
        int max_segment_idx = -1;
        for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
            int seg_id = *it;
            float min_distance = POSITIVE_INFINITY;
            // Compute distance between current segment and the already selected cluster center segments.
            for (size_t ith_selected = 0; ith_selected < selected_centers.size(); ++ith_selected) {
                int other_seg_id = selected_centers[ith_selected];
                // Reuse distance if it has been computed before
                key_type this_key(seg_id, other_seg_id);
                key_type reverse_key(other_seg_id, seg_id);
                map<key_type, float>::iterator m_it = computed_dist.find(this_key);
                float dist = 0.0f;
                if (m_it != computed_dist.end()) {
                    dist = m_it->second;
                }
                else{
                    // Compute dist.
                    dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                    // Insert the computed dist for reuse.
                    computed_dist[this_key] = dist;
                    computed_dist[reverse_key] = dist;
                }
                
                if (dist < min_distance) {
                    min_distance = dist;
                }
            }
            // Record the current max distance segment and its id
            if (max_distance < min_distance) {
                max_distance = min_distance;
                max_segment_idx = seg_id;
            }
        }
        
        // Insert max_segment_idx into selected_centers
        selected_centers.push_back(max_segment_idx);
        max_distances[i] = max_distance;
    }
    
    // Determine the number of clusters we need to use
    int n_clusters = 0;
    float threshold = POSITIVE_INFINITY;
    bool threshold_selected = false;
    for (int i = MAX_CLUSTERS-1; i >= 0; --i) {
        if (max_distances[i] > 1e-3 && !threshold_selected) {
            threshold = 1.5*max_distances[i];
            threshold_selected = true;
            //printf("Min value is: %.2f\n", threshold);
        }
        if (max_distances[i] > threshold) {
            n_clusters = i + 2;
            break;
        }
    }
    
    // Initialiaze tmp_clusters
    for (int i = 0; i < n_clusters; ++i) {
        vector<int> new_cluster;
        new_cluster.push_back(selected_centers[i]);
        tmp_clusters.push_back(new_cluster);
        remaining_segments.erase(selected_centers[i]);
    }
    
    // Put remaining segments to clusters
    for (set<int>::iterator it=remaining_segments.begin(); it != remaining_segments.end(); ++it) {
        int seg_id = *it;
        float min_distance = POSITIVE_INFINITY;
        int min_cluster_id = -1;
        // Compute distance between current segment and the already selected cluster center segments.
        for (size_t ith_cluster = 0; ith_cluster < tmp_clusters.size(); ++ith_cluster) {
            int other_seg_id = tmp_clusters[ith_cluster][0];
            // Reuse distance if it has been computed before
            key_type this_key(seg_id, other_seg_id);
            key_type reverse_key(other_seg_id, seg_id);
            map<key_type, float>::iterator m_it = computed_dist.find(this_key);
            float dist = 0.0f;
            if (m_it != computed_dist.end()) {
                dist = m_it->second;
            }
            else{
                // Compute dist.
                dist = computeSegmentDistance(seg_id, other_seg_id, sigma);
                // Insert the computed dist for reuse.
                computed_dist[this_key] = dist;
                computed_dist[reverse_key] = dist;
            }
            
            if (dist < min_distance) {
                min_distance = dist;
                min_cluster_id = ith_cluster;
            }
        }
        if (min_cluster_id != -1)
            tmp_clusters[min_cluster_id].push_back(seg_id);
    }
    
    // Record GOOD clusters into clusters.
    clusters.clear();
    clusters_.clear();
    for (int i = 0; i < tmp_clusters.size(); ++i) {
        vector<int> &cluster = tmp_clusters[i];
        if (cluster.size() >= minClusterSize) {
            clusters.push_back(cluster);
            clusters_.push_back(cluster);
        }
    }
    
    clock_t end = clock();
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    printf("Clustering %lu segments took %.1f sec.\n", nearby_segment_idxs.size(), elapsed_time);
    
    // Visualization
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        vector<int> &cluster = clusters[cluster_id];
        Color cluster_color = ColorMap::getInstance().getDiscreteColor(cluster_id);
        for (size_t i = 0; i < cluster.size(); ++i) {
            segments_to_draw_for_samples_.push_back(cluster[i]);
            segments_to_draw_for_samples_colors_.push_back(cluster_color);
        }
    }
    updateSelectedSegments();
    
    // Write clustering results to .txt file
    //    ofstream test;
    //    test.open("clustering_results.txt");
    //    test<<clusters.size()<<endl;
    //    for (size_t i = 0; i < clusters.size(); ++i) {
    //        vector<int> &cluster = clusters[i];
    //        test << cluster.size() << endl;
    //        for (size_t j = 0; j < cluster.size(); ++j) {
    //            Segment s = segments_[cluster[j]];
    //            test << s.points().size() <<endl;
    //            for (size_t k = 0; k < s.points().size(); ++k) {
    //                test << s.points()[k].x << " " << s.points()[k].y << " " << s.points()[k].t << endl;
    //            }
    //        }
    //    }
    //    test.close();
    
    return clusters.size();
}

void Trajectories::showClusterAtIdx(int idx){
    if (idx < 0 || idx >= clusters_.size())
        return;
    
    segments_to_draw_for_samples_.clear();
    segments_to_draw_for_samples_colors_.clear();
    vector<int> &cluster = clusters_[idx];
    
    Color cluster_color = ColorMap::getInstance().getDiscreteColor(idx);
    for (size_t i = 0; i < cluster.size(); ++i) {
        segments_to_draw_for_samples_.push_back(cluster[i]);
        segments_to_draw_for_samples_colors_.push_back(cluster_color);
    }
    updateSelectedSegments();
}

void Trajectories::exportSampleSegments(const string &filename){
    ofstream output;
    output.open(filename);
    int selected_sample = *picked_sample_idxs_.begin();
    output << samples_->at(selected_sample).x << " " << samples_->at(selected_sample).y << endl;
    output<<segments_to_draw_for_samples_.size()<<endl;
    for (size_t i = 0; i < segments_to_draw_for_samples_.size(); ++i) {
        Segment s = segments_[segments_to_draw_for_samples_[i]];
        output << s.points().size() <<endl;
        for (size_t k = 0; k < s.points().size(); ++k) {
            int orig_pt_idx = s.points()[k].orig_idx;
            float angle = static_cast<float>(360 - data_->at(orig_pt_idx).head + 90) * PI / 180.0f;
            float head_x = cos(angle);
            float head_y = sin(angle);
            output << s.points()[k].x << " " << s.points()[k].y << " " << s.points()[k].t << " " << head_x << " " << head_y << " " << data_->at(orig_pt_idx).speed << endl;
        }
    }
    output.close();
}

void Trajectories::extractSampleFeatures(const string &filename){
    int n_bins = 12;
    float delta_bin_angle = 360.0f / n_bins;
    
    float search_radius = 5.0f;
    
    ofstream output;
    output.open(filename);
    output << samples_->size() << endl;
    output.precision(6);
    output.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < samples_->size(); ++i) {
        // Search nearby trajectories and compute the turning angle for each
        vector<int> nearby_pt_idxs;
        vector<float> nearby_pt_dist_squares;
        tree_->radiusSearch(samples_->at(i), search_radius, nearby_pt_idxs, nearby_pt_dist_squares);
        
        set<int> computed_traj_idxs;
        vector<float> descriptor(n_bins, 0.0f);
        
        for (size_t j = 0; j < nearby_pt_idxs.size(); ++j) {
            int this_pt_idx = nearby_pt_idxs[j];
            
            Segment &this_seg = segments_[this_pt_idx];
            int entering_pt_idx = this_seg.points()[0].orig_idx;
            int end_pt = this_seg.points().size();
            end_pt -= 1;
            if (end_pt < 0) {
                end_pt = 0;
            }
            int leaving_pt_idx = this_seg.points()[end_pt].orig_idx;
            
            int entering_angle = 450 - data_->at(entering_pt_idx).head;
            if (entering_angle > 360){
                entering_angle -= 360;
            }
            
            int leaving_angle = 450 - data_->at(leaving_pt_idx).head;
            if (leaving_angle > 360){
                leaving_angle -= 360;
            }
            
            float delta_angle = entering_angle - leaving_angle;
            
            if (delta_angle > 180){
                delta_angle = 360 - delta_angle;
            }
            if (delta_angle < -180){
                delta_angle = 360 + delta_angle;
            }
            
            int bin_id = floor((180.0f + delta_angle) / delta_bin_angle);
            descriptor[bin_id] = 1.0f;
        }
        
        // Normalize the descriptor
        float sum = 0.0f;
        for (int j = 0; j < n_bins; ++j) {
            sum += descriptor[j];
        }
        
        if (sum < 1e-3){
            sum = 1.0f;
        }
        
        // Output descriptor to file
        double sample_lat = static_cast<double>(samples_->at(i).lat) / 1e5;
        double sample_lon = static_cast<double>(samples_->at(i).lon) / 1e5;
        output << sample_lat << ", " << sample_lon << ", ";
        for (int j = 0; j < n_bins - 1; ++j) {
            descriptor[j] /= sum;
            output << descriptor[j] << ", ";
        }
        output << descriptor[n_bins - 1]/sum << endl;
    }
    output.close();
    
    /*
     ofstream output;
     output.open(filename);
     int selected_sample = *picked_sample_idxs_.begin();
     output << samples_->at(selected_sample).x << " " << samples_->at(selected_sample).y << endl;
     output<<segments_to_draw_for_samples_.size()<<endl;
     for (size_t i = 0; i < segments_to_draw_for_samples_.size(); ++i) {
     Segment s = segments_[segments_to_draw_for_samples_[i]];
     output << s.points().size() <<endl;
     for (size_t k = 0; k < s.points().size(); ++k) {
     int orig_pt_idx = s.points()[k].orig_idx;
     float angle = static_cast<float>(360 - data_->at(orig_pt_idx).head + 90) * PI / 180.0f;
     float head_x = cos(angle);
     float head_y = sin(angle);
     output << s.points()[k].x << " " << s.points()[k].y << " " << s.points()[k].t << " " << head_x << " " << head_y << " " << data_->at(orig_pt_idx).speed << endl;
     }
     }
     output.close();
     */
}

float Trajectories::computeSegmentDistance(int seg_idx1, int seg_idx2, double sigma){
    // Dynamic Time Wraping distance using different search range with t.
    Segment &seg1 = segments_[seg_idx1];
    Segment &seg2 = segments_[seg_idx2];
    
    if(seg1.points().size() < 2 || seg2.points().size() < 2)
        return POSITIVE_INFINITY;
    
    // Add head and tail nodes for DTWD
    SegPoint &seg1_start = seg1.points()[0];
    int seg1_size = seg1.points().size();
    SegPoint &seg1_end = seg1.points()[seg1_size-1];
    
    SegPoint &seg2_start = seg2.points()[0];
    int seg2_size = seg2.points().size();
    SegPoint &seg2_end = seg2.points()[seg2_size-1];
    
    float seg1_radius1 = sigma * abs(seg1_start.t);
    float seg1_radius2 = sigma * abs(seg1_end.t);
    float seg2_radius1 = sigma * abs(seg2_start.t);
    float seg2_radius2 = sigma * abs(seg2_end.t);
    
    float delta1_x = seg1_start.x - seg2_start.x;
    float delta1_y = seg1_start.y - seg2_start.y;
    float dist1 = sqrt(delta1_x*delta1_x+delta1_y*delta1_y);
    
    float delta2_x = seg1_end.x - seg2_end.x;
    float delta2_y = seg1_end.y - seg2_end.y;
    float dist2 = sqrt(delta2_x*delta2_x+delta2_y*delta2_y);
    
    float delta3_x = seg1_start.x - seg2_end.x;
    float delta3_y = seg1_start.y - seg2_end.y;
    float dist3 = sqrt(delta3_x*delta3_x+delta3_y*delta3_y);
    
    float delta4_x = seg1_end.x - seg2_start.x;
    float delta4_y = seg1_end.y - seg2_start.y;
    float dist4 = sqrt(delta4_x*delta4_x+delta4_y*delta4_y);
    
    float tmp_dist1 = dist1 - seg1_radius1 - seg2_radius1;
    float tmp_dist2 = dist2 - seg1_radius2 - seg2_radius2;
    
    if (tmp_dist1 < 0)
        tmp_dist1 = 0.0f;
    if (tmp_dist2 < 0)
        tmp_dist2 = 0.0f;
    
    float penalty = 10.0f;
    
    float dist = tmp_dist1 + tmp_dist2;
    
    if (dist1 + dist2 > dist3 + dist4)
        dist *= penalty;
    
    return dist;
}

float Trajectories::segPointDistance(int idx1, int idx2, Segment &seg1, Segment &seg2, double sigma){
    SegPoint &pt1 = seg1.points()[idx1];
    SegPoint &pt2 = seg2.points()[idx2];
    
    float dist = 0.0f;
    
    float radius1 = sigma * abs(pt1.t);
    float radius2 = sigma * abs(pt2.t);
    float sign1 = 1.0f;
    float sign2 = 1.0f;
    if (pt1.t < 0)
        sign1 = -1.0f;
    if (pt2.t < 0)
        sign2 = -1.0f;
    
    float delta_x = sign1*pt1.x - sign2*pt2.x;
    float delta_y = sign1*pt1.y - sign2*pt2.y;
    dist += sqrt(delta_x*delta_x + delta_y*delta_y);
    dist -= radius1;
    dist -= radius2;
    if (dist < 0.0f) {
        dist = 0.0f;
    }
    
    return dist;
}

void Trajectories::computeDistanceGraph(){
    if (distance_graph_ != NULL) {
        delete distance_graph_;
    }
    
    distance_graph_ = new DistanceGraph();
    float grid_size = 10; //in meters
    samplePointCloud(grid_size, distance_graph_vertices_, distance_graph_search_tree_);
    
    distance_graph_->computeGraphUsingGpsPointCloud(data_, tree_, distance_graph_vertices_, distance_graph_search_tree_, grid_size);
    prepareForVisualization(display_bound_box_);
    
    distance_graph_->Distance(data_->at(10), data_->at(10000));
}

void Trajectories::samplePointCloud(float neighborhood_size, PclPointCloud::Ptr &samples, PclSearchTree::Ptr &sample_tree){
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    printf("bound box: %.2f, %.2f, %.2f, %.2f\n", bound_box_[0], bound_box_[1], bound_box_[2], bound_box_[3]);
    
    // Clear samples
    PclPoint point;
    samples->clear();
    
    // Compute samples
    // Initialize the samples using a unified grid
    vector<Vertex> seed_samples;
    set<long> grid_idxs;
    int n_x = static_cast<int>(delta_x / neighborhood_size) + 1;
    int n_y = static_cast<int>(delta_y / neighborhood_size) + 1;
    float step_x = delta_x / n_x;
    float step_y = delta_y / n_y;
    for (size_t pt_idx = 0; pt_idx < data_->size(); ++pt_idx) {
        PclPoint &point = data_->at(pt_idx);
        long pt_i = floor((point.x - bound_box_[0]) / step_x);
        long pt_j = floor((point.y - bound_box_[2]) / step_y);
        if (pt_i == n_x){
            pt_i -= 1;
        }
        if (pt_j == n_y) {
            pt_j -= 1;
        }
        long idx = pt_i + pt_j*n_x;
        grid_idxs.insert(idx);
    }
    for (set<long>::iterator it = grid_idxs.begin(); it != grid_idxs.end(); ++it) {
        long idx = *it;
        int pt_i = idx % n_x;
        int pt_j = idx / n_x;
        float pt_x = bound_box_[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box_[2] + (pt_j+0.5)*step_y;
        seed_samples.push_back(Vertex(pt_x, pt_y, 0.0));
    }
    
    int n_iter = 1;
    float search_radius = (step_x > step_y) ? step_x : step_y;
    search_radius /= sqrt(2.0);
    search_radius += 1.0;
    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter) {
        // For each seed, search its neighbors and do an average
        for (vector<Vertex>::iterator it = seed_samples.begin(); it != seed_samples.end(); ++it) {
            PclPoint search_point;
            search_point.setCoordinate((*it).x, (*it).y, 0.0f);
            // Search nearby and mark nearby
            vector<int> k_indices;
            vector<float> k_distances;
            tree_->radiusSearch(search_point, search_radius, k_indices, k_distances);
            
            if (k_indices.size() == 0) {
                printf("WARNING!!! Empty sampling cell! %.1f, %.1f\n", search_point.x, search_point.y);
                continue;
            }
            float sum_x = 0.0;
            float sum_y = 0.0;
            for (vector<int>::iterator neighbor_it = k_indices.begin(); neighbor_it != k_indices.end(); ++neighbor_it) {
                PclPoint &nb_pt = data_->at(*neighbor_it);
                sum_x += nb_pt.x;
                sum_y += nb_pt.y;
            }
            
            sum_x /= k_indices.size();
            sum_y /= k_indices.size();
            
            point.setCoordinate(sum_x, sum_y, 0.0f);
            point.t = 0;
            point.id_trajectory = 0;
            point.id_sample = 0;
            
            samples->push_back(point);
        }
    }
    
    sample_tree->setInputCloud(samples);
    printf("Sampling Done. Totally %zu samples.\n", samples->size());
}

void Trajectories::samplePointCloud(float neighborhood_size){
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
    if (neighborhood_size > delta_x || neighborhood_size > delta_y) {
        fprintf(stderr, "ERROR: Neighborhood size is bigger than the trajectory boundbox!\n");
        return;
    }
    
    // Clear samples
    PclPoint point;
    samples_->clear();
    picked_sample_idxs_.clear();
    
    normalized_sample_locs_.clear();
    sample_vertex_colors_.clear();
    
    // Compute samples
    // Initialize the samples using a unified grid
    vector<Vertex> seed_samples;
    set<int> grid_idxs;
    int n_x = static_cast<int>(delta_x / neighborhood_size) + 1;
    int n_y = static_cast<int>(delta_y / neighborhood_size) + 1;
    float step_x = delta_x / n_x;
    float step_y = delta_y / n_y;
    for (size_t pt_idx = 0; pt_idx < data_->size(); ++pt_idx) {
        PclPoint &point = data_->at(pt_idx);
        int pt_i = floor((point.x - bound_box_[0]) / step_x);
        int pt_j = floor((point.y - bound_box_[2]) / step_y);
        if (pt_i == n_x){
            pt_i -= 1;
        }
        if (pt_j == n_y) {
            pt_j -= 1;
        }
        int idx = pt_i + pt_j*n_x;
        grid_idxs.insert(idx);
    }
    for (set<int>::iterator it = grid_idxs.begin(); it != grid_idxs.end(); ++it) {
        int idx = *it;
        int pt_i = idx % n_x;
        int pt_j = idx / n_x;
        float pt_x = bound_box_[0] + (pt_i+0.5)*step_x;
        float pt_y = bound_box_[2] + (pt_j+0.5)*step_y;
        seed_samples.push_back(Vertex(pt_x, pt_y, 0.0));
    }
    
    int n_iter = 1;
    float search_radius = (step_x > step_y) ? step_x : step_y;
    search_radius /= sqrt(2.0);
    search_radius += 1.0;
    for (size_t i_iter = 0; i_iter < n_iter; ++i_iter) {
        // For each seed, search its neighbors and do an average
        for (vector<Vertex>::iterator it = seed_samples.begin(); it != seed_samples.end(); ++it) {
            PclPoint search_point;
            search_point.setCoordinate((*it).x, (*it).y, 0.0f);
            // Search nearby and mark nearby
            vector<int> k_indices;
            vector<float> k_distances;
            tree_->radiusSearch(search_point, search_radius, k_indices, k_distances);
            
            if (k_indices.size() == 0) {
                printf("WARNING!!! Empty sampling cell!\n");
                continue;
            }
            float sum_x = 0.0;
            float sum_y = 0.0;
            double sum_lat = 0;
            double sum_lon = 0;
            for (vector<int>::iterator neighbor_it = k_indices.begin(); neighbor_it != k_indices.end(); ++neighbor_it) {
                PclPoint &nb_pt = data_->at(*neighbor_it);
                sum_x += nb_pt.x;
                sum_y += nb_pt.y;
                sum_lat += nb_pt.lat;
                sum_lon += nb_pt.lon;
            }
            
            sum_x /= k_indices.size();
            sum_y /= k_indices.size();
            sum_lat /= k_indices.size();
            sum_lon /= k_indices.size();
            
            point.setCoordinate(sum_x, sum_y, 0.0f);
            point.lat = floor(sum_lat);
            point.lon = floor(sum_lon);
            point.t = 0;
            point.id_trajectory = 0;
            point.id_sample = 0;
            
            samples_->push_back(point);
            sample_vertex_colors_.push_back(sample_color_);
        }
    }
    
    sample_tree_->setInputCloud(samples_);
    printf("Sampling Done. Totally %zu samples.\n", samples_->size());
}

void Trajectories::pickAnotherSampleNear(float x, float y, float max_distance){
    if (samples_->size() == 0)
        return;
    
    PclPoint center_point;
    center_point.setCoordinate(x, y, 0.0f);
    vector<int> k_indices;
    vector<float> k_sqr_distances;
    
    sample_tree_->nearestKSearch(center_point, 1, k_indices, k_sqr_distances);
    
    if (k_indices.empty())
        return;
    
    if (sqrt(k_sqr_distances[0]) < search_distance){
        auto ret = picked_sample_idxs_.emplace(k_indices[0]);
        if (ret.second) {
            // Update segment_to_draw_for_samples_ and corresponding segment vertices and colors
            selectPathletsWithSearchRange(search_range_);
        }
    }
}

void Trajectories::clearPickedSamples(void){
    picked_sample_idxs_.clear();
    pathlet_to_draw_.clear();
}
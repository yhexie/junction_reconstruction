#include "graph.h"
#include <ctime>
#include <Eigen/Dense>
#include <algorithm>
using namespace Eigen;

static const float arrow_size = 5.0f;   // in meters
static const int arrow_angle = 15; // in degrees

struct location{
    float x, y;
    location() {x = -1; y = -1;}
    location(float vx, float vy) {x = vx; y = vy;}
};

// euclidean distance heuristic
template <class Graph, class CostType, class LocMap>
class distance_heuristic : public astar_heuristic<Graph, CostType>
{
public:
    typedef typename graph_traits<Graph>::vertex_descriptor Vertex;
    distance_heuristic(LocMap l, Vertex goal)
    : m_location(l), m_goal(goal) {}
    CostType operator()(Vertex u)
    {
        CostType dx = m_location[m_goal].x - m_location[u].x;
        CostType dy = m_location[m_goal].y - m_location[u].y;
        return ::sqrt(dx * dx + dy * dy);
    }
private:
    LocMap m_location;
    Vertex m_goal;
};

struct found_goal {}; // exception for termination

// visitor that terminates when we find the goal
template <class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor
{
public:
    astar_goal_visitor(Vertex goal) : m_goal(goal) {}
    template <class Graph>
    void examine_vertex(Vertex u, Graph& g) {
        if(u == m_goal)
            throw found_goal();
    }
private:
    Vertex m_goal;
};

Graph::Graph(QObject *parent) : Renderable(parent){
    line_width_ = 4.0f;
    point_size_ = 5.0f;
    
    paths_to_draw_.clear();
    g_ = NULL;
}

Graph::~Graph(){
    if(g_ != NULL)
        delete g_;
}

void Graph::draw(){
    if (vertices_ == NULL) {
        return;
    }
    if (vertices_->size() == 0){
        return;
    }
    // Draw Edges
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&normalized_vertices_locs_[0], 3*normalized_vertices_locs_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&vertex_colors_[0], 4*vertex_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
    element_buffer.create();
    element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    element_buffer.bind();
    element_buffer.allocate(&(edge_idxs_[0]), edge_idxs_.size()*sizeof(int));
    glLineWidth(line_width_);
    glDrawElements(GL_LINES, edge_idxs_.size(), GL_UNSIGNED_INT, 0);
    
    // Draw Vertices
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&normalized_vertices_locs_[0], 3*normalized_vertices_locs_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&vertex_colors_[0], 4*vertex_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    glPointSize(point_size_);
    glDrawArrays(GL_POINTS, 0, normalized_vertices_locs_.size());
    
    // Draw Directions
    QOpenGLBuffer vertex_buffer;
    vertex_buffer.create();
    vertex_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertex_buffer.bind();
    vertex_buffer.allocate(&edge_direction_vertices_[0], 3*edge_direction_vertices_.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    QOpenGLBuffer color_buffer;
    color_buffer.create();
    color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    color_buffer.bind();
    color_buffer.allocate(&edge_direction_colors_[0], 4*edge_direction_colors_.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    element_buffer.create();
    element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    element_buffer.bind();
    element_buffer.allocate(&(edge_direction_idxs_[0]), edge_direction_idxs_.size()*sizeof(int));
    glLineWidth(line_width_);
    glDrawElements(GL_TRIANGLES, edge_direction_idxs_.size(), GL_UNSIGNED_INT, 0);
    
    // Draw paths
    if(paths_to_draw_.size() == 0)
        return;
    
    vector<Vertex>  path_vertices;
    vector<Color>   path_colors;
    vector<vector<int>> path_idxs;
    
    for (size_t path_id = 0; path_id < paths_to_draw_.size(); ++path_id) {
        vector<int> &path = paths_to_draw_[path_id];
        vector<int> path_idx(path.size(), 0);
        Color path_color = ColorMap::getInstance().getDiscreteColor(path_id);
        for (size_t j = 0; j < path.size(); ++j) {
            int pt_idx = path[j];
            float alpha = 1.0 - 0.7 * static_cast<float>(j) / path.size();
            path_idx[j] = path_vertices.size();
            path_vertices.push_back(Vertex(normalized_vertices_locs_[pt_idx].x, normalized_vertices_locs_[pt_idx].y, Z_PATH));
            Color pt_color(path_color.r*alpha, path_color.g*alpha, path_color.b*alpha, 1.0);
            path_colors.push_back(pt_color);
        }
        path_idxs.push_back(path_idx);
    }
    
    vertexPositionBuffer_.create();
    vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexPositionBuffer_.bind();
    vertexPositionBuffer_.allocate(&path_vertices[0], 3*path_vertices.size()*sizeof(float));
    shadder_program_->setupPositionAttributes();
    
    vertexColorBuffer_.create();
    vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vertexColorBuffer_.bind();
    vertexColorBuffer_.allocate(&path_colors[0], 4*path_colors.size()*sizeof(float));
    shadder_program_->setupColorAttributes();
    
    for (int i=0; i < path_idxs.size(); ++i) {
        QOpenGLBuffer element_buffer(QOpenGLBuffer::IndexBuffer);
        element_buffer.create();
        element_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        element_buffer.bind();
        element_buffer.allocate(&(path_idxs[i][0]), path_idxs[i].size()*sizeof(size_t));
        glLineWidth(2*line_width_);
        glDrawElements(GL_LINE_STRIP, path_idxs[i].size(), GL_UNSIGNED_INT, 0);
        glPointSize(2*point_size_);
        glDrawElements(GL_POINTS, path_idxs[i].size(), GL_UNSIGNED_INT, 0);
    }
}

void Graph::prepareForVisualization(QVector4D bound_box){
    if(vertices_ == NULL)
        return;
    
    normalized_vertices_locs_.clear();
    vertex_colors_.clear();
    edge_direction_vertices_.clear();
    edge_direction_idxs_.clear();
    edge_direction_colors_.clear();
    
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    if (delta_x < 0 || delta_y < 0) {
        fprintf(stderr, "Trajectory bounding box error! Min greater than Max!\n");
    }
    
    float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
    float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
    scale_factor_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
    
    for (size_t i = 0; i < vertices_->size(); ++i){
        const PclPoint &point = vertices_->at(i);
        float n_x = (point.x - center_x) / scale_factor_;
        float n_y = (point.y - center_y) / scale_factor_;
        normalized_vertices_locs_.push_back(Vertex(n_x, n_y, Z_GRAPH));
    }
    
    Color dark_gray = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
    vertex_colors_.resize(vertices_->size(), dark_gray);
    
    if(edge_idxs_.size() == 0)
        return;
    
    float arrow_length = arrow_size / scale_factor_;
    int n_edge = edge_idxs_.size() / 2;
    
    for(size_t edge_idx = 0; edge_idx < n_edge; ++edge_idx){
        int from = edge_idxs_[edge_idx*2];
        int to = edge_idxs_[edge_idx*2+1];
        
        float dx = normalized_vertices_locs_[to].x - normalized_vertices_locs_[from].x;
        float dy = normalized_vertices_locs_[to].y  - normalized_vertices_locs_[from].y;
        float length = sqrt(dx*dx + dy*dy);
        
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
        float center_x = 0.2*normalized_vertices_locs_[from].x + 0.8*normalized_vertices_locs_[to].x;
        float center_y = 0.2*normalized_vertices_locs_[from].y + 0.8*normalized_vertices_locs_[to].y;
        edge_direction_idxs_.push_back(edge_direction_vertices_.size());
        edge_direction_vertices_.push_back(Vertex(center_x - vec1.x(), center_y - vec1.y(), Z_GRAPH));
        edge_direction_idxs_.push_back(edge_direction_vertices_.size());
        edge_direction_vertices_.push_back(Vertex(center_x, center_y, Z_GRAPH));
        edge_direction_idxs_.push_back(edge_direction_vertices_.size());
        edge_direction_vertices_.push_back(Vertex(center_x - vec2.x(), center_y - vec2.y(), Z_GRAPH));
        edge_direction_colors_.push_back(dark_gray);
        edge_direction_colors_.push_back(dark_gray);
        edge_direction_colors_.push_back(dark_gray);
    }
}

void Graph::updateGraphUsingDescriptor(PclPointCloud::Ptr &cluster_centers, PclSearchTree::Ptr &cluster_center_search_tree, cv::Mat *descriptors, vector<int> &cluster_popularity, PclPointCloud::Ptr &gps_pointcloud, PclSearchTree::Ptr &gps_pointcloud_search_tree){
    // Reset point clouds
    vertices_ = cluster_centers;
    tree_ = cluster_center_search_tree;
    
    gps_pointcloud_ = gps_pointcloud;
    gps_pointcloud_search_tree_ = gps_pointcloud_search_tree;
    
    edge_idxs_.clear();
    int num_nodes = cluster_centers->size();
    g_ = new graph_t(num_nodes);
    locations_.clear();
    locations_.resize(num_nodes);
    for (size_t pt_idx = 0; pt_idx < cluster_centers->size(); ++pt_idx) {
        PclPoint &pt = cluster_centers->at(pt_idx);
        locations_[pt_idx].x = pt.x;
        locations_[pt_idx].y = pt.y;
    }
    
    typedef graph_t::vertex_descriptor vertex;
    typedef graph_t::edge_descriptor edge_descriptor;
    WeightMap weightmap = get(edge_weight, *g_);
    float SEARCH_RANGE = 30.0f; // in meters
    float MIN_DISTANCE_SQUARE = 225.0f; // in meters^2
    float MIN_DISTANCE_SQUARE_FOR_EDGE = 225.0f; // in meters^2
    int K = 2;
    for (size_t sample_idx = 0; sample_idx < cluster_centers->size(); ++sample_idx) {
        vector<int> k_indices;
        vector<float> k_distances;
        
        cv::Mat d1 = descriptors->row(sample_idx);
        tree_->radiusSearch(cluster_centers->at(sample_idx), SEARCH_RANGE, k_indices, k_distances);
       
        vector<int> to_idxs_in_indices;
        vector<float> descriptor_distances;
        vector<float> to_sort;
        for (size_t neighbor_idx = 0; neighbor_idx < k_indices.size(); ++neighbor_idx) {
            if (k_distances[neighbor_idx] < MIN_DISTANCE_SQUARE_FOR_EDGE) {
                // Add edge
                if(k_distances[neighbor_idx] > 0.1){
                    int to_idx = k_indices[neighbor_idx];
                    // Check edge compatibility with the descriptor
                    float delta_x = cluster_centers->at(to_idx).x - cluster_centers->at(sample_idx).x;
                    float delta_y = cluster_centers->at(to_idx).y - cluster_centers->at(sample_idx).y;
                    float r = sqrt(delta_x*delta_x + delta_y*delta_y);
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
                    
                    int bin_1_id = theta_bin_id + 8;
                    int bin_2_id = theta_bin_id + 4 + 8;
                    bool is_compatible = false;
                    if (d1.at<float>(0, bin_1_id) > 1e-3) {
                        is_compatible = true;
                    }
                    if (d1.at<float>(0, bin_2_id) > 1e-3) {
                        is_compatible = true;
                    }
                
                    if (is_compatible) {
                        edge_descriptor e;
                        bool inserted;
                        tie(e, inserted) = add_edge(sample_idx, to_idx, *g_);
                        float edge_weight = sqrt(k_distances[neighbor_idx]);
                        weightmap[e] = edge_weight;
                        edge_idxs_.push_back(sample_idx);
                        edge_idxs_.push_back(to_idx);
                    }
                }
                continue;
            }
            
            if (k_distances[neighbor_idx] < MIN_DISTANCE_SQUARE) {
                continue;
            }
            
            if (cv::norm(d1) < 0.1)
                continue;
            
            size_t to_idx = static_cast<size_t>(k_indices[neighbor_idx]);
            // Compare descriptor distance
            to_idxs_in_indices.push_back(neighbor_idx);
            cv::Mat d2 = descriptors->row(to_idx);
            float descriptor_distance = cv::norm(d2-d1);
            if (cv::norm(d2) < 0.1) {
                continue;
            }
            
            // Check edge compatibility with the descriptor
            float delta_x = cluster_centers->at(to_idx).x - cluster_centers->at(sample_idx).x;
            float delta_y = cluster_centers->at(to_idx).y - cluster_centers->at(sample_idx).y;
            float r = sqrt(delta_x*delta_x + delta_y*delta_y);
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
            
            int bin_1_id = theta_bin_id + 8;
            int bin_2_id = theta_bin_id + 4 + 8;
            bool is_compatible = false;
            if (d1.at<float>(0, bin_1_id) > 1e-3) {
                is_compatible = true;
            }
            if (d1.at<float>(0, bin_2_id) > 1e-3) {
                is_compatible = true;
            }
            
            if (!is_compatible) {
                descriptor_distance *= 100;
            }
            
            descriptor_distances.push_back(descriptor_distance);
            to_sort.push_back(descriptor_distance);
        }
        
        if (to_sort.size() == 0)
            continue;
        
        std::sort(to_sort.begin(), to_sort.end());
        int range_idx = to_sort.size() - 1;
        if (to_sort.size() > K) {
            range_idx = K;
        }
        
        float range_dist = to_sort[K];
        int n_visited = 0;
        for (size_t i = 0; i < descriptor_distances.size(); ++i) {
            if (descriptor_distances[i] <= range_dist) {
                // Add edge
                int to_idx = k_indices[to_idxs_in_indices[i]];
                int popularity = (cluster_popularity[sample_idx] < cluster_popularity[to_idx]) ? cluster_popularity[sample_idx] : cluster_popularity[to_idx] + 1;
                
                float dist = sqrt(k_distances[to_idxs_in_indices[i]]);
                dist /= popularity;
                
                edge_descriptor e;
                bool inserted;
                tie(e, inserted) = add_edge(sample_idx, to_idx, *g_);
                float edge_weight = dist; //* descriptor_distances[i];
                weightmap[e] = edge_weight;
                edge_idxs_.push_back(sample_idx);
                edge_idxs_.push_back(to_idx);
                n_visited += 1;
                if (n_visited >= K) {
                    break;
                }
            }
        }
        
    }
}

void Graph::updateGraphUsingSamplesAndGpsPointCloud(PclPointCloud::Ptr &samples, PclSearchTree::Ptr &sample_search_tree, PclPointCloud::Ptr &gps_pointcloud, PclSearchTree::Ptr &gps_pointcloud_search_tree){
    // Reset point clouds
    vertices_ = samples;
    tree_ = sample_search_tree;
    
    gps_pointcloud_ = gps_pointcloud;
    gps_pointcloud_search_tree_ = gps_pointcloud_search_tree;
    
    edge_idxs_.clear();
    int num_nodes = samples->size();
    g_ = new graph_t(num_nodes);
    locations_.clear();
    locations_.resize(num_nodes);
    for (size_t pt_idx = 0; pt_idx < samples->size(); ++pt_idx) {
        PclPoint &pt = samples->at(pt_idx);
        locations_[pt_idx].x = pt.x;
        locations_[pt_idx].y = pt.y;
    }
    
    typedef graph_t::vertex_descriptor vertex;
    typedef graph_t::edge_descriptor edge_descriptor;
    WeightMap weightmap = get(edge_weight, *g_);
    for (size_t sample_idx = 0; sample_idx < samples->size(); ++sample_idx) {
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->nearestKSearch(samples->at(sample_idx), 10, k_indices, k_distances);
        
        for (size_t neighbor_idx = 0; neighbor_idx < k_indices.size(); ++neighbor_idx) {
            size_t to_idx = static_cast<size_t>(k_indices[neighbor_idx]);
            if (to_idx <= sample_idx) {
                continue;
            }
            
            Vector2d v(vertices_->at(to_idx).x - vertices_->at(sample_idx).x,
                       vertices_->at(to_idx).y - vertices_->at(sample_idx).y);
            float edge_length = v.norm();
            v.normalize();
            
            vector<int> nearby_pt_idxs;
            vector<float> nearby_pt_dists;
            
            gps_pointcloud_search_tree_->radiusSearch(samples->at(sample_idx), edge_length, nearby_pt_idxs, nearby_pt_dists);
            
            int n_knots = static_cast<int>(edge_length);
            vector<float> density(n_knots, 0.0f);
            for (size_t i = 0; i < nearby_pt_idxs.size(); ++i) {
                int pt_idx = nearby_pt_idxs[i];
                Vector2d v1(gps_pointcloud_->at(pt_idx).x - vertices_->at(sample_idx).x,
                            gps_pointcloud_->at(pt_idx).y - vertices_->at(sample_idx).y);
                float v1_length = v1.norm();
                float dot_value = v1.dot(v);
                
                float v_dist = sqrt(v1_length*v1_length - dot_value*dot_value);
                if (v_dist > 3.0f) {
                    continue;
                }
                
                if (dot_value >= 0 && dot_value <= edge_length) {
                    for (int j = 0; j < n_knots; ++j) {
                        float delta = dot_value - j;
                        density[j] += exp(-1*delta*delta/2.0f);
                    }
                }
            }
            
            float pt_density = 1e6;
            for (int i = 0; i < n_knots; ++i) {
                if (pt_density > density[i]) {
                    pt_density = density[i];
                }
            }
            
            if (to_idx > sample_idx && pt_density > 1e-3) {
                edge_descriptor e1;
                bool inserted1;
                tie(e1, inserted1) = add_edge(sample_idx, to_idx, *g_);
                float edge_weight = sqrt(k_distances[neighbor_idx]) + 20.0 /pt_density;
                weightmap[e1] = edge_weight;
                edge_idxs_.push_back(sample_idx);
                edge_idxs_.push_back(to_idx);
                edge_descriptor e2;
                bool inserted2;
                tie(e2, inserted2) = add_edge(to_idx, sample_idx, *g_);
                weightmap[e2] = edge_weight;
                edge_idxs_.push_back(to_idx);
                edge_idxs_.push_back(sample_idx);
            }
        }
    }
}

void Graph::updateGraphUsingSamplesAndSegments(PclPointCloud::Ptr &samples, PclSearchTree::Ptr &sample_search_tree, vector<Segment> &segments, vector<vector<int>> &cluster_centers, vector<vector<int>> &cluster_sizes, PclPointCloud::Ptr &gps_pointcloud, PclSearchTree::Ptr &gps_pointcloud_search_tree){
    /*
     This function generate a graph by connecting sample points based on the compatibility of their corresponding segment cluster centers. Each sample point has several clustered segment center, and we compute the pairwise distances between these segment centers and decide whether a directional edge should be added to the graph. The target use of the resulting graph is to interpolate sparse trajectories to generate denser trajectories.
     
     Parameters:
     - samples: sample point cloud pointer;
     - sample_search_tree: sample point cloud kdtree pointer;
     - segments: reference to a lists of all segments extracted;
     - cluster_centers: each element is a list of segment idxs for each sample pointj;
     - gps_pointcloud: original gps point cloud pointer;
     - gps_pointcloud_search_tree: gps point cloud kdtree pointer.
     */
    
    // Reset point clouds
    vertices_ = samples;
    tree_ = sample_search_tree;
    
    gps_pointcloud_ = gps_pointcloud;
    gps_pointcloud_search_tree_ = gps_pointcloud_search_tree;
    
    edge_idxs_.clear();
    int num_nodes = samples->size();
    
    if (g_ != NULL) {
        delete g_;
    }
    
    g_ = new graph_t(num_nodes);
    
    // Copy sample locations to locations_ for graph initialization.
    locations_.clear();
    locations_.resize(num_nodes);
    for (size_t pt_idx = 0; pt_idx < samples->size(); ++pt_idx) {
        PclPoint &pt = samples->at(pt_idx);
        locations_[pt_idx].x = pt.x;
        locations_[pt_idx].y = pt.y;
    }
    
    typedef graph_t::vertex_descriptor vertex;
    typedef graph_t::edge_descriptor edge_descriptor;
    WeightMap weightmap = get(edge_weight, *g_);
    
    // Iterate over the samples
    for (size_t sample_idx = 0; sample_idx < samples->size(); ++sample_idx) {
        // Find nearby other samples
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->nearestKSearch(samples->at(sample_idx), 11, k_indices, k_distances);
        
        for (size_t neighbor_idx = 0; neighbor_idx < k_indices.size(); ++neighbor_idx) {
            size_t to_idx = static_cast<size_t>(k_indices[neighbor_idx]);
            
            if (to_idx < sample_idx) {
                continue;
            }
            
            // Check the compatibility between its cluster centers
            vector<int> &from_cluster_centers = cluster_centers[sample_idx];
            vector<int> &to_cluster_centers = cluster_centers[to_idx];
            Vector2d v(vertices_->at(to_idx).x - vertices_->at(sample_idx).x,
                       vertices_->at(to_idx).y - vertices_->at(sample_idx).y);
            
            float edge_length = v.norm();
            v.normalize();
            
            vector<int> nearby_pt_idxs;
            vector<float> nearby_pt_dists;
            
            gps_pointcloud_search_tree_->radiusSearch(samples->at(sample_idx), edge_length, nearby_pt_idxs, nearby_pt_dists);
            
            int n_knots = static_cast<int>(edge_length);
            vector<float> density(n_knots, 0.0f);
            for (size_t i = 0; i < nearby_pt_idxs.size(); ++i) {
                int pt_idx = nearby_pt_idxs[i];
                Vector2d v1(gps_pointcloud_->at(pt_idx).x - vertices_->at(sample_idx).x,
                            gps_pointcloud_->at(pt_idx).y - vertices_->at(sample_idx).y);
                float v1_length = v1.norm();
                float dot_value = v1.dot(v);
                
                float v_dist = sqrt(v1_length*v1_length - dot_value*dot_value);
                if (v_dist > 3.0f) {
                    continue;
                }
                
                if (dot_value >= 0 && dot_value <= edge_length) {
                    for (int j = 0; j < n_knots; ++j) {
                        float delta = dot_value - j;
                        density[j] += exp(-1*delta*delta/2.0f);
                    }
                }
            }
            
            float pt_density = 1e6;
            for (int i = 0; i < n_knots; ++i) {
                if (pt_density > density[i]) {
                    pt_density = density[i];
                }
            }
            
            if (pt_density < 1.0f) {
                continue;
            }
            
            bool edge_inserted = false;
            bool reverse_edge_inserted = false;
            for (size_t c1 = 0; c1 < from_cluster_centers.size(); ++c1) {
                for (size_t c2 = 0; c2 < to_cluster_centers.size(); ++c2) {
                    Segment &seg1 = segments[from_cluster_centers[c1]];
                    Segment &seg2 = segments[to_cluster_centers[c2]];
                    float distance = simpleSegDistance(seg1, seg2);
                    if (distance < 100) {
                        if (edge_inserted && reverse_edge_inserted) {
                            break;
                        }
                        
                        int cluster_size1 = cluster_sizes[sample_idx][c1];
                        int cluster_size2 = cluster_sizes[to_idx][c2];
                        int cluster_size = (cluster_size1 < cluster_size2) ? cluster_size1 : cluster_size2;
                        printf("cluster_size = %d\n", cluster_size);
                        
                        int seg1_size = seg1.points().size();
                        Vector2d v1(seg1.points()[seg1_size-1].x - vertices_->at(sample_idx).x,
                                    seg1.points()[seg1_size-1].y - vertices_->at(sample_idx).y);
                        Vector2d v2(vertices_->at(to_idx).x - seg2.points()[0].x,
                                    vertices_->at(to_idx).y - seg2.points()[0].y);
                        bool dot1_positive = false;
                        bool dot2_positive = false;
                        if (v.dot(v1) > 0) {
                            dot1_positive = true;
                        }
                        if (v.dot(v2) > 0) {
                            dot2_positive = true;
                        }
                        
                        if (dot1_positive && dot2_positive && !edge_inserted){
                            // Add an edge
                            edge_descriptor e;
                            bool inserted;
                            tie(e, inserted) = add_edge(sample_idx, to_idx, *g_);
                            float edge_weight = v.norm();
                            weightmap[e] = edge_weight / cluster_size;
                            edge_idxs_.push_back(sample_idx);
                            edge_idxs_.push_back(to_idx);
                            edge_inserted = true;
                        }
                        
                        if (!dot1_positive && !dot2_positive && !reverse_edge_inserted) {
                            // Add an edge
                            edge_descriptor e;
                            bool inserted;
                            tie(e, inserted) = add_edge(to_idx, sample_idx, *g_);
                            float edge_weight = v.norm();
                            weightmap[e] = edge_weight / cluster_size;
                            edge_idxs_.push_back(to_idx);
                            edge_idxs_.push_back(sample_idx);
                            reverse_edge_inserted = true;
                        }
                    }
                }
                
                if (edge_inserted && reverse_edge_inserted) {
                    break;
                }
            }
        }
    }
}

bool Graph::shortestPath(int start, int goal, vector<int> &path){
    if (vertices_->empty()) {
        printf("Warning: cannot search in empty graph!\n");
        return false;
    }
    
    if (g_ == NULL){
        printf("Warning: cannot search in empty graph!\n");
        return false;
    }
    
    typedef graph_t::vertex_descriptor vertex;
    typedef graph_t::edge_descriptor edge_descriptor;
    path.clear();
    vector<graph_t::vertex_descriptor> p(num_vertices(*g_));
    vector<float> d(num_vertices(*g_));
    try {
        // call astar named parameter interface
        astar_search_tree(*g_, start, distance_heuristic<graph_t, float, location*>(&locations_[0], goal), predecessor_map(make_iterator_property_map(p.begin(), get(vertex_index, *g_))).distance_map(make_iterator_property_map(d.begin(), get(vertex_index, *g_))).visitor(astar_goal_visitor<vertex>(goal)));
        
    } catch(found_goal fg){ // found a path to the goal
        list<vertex> shortest_path;
        for(vertex v = goal;; v = p[v]) {
            shortest_path.push_front(v);
            if(p[v] == v)
                break;
        }
        list<vertex>::iterator spi = shortest_path.begin();
        path.push_back(*spi);
        //cout << start;
        for(++spi; spi != shortest_path.end(); ++spi){
            path.push_back(*spi);
        }
        return true;
    }
   
    return false;
    //cout << "Didn't find a path from " << start << " to "<< goal << "!" << endl;
}

void Graph::shortestPathInterpolation(vector<int> &query_path, vector<int> &result_path){
    /*
     Given a query path which contains multiple GPS points, this function compute a shortest path interpolation on the graph.
     
     Arguments:
     - query_path: a vector of GPS point indices pointing to points in vertices_. For example, [2,5,7] is a path containing 3 GPS points: [gps_pointcloud_->at(2), gps_pointcloud_->at(5), gps_pointcloud_->at(7)];
     - result_path: the resulting path as a path on graph g_. Notice the indices are point to vertices_;
     */
    if (vertices_ == NULL)
        return;
    
    if (vertices_->empty()) {
        printf("Warning: cannot search in empty graph!\n");
        return;
    }
    
    if (gps_pointcloud_->empty()) {
        printf("Warning: cannot search with empty gps_pointcloud!\n");
        return;
    }
    
    if (g_ == NULL){
        printf("Warning: cannot search in empty graph!\n");
        return;
    }
    
    if (query_path.empty()) {
        result_path.clear();
        return;
    }
    
    // For each GPS point in the query_path, search its nearby graph vertices.
    vector<vector<int>> nearby_vertex_idxs(query_path.size());
    for(size_t idx = 0; idx < query_path.size(); ++idx){
        PclPoint &pt = gps_pointcloud_->at(query_path[idx]);
        vector<int> k_indices;
        vector<float> k_sqr_distances;
        tree_->nearestKSearch(pt, 10, k_indices, k_sqr_distances);
        if (!k_indices.empty()) {
            float min_distance_square = k_sqr_distances[0] + 0.1;
            nearby_vertex_idxs[idx].push_back(k_indices[0]);
            for (size_t i = 1; i < 10; ++i) {
                if (k_sqr_distances[i] < min_distance_square) {
                    nearby_vertex_idxs[idx].push_back(k_indices[i]);
                }
                else{
                    break;
                }
            }
        }
    }
    
    // Compute shortest path between adjascent vertices.
    result_path.clear();
    for (size_t idx = 0; idx < nearby_vertex_idxs.size() - 1; ++idx) {
        vector<int> &from_vertex_idxs = nearby_vertex_idxs[idx];
        vector<int> &to_vertex_idxs = nearby_vertex_idxs[idx+1];
        vector<int> path;
        bool found_path = false;
        for (size_t i = 0; i < from_vertex_idxs.size(); ++i) {
            for (size_t j = 0; j < to_vertex_idxs.size(); ++j) {
                int start = from_vertex_idxs[i];
                int goal = to_vertex_idxs[j];
                if (start == goal) {
                    continue;
                }
                vector<int> tmp_path;
                if (shortestPath(start, goal, tmp_path)) {
                    for (size_t k = 0; k < tmp_path.size(); ++k) {
                        path.push_back(tmp_path[k]);
                    }
                    found_path = true;
                    break;
                }
            }
            if (found_path)
                break;
        }
        
        if (path.size() == 0) {
            result_path.push_back(from_vertex_idxs[0]);
        }
        else{
            // Insert the whole path except the last point into result
            for (size_t path_pt_idx = 0; path_pt_idx < path.size() - 1; ++path_pt_idx) {
                result_path.push_back(path[path_pt_idx]);
            }
        }
    }
    result_path.push_back(nearby_vertex_idxs.back()[0]);
}

// Simple segment distance
float Graph::simpleSegDistance(Segment &seg1, Segment &seg2){
    if(seg1.points().size() < 2 || seg2.points().size() < 2){
        return POSITIVE_INFINITY;
    }
    
    SegPoint seg1_start = seg1.points()[0];
    SegPoint seg2_start = seg2.points()[0];
    SegPoint seg1_end = seg1.points()[seg1.points().size()-1];
    SegPoint seg2_end = seg2.points()[seg2.points().size()-1];
    
    float sigma = 0.5f;
    float seg1_radius1 = sigma * abs(seg1_start.t);
    float seg1_radius2 = sigma * abs(seg1_end.t);
    float seg2_radius1 = sigma * abs(seg2_start.t);
    float seg2_radius2 = sigma * abs(seg2_end.t);
    
    float delta1_x = seg1_start.x - seg2_start.x;
    float delta1_y = seg1_start.y - seg2_start.y;
    float dist1 = sqrt(delta1_x*delta1_x+delta1_y*delta1_y) - seg1_radius1 - seg2_radius1;
    if (dist1 < 0.0f) {
        dist1 = 0.0f;
    }
    float delta2_x = seg1_end.x - seg2_end.x;
    float delta2_y = seg1_end.y - seg2_end.y;
    float dist2 = sqrt(delta2_x*delta2_x + delta2_y*delta2_y) - seg1_radius2 - seg2_radius2;
    
    if (dist2 < 0.0f) {
        dist2 = 0.0f;
    }
    
    return dist1+dist2;
}

// Graph Dynamic Time Warping Distance for two graph paths
float Graph::SegDistance(vector<int> &query_path1, vector<int> &query_path2){
    vector<int> result_path1;
    vector<int> result_path2;
    shortestPathInterpolation(query_path1, result_path1);
    shortestPathInterpolation(query_path2, result_path2);
    return DTWDistance(result_path1, result_path2);
}

float Graph::DTWDistance(vector<int> &path1, vector<int> &path2){
    if(path1.empty() || path2.empty())
        return 1e6;
    
    vector<int> new_path1(path1.size()+2, -1);
    vector<int> new_path2(path2.size()+2, -1);
    
    for (size_t i = 0; i < path1.size(); ++i) {
        new_path1[i+1] = path1[i];
    }
    
    for (size_t i = 0; i < path2.size(); ++i) {
        new_path2[i+1] = path2[i];
    }
    
    new_path1[0] = -2;
    new_path2[0] = -2;
    
    int n = path1.size() + 2;
    int m = path2.size() + 2;
    vector<vector<int>> DTW(n+1, vector<int>(m+1, 1e6));
    
    DTW[0][0] = 0;
    
    for (size_t i = 1; i < n+1; ++i) {
        for (size_t j = 1; j < m+1; ++j) {
            bool is_head1 = false;
            bool is_head2 = false;
            bool is_tail1 = false;
            bool is_tail2 = false;
            if (i == 2) {
                is_head1 = true;
            }
            if (j == 2){
                is_head2 = true;
            }
            if (i == n-1) {
                is_tail1 = true;
            }
            if (j == m-1) {
                is_tail2 = true;
            }
            
            float cost = Distance(new_path1[i-1], new_path2[j-1], is_head1, is_head2, is_tail1, is_tail2);
            float min1 = (DTW[i-1][j] < DTW[i][j-1]) ? DTW[i-1][j] : DTW[i][j-1];
            float min2 = (min1 < DTW[i-1][j-1]) ? min1 : DTW[i-1][j-1];
            DTW[i][j] = cost + min2;
        }
    }
    
    return DTW[n][m];
}

float Graph::Distance(int v_id1, int v_id2, bool is_head1, bool is_head2, bool is_tail1, bool is_tail2){
    if (v_id1 == -2 && !is_tail2) {
        return 0.0f;
    }
    if (v_id2 == -2 && !is_tail1) {
        return 0.0f;
    }
    if (v_id1 == -1 && !is_head2) {
        return 0.0f;
    }
    if (v_id2 == -1 && !is_head1) {
        return 0.0f;
    }
    
    if (v_id1 == -1 || v_id1 == -2 || v_id2 == -1 || v_id2 == -2){
        return 1e6;
    }
    PclPoint &v1 = vertices_->at(v_id1);
    PclPoint &v2 = vertices_->at(v_id2);
    float delta_x = v1.x - v2.x;
    float delta_y = v1.y - v2.y;
    float dist = sqrt(delta_x*delta_x + delta_y*delta_y);
    if (dist <= 10.0) {
        return dist;
    }
    else{
        return 1e6;
    }
}

void Graph::drawShortestPathInterpolationFor(vector<int> &query_path){
    if (vertices_ == NULL){
        return;
    }
    vector<int> result_path;
    shortestPathInterpolation(query_path, result_path);
    insertPathToDraw(result_path);
}

void Graph::insertPathToDraw(vector<int> &path){
    //paths_to_draw_.clear();
    paths_to_draw_.push_back(path);
}

void Graph::clearPathsToDraw(){
    paths_to_draw_.clear();
}

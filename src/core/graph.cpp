#include "graph.h"
#include <ctime>
#include <Eigen/Dense>
using namespace Eigen;

static const float Z_GRAPH = -0.3f;
static const float Z_PATH = -0.2f;
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

void Graph::updateGraphUsingSamplesAndSegments(PclPointCloud::Ptr &samples, PclSearchTree::Ptr &sample_search_tree, vector<Segment> &segments, PclPointCloud::Ptr &gps_pointcloud, PclSearchTree::Ptr &gps_pointcloud_search_tree){
    // Reset point clouds
    vertices_ = samples;
    tree_ = sample_search_tree;
    
    gps_pointcloud_ = gps_pointcloud;
    gps_pointcloud_search_tree_ = gps_pointcloud_search_tree;
    
    edge_idxs_.clear();
    int num_nodes = samples->size();
   
    // Interpolate segments using current graph.
    clock_t begin = clock();
    vector<vector<int>> interpolated_segments(segments.size());
    for (size_t seg_id = 0; seg_id < segments.size(); ++seg_id) {
        if (seg_id % 10000 == 0)
            printf("Now at segment %lu\n", seg_id);
        Segment &seg = segments[seg_id];
        vector<int> seg_pt_idxs;
        for (size_t i = 0; i < seg.points().size(); ++i) {
            SegPoint &pt = seg.points()[i];
            seg_pt_idxs.push_back(pt.orig_idx);
        }
        vector<int> &result_path = interpolated_segments[seg_id];
        shortestPathInterpolation(seg_pt_idxs, result_path);
    }
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("time elapsed: %.1f sec.\n", elapsed_secs);
    
    graph_t *new_g = new graph_t(num_nodes);
    
    locations_.clear();
    locations_.resize(num_nodes);
    for (size_t pt_idx = 0; pt_idx < samples->size(); ++pt_idx) {
        PclPoint &pt = samples->at(pt_idx);
        locations_[pt_idx].x = pt.x;
        locations_[pt_idx].y = pt.y;
    }
    
    typedef graph_t::vertex_descriptor vertex;
    typedef graph_t::edge_descriptor edge_descriptor;
    WeightMap weightmap = get(edge_weight, *new_g);
    for (size_t sample_idx = 0; sample_idx < samples->size(); ++sample_idx) {
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->nearestKSearch(vertices_->at(sample_idx), 10, k_indices, k_distances);
        
        // Get nearby segments for sample_idx
        vector<int> from_gps_point;
        vector<float> from_gps_distance;
        gps_pointcloud_search_tree_->radiusSearch(vertices_->at(sample_idx), 10, from_gps_point, from_gps_distance);
        
        for (size_t neighbor_idx = 0; neighbor_idx < k_indices.size(); ++neighbor_idx) {
            size_t to_idx = static_cast<size_t>(k_indices[neighbor_idx]);
            if (to_idx == sample_idx) {
                continue;
            }
            vector<int> edge_path;
            shortestPath(sample_idx, to_idx, edge_path);
            
            // Compute distance between edge_path and source segments
            float source_compatibility = 1e6;
            for (size_t i = 0; i < from_gps_point.size(); ++i){
                int seg_id = from_gps_point[i];
                float dist = DTWDistance(edge_path, interpolated_segments[seg_id]);
                if (dist < source_compatibility) {
                    source_compatibility = dist;
                }
            }
           
            vector<int> to_gps_point;
            vector<float> to_gps_distance;
            gps_pointcloud_search_tree_->radiusSearch(vertices_->at(to_idx), 10, to_gps_point, to_gps_distance);
            
            // Compute distance between edge_path and dst segments
            float dst_compatibility = 1e6;
            for (size_t i = 0; i < to_gps_point.size(); ++i){
                int seg_id = to_gps_point[i];
                float dist = DTWDistance(edge_path, interpolated_segments[seg_id]);
                if (dist < dst_compatibility) {
                    dst_compatibility = dist;
                }
            }
            
            if(source_compatibility < 50.0f && dst_compatibility < 50.0f){
                // Add an edge
                edge_descriptor e;
                bool inserted;
                tie(e, inserted) = add_edge(sample_idx, to_idx, *g_);
                weightmap[e] = sqrt(k_distances[neighbor_idx]);
                edge_idxs_.push_back(sample_idx);
                edge_idxs_.push_back(to_idx);
            }
        }
    }
    
    delete g_;
    g_ = new_g;
}

void Graph::shortestPath(int start, int goal, vector<int> &path){
    if (vertices_->empty()) {
        printf("Warning: cannot search in empty graph!\n");
        return;
    }
    
    if (g_ == NULL){
        printf("Warning: cannot search in empty graph!\n");
        return;
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
        return;
    }
    
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
    vector<int> nearby_vertex_idxs(query_path.size(), -1);
    for(size_t idx = 0; idx < query_path.size(); ++idx){
        PclPoint &pt = gps_pointcloud_->at(query_path[idx]);
        vector<int> k_indices;
        vector<float> k_sqr_distances;
        tree_->nearestKSearch(pt, 1, k_indices, k_sqr_distances);
        if (!k_indices.empty()) {
            nearby_vertex_idxs[idx] = k_indices[0];
        }
    }
    
    // Compute shortest path between adjascent vertices.
    result_path.clear();
    for (size_t idx = 0; idx < nearby_vertex_idxs.size() - 1; ++idx) {
        int start = nearby_vertex_idxs[idx];
        int goal = nearby_vertex_idxs[idx+1];
        if (start != goal) {
            vector<int> path;
            shortestPath(start, goal, path);
            if (path.empty()) {
                // Insert start to result_path
                result_path.push_back(start);
            }else{
                // Insert the whole path except the last point into result
                for (size_t path_pt_idx = 0; path_pt_idx < path.size() - 1; ++path_pt_idx) {
                    result_path.push_back(path[path_pt_idx]);
                }
            }
        }
    }
    result_path.push_back(nearby_vertex_idxs.back());
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

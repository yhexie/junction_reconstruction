#include "distance_graph.h"
#include <ctime>
#include <Eigen/Dense>
using namespace Eigen;

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

DistanceGraph::DistanceGraph(){
    g_ = NULL;
    computed_distances.clear();
}

DistanceGraph::~DistanceGraph(){
    if(g_ != NULL)
        delete g_;
}

void DistanceGraph::computeGraphUsingGpsPointCloud(PclPointCloud::Ptr &gps_pointcloud, PclSearchTree::Ptr &gps_pointcloud_search_tree, PclPointCloud::Ptr &vertices, PclSearchTree::Ptr &vertice_tree, float grid_size){
    grid_size_ = grid_size;
    gps_pointcloud_ = gps_pointcloud;
    gps_pointcloud_search_tree_ = gps_pointcloud_search_tree;
    vertices_ = vertices;
    tree_ = vertice_tree;
    
    // Build distance graph
    edge_idxs_.clear();
    int num_nodes = vertices_->size();
    g_ = new graph_t(num_nodes);
    locations_.clear();
    locations_.resize(num_nodes);
    for (size_t pt_idx = 0; pt_idx < vertices_->size(); ++pt_idx) {
        PclPoint &pt = vertices_->at(pt_idx);
        locations_[pt_idx].x = pt.x;
        locations_[pt_idx].y = pt.y;
    }
    
    typedef graph_t::vertex_descriptor vertex;
    typedef graph_t::edge_descriptor edge_descriptor;
    WeightMap weightmap = get(edge_weight, *g_);
    for (size_t sample_idx = 0; sample_idx < vertices_->size(); ++sample_idx) {
        vector<int> k_indices;
        vector<float> k_distances;
        tree_->nearestKSearch(vertices_->at(sample_idx), 8, k_indices, k_distances);
       
        for (size_t neighbor_idx = 1; neighbor_idx < k_indices.size(); ++neighbor_idx) {
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
            
            gps_pointcloud_search_tree_->radiusSearch(vertices_->at(sample_idx), edge_length, nearby_pt_idxs, nearby_pt_dists);
           
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
            
            float pt_density = 1e6; // points per meter
            for (int i = 0; i < n_knots; ++i) {
                if (pt_density > density[i]) {
                    pt_density = density[i];
                }
            }
            
            if (to_idx > sample_idx && pt_density > 1e-3) {
                edge_descriptor e1;
                bool inserted1;
                tie(e1, inserted1) = add_edge(sample_idx, to_idx, *g_);
                weightmap[e1] = edge_length;
                edge_idxs_.push_back(sample_idx);
                edge_idxs_.push_back(to_idx);
                edge_idxs_.push_back(to_idx);
                edge_idxs_.push_back(sample_idx);
                
                edge_descriptor e2;
                bool inserted2;
                tie(e2, inserted2) = add_edge(to_idx, sample_idx, *g_);
                weightmap[e2] = edge_length;
            }
        }
    }
    
    printf("Graph: %d nodes, %d edges\n", nVertices(), nEdges());
}

void DistanceGraph::shortestPath(int start, int goal, vector<int> &path){
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

float DistanceGraph::Distance(PclPoint &pt1, PclPoint &pt2){
    vector<int> k_indices;
    vector<float> k_distances;
    tree_->nearestKSearch(pt1, 1, k_indices, k_distances);
    if (k_indices.size() == 0) {
        return POSITIVE_INFINITY;
    }
    int idx1 = k_indices[0];
    tree_->nearestKSearch(pt2, 1, k_indices, k_distances);
    if (k_indices.size() == 0) {
        return POSITIVE_INFINITY;
    }
    int idx2 = k_indices[0];
    
    key_type this_key(idx1, idx2);
    key_type reverse_key(idx2, idx1);
    
    float dist = POSITIVE_INFINITY;
    map<key_type, float>::iterator m_it = computed_distances.find(this_key);
    if (m_it != computed_distances.end()) {
        dist = m_it->second;
        return dist;
    }
    
    // Find shortest path
    vector<int> path;
    shortestPath(idx1, idx2, path);
    if (path.size() != 0){
        // Add the path distance
        float tmp_dist = 0.0f;
        for (size_t pt_idx = 0; pt_idx < path.size()-1; ++pt_idx) {
            PclPoint &src_pt = vertices_->at(path[pt_idx]);
            PclPoint &dst_pt = vertices_->at(path[pt_idx+1]);
            float delta_x = dst_pt.x - src_pt.x;
            float delta_y = dst_pt.y - src_pt.y;
            tmp_dist += sqrt(delta_x*delta_x + delta_y*delta_y);
        }
        dist = tmp_dist;
    }
    
    computed_distances[this_key] = dist;
    computed_distances[reverse_key] = dist;
    
    return dist;
}
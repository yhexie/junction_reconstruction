#ifndef TRAJECTORIES_H_
#define TRAJECTORIES_H_

#include <QObject>
#include <set>
#include "common.h"
#include "color_map.h"
#include "renderable.h"

using namespace std;

class Graph;

class Trajectories : public Renderable
{
public:
    explicit Trajectories(QObject *parent);
    ~Trajectories();
    
    // IO
    bool load(const string &filename);
    bool save(const string &filename);
    
    bool extractFromFiles(const QStringList &filenames, QVector4D bound_box, PclSearchTree::Ptr &map_search_tree, int min_inside_pts = 2);
    bool extractTrajectoriesFromRegion(QVector4D bound_box, Trajectories *container, PclSearchTree::Ptr &map_search_tree, int min_inside_pts);
    bool isQualifiedTrajectory(vector<PclPoint> &trajectory);
    bool insertNewTrajectory(vector<PclPoint> &pt_container);
    
    // Rendering
    void prepareForVisualization();
    void draw();
    const vector<Vertex>& normalized_vertices(void) const {return normalized_vertices_;}
    vector<Color>&  vertex_colors(void) { return vertex_colors_const_; }
    void toggleRenderMode() { render_mode_ = !render_mode_;}
    void setShowDirection(bool  val) { show_direction_ = val; }
    
    // Point Cloud
    PclPointCloud::Ptr& data(void) {return data_;}
    PclSearchTree::Ptr& tree(void) {return tree_;}

    const float& minSpeed() const { return min_speed_; }
    const float& avgSpeed() const { return avg_speed_; }
    const float& maxSpeed() const { return max_speed_; }

    bool isEmpty(); // Check if the data point cloud is empty.
    const size_t getNumPoint() const {return data_->size();}
    
    // Trajectories
    const vector<vector<int> >& trajectories(void) const {return trajectories_;} // Trajectories in index format
    const size_t getNumTraj() const {return trajectories_.size();}
    
    // Sampling
    PclPointCloud::Ptr& samples(void) {return samples_;}
    PclSearchTree::Ptr& sample_tree(void) {return sample_tree_;}
    void samplePointCloud(float neighborhood_size, PclPointCloud::Ptr &, PclSearchTree::Ptr &);
    void samplePointCloud(float neighborhood_size);
    void pickASampleNear(float x, float y, float max_distance = 25.0f);
    void pickAnotherSampleNear(float x, float y, float max_distance = 25.0f);
    set<int> &picked_sample_idxs(void) {return picked_sample_idxs_;}
    void clearPickedSamples(void);
    void setShowSamples(bool val){show_samples_ = val;}
    
    void singleDirectionSamplePointCloud(float neighborhood_size); // angle is measured in degree with x-axis
    
    // Selection
    void selectNearLocation(float x, float y, float max_distance = 25.0f);
    void setSelectionMode(bool mode) {selection_mode_ = mode;}
    void setSelectedTrajectories(vector<int> &);
    
    // Clear Data
    void clearData(void);
    
    // Distance Functions
    
protected:
    // IO
    bool loadTXT(const string &filename);
    bool loadPBF(const string &filename);
    bool savePBF(const string &filename);
    
    // Rendering
    void drawSelectedTrajectories(vector<int> &traj_indices);
    
private:
    // Data
    PclPointCloud::Ptr   data_;
    PclSearchTree::Ptr   tree_;
    vector<vector<int> > trajectories_;

    // Data statistics
    float                min_speed_;
    float                max_speed_;
    float                avg_speed_;
    
    // Sample
    PclPointCloud::Ptr   samples_;
    PclSearchTree::Ptr   sample_tree_;
    vector<Vertex>       sample_locs_;
    vector<float>        sample_scales_;
    
    // Rendering
    float                scale_factor_;
    float                point_size_;
    float                line_width_;
    vector<Vertex>       normalized_vertices_;
    vector<Color>        vertex_colors_const_;
    vector<Color>        vertex_colors_individual_;
    vector<Vertex>       vertex_speed_;
    vector<Color>        vertex_speed_colors_;
    vector<int>          vertex_speed_indices_;
    vector<int>          selected_trajectories_;
    
    // Rendering Samples
    bool                 show_samples_;
    float                sample_point_size_;
    Color                sample_color_;
    vector<Vertex>       normalized_sample_locs_;
    vector<Vertex>       normalized_sample_headings_;
    vector<Color>        sample_vertex_colors_;
    vector<Color>        sample_heading_colors_;
    set<int>             picked_sample_idxs_;
    
    bool                 render_mode_;
    bool                 selection_mode_;
    bool                 show_direction_;
};

#endif // TRAJECTORIES_H_

#ifndef TRAJECTORIES_H_
#define TRAJECTORIES_H_

#include <QObject>
#include <set>
#include "common.h"
#include "color_map.h"
#include "segment.h"
#include "renderable.h"

using namespace std;

class Graph;

class Trajectories : public Renderable
{
public:
	explicit Trajectories(QObject *parent);
    ~Trajectories();
    
    void draw();
    
	PclPointCloud::Ptr& data(void) {return data_;}
	PclSearchTree::Ptr& tree(void) {return tree_;}
    
    PclPointCloud::Ptr& samples(void) {return samples_;}
    PclSearchTree::Ptr& sample_tree(void) {return sample_tree_;}
    
    int nSegments() {return segments_.size();}
    vector<Segment> &segments() {return segments_;}
    
    const vector<Vertex>& normalized_vertices(void) const {return normalized_vertices_;}
	const vector<vector<int> >& trajectories(void) const {return trajectories_;}
    const QVector4D&    BoundBox(void) const {return bound_box_;}
    
    void prepareForVisualization(QVector4D bound_box);
    
	bool load(const string &filename);
	bool save(const string &filename);
    
    bool isEmpty();
    const size_t getNumTraj() const {return trajectories_.size();}
    const size_t getNumPoint() const {return data_->size();}
    
    bool extractFromFiles(const QStringList &filenames, QVector4D bound_box, int min_inside_pts = 2);
    bool extractTrajectoriesFromRegion(QVector4D bound_box, Trajectories *container, int min_inside_pts);
    bool isQualifiedTrajectory(vector<PclPoint> &trajectory);
    bool insertNewTrajectory(vector<PclPoint> &pt_container);
    
	void setColorMode(TrajectoryColorMode color_mode);
    void toggleRenderMode() { render_mode_ = !render_mode_;}
    
    void selectNearLocation(float x, float y, float max_distance = 25.0f);
    
	string getInformation(void) const;
	TrajectoryColorMode getColorMode(void) const {return color_mode_;}
    void setSelectionMode(bool mode) {selection_mode_ = mode;}
    void setSelectedTrajectories(vector<int> &);
    void clearData(void);
    
    void computeWeightAtScale(int neighborhood_size);
    
    // Compute segments around each GPS point
    void computeSegments(float extension);
    void setShowSegments(bool val) {show_segments_ = val;}
    void selectSegmentsWithSearchRange(float range);
    void drawSegmentAt(int);
    void clearDrawnSegments();
    Segment &segmentAt(int);
    void interpolateSegmentWithGraph(Graph *g);
    void interpolateSegmentWithGraphAtSample(int sample_id, Graph *g, vector<int> &nearby_segment_idxs, vector<vector<int>> &results);
    void clusterSegmentsWithGraphAtSample(int sample_id, Graph *g);
    vector<vector<int>> &interpolatedSegments() {return graph_interpolated_segments_;}
    int clusterSegmentsUsingDistanceAtSample(int sample_id, double sigma, double threshold, int minClusterSize);
    float computeSegmentDistance(int seg_idx1, int seg_idx2, double sigma);
    float segPointDistance(int, int, Segment &, Segment &, double sigma);
    
    // Sampling GPS point cloud
    void samplePointCloud(float neighborhood_size);
    void pickAnotherSampleNear(float x, float y, float max_distance = 25.0f);
    set<int> &picked_sample_idxs(void) {return picked_sample_idxs_;}
    void clearPickedSamples(void);
    void setShowSamples(bool val){show_samples_ = val;}
    
protected:
	bool loadTXT(const string &filename);
    bool loadPBF(const string &filename);
    bool savePBF(const string &filename);
    
    void drawSelectedTrajectories(vector<int> &traj_indices);
    void updateSelectedSegments(void);
    void recomputeSegmentsForSamples(void);
    void selectSegmentsNearSample(int);
    
private:
	PclPointCloud::Ptr                data_;
	PclSearchTree::Ptr                tree_;
	vector<vector<int> >			trajectories_;
    
    // Samples
    bool                            show_samples_;
    float                           sample_point_size_;
    Color                           sample_color_;
    PclPointCloud::Ptr              samples_;
    PclSearchTree::Ptr              sample_tree_;
    vector<Vertex>                  sample_locs_;
    vector<Vertex>                  normalized_sample_locs_;
    vector<Color>                   sample_vertex_colors_;
    set<int>                        picked_sample_idxs_;
    vector<float>                   sample_weight_;
    
    // Segments
    bool                            show_segments_;
    float                           search_range_;
    vector<Segment>                 segments_;
    vector<vector<int>>             graph_interpolated_segments_;
    
    vector<int>                     segments_to_draw_;
    vector<int>                     segments_to_draw_for_samples_;
    vector<Color>                   segments_to_draw_for_samples_colors_;
    
        // Segment OpenGL
    vector<Vertex>                  segment_vertices_;
    vector<Color>                   segment_colors_;
    vector<vector<int>>             segment_idxs_;
    
    // Below are for trajectory display
    float                           scale_factor_;
    float                           point_size_;
    float                           line_width_;
    vector<Vertex>                  normalized_vertices_;
    vector<Color>                   vertex_colors_const_;
    vector<Color>                   vertex_colors_individual_;
    vector<int>                     selected_trajectories_;
	TrajectoryColorMode             color_mode_;
    bool                            render_mode_;
    bool                            selection_mode_;
};

#endif // TRAJECTORIES_H_
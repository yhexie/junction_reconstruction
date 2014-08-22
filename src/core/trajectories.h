#ifndef TRAJECTORIES_H_
#define TRAJECTORIES_H_

#include <QObject>

#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include "common.h"
#include "color_map.h"
#include "segment.h"
#include "pcl_wrapper_types.h"
#include "renderable.h"

using namespace std;

typedef RichPoint                       PclPoint;
typedef pcl::PointCloud<PclPoint>       PclPointCloud;
typedef pcl::search::Search<PclPoint>   PclSearchTree;

class Trajectories : public Renderable
{
public:
	explicit Trajectories(QObject *parent);
    ~Trajectories();
    
    void draw();
    
	PclPointCloud::Ptr& data(void) {return data_;}
	PclSearchTree::Ptr& tree(void) {return tree_;}
    
    PclPointCloud::Ptr& samples(void) {return samples_;}
    
    int nSegments() {return segments_.size();}
    
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
    
    // Compute segments around each GPS point
    void computeSegments(float extension);
    void selectSegmentsWithSearchRange(float range);
    void toggleDrawSegmentNearSelectedSamples(void) {draw_segment_near_selected_samples_ = !draw_segment_near_selected_samples_;}
    bool drawSegment(void) {return draw_segment_near_selected_samples_;}
    
    // Sampling GPS point cloud
    void samplePointCloud(float neighborhood_size);
    void pickAnotherSampleNear(float x, float y, float max_distance = 25.0f);
    void clearPickedSamples(void);
    
protected:
	bool loadTXT(const string &filename);
    bool loadPBF(const string &filename);
    bool savePBF(const string &filename);
    
    void drawSelectedTrajectories(vector<int> &traj_indices);
    void updateSelectedSegments(void);
    
private:
	PclPointCloud::Ptr                data_;
	PclSearchTree::Ptr                tree_;
	vector<vector<int> >			trajectories_;
    
    // Samples
    float                           sample_point_size_;
    Color                           sample_color_;
    
    PclPointCloud::Ptr              samples_;
    PclSearchTree::Ptr              sample_tree_;
    vector<Vertex>                  sample_locs_;
    vector<Vertex>                  normalized_sample_locs_;
    vector<Color>                   sample_vertex_colors_;
    
    vector<int>                     picked_sample_idxs_;
    
    // Segments
    vector<Segment>                 segments_;
    vector<int>                     segments_to_draw_;
    bool                            draw_segment_near_selected_samples_;
    float                           search_range_;
    vector<Vertex>                  segment_vertices_;
    vector<Color>                   segment_colors_;
    vector<vector<int>>             segment_idxs_;
    
    // Below are for trajectory display
    float                                   scale_factor_;
    float                       point_size_;
    float                       line_width_;
    vector<Vertex>         normalized_vertices_;
    vector<Color>          vertex_colors_const_;
    vector<Color>          vertex_colors_individual_;
    vector<int>            selected_trajectories_;
	TrajectoryColorMode         color_mode_;
    bool                        render_mode_;
    bool                        selection_mode_;
};

#endif // TRAJECTORIES_H_
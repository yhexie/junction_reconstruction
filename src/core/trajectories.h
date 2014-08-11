#ifndef TRAJECTORIES_H_
#define TRAJECTORIES_H_

#include <QObject>

#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include "common.h"
#include "color_map.h"
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
    
    const vector<Vertex>& normalized_vertices(void) const {return normalized_vertices_;}
	const vector<vector<int> >& trajectories(void) const {return trajectories_;}
    const QVector4D&    BoundBox(void) const {return bound_box_;}
    
    void prepareForVisualization(QVector4D bound_box);
    
	bool load(const string &filename);
	bool save(const string &filename);
    
    bool extractFromFiles(const QStringList &filenames, QVector4D bound_box, int min_inside_pts = 4);
    bool extractTrajectoriesFromRegion(QVector4D bound_box, Trajectories *container, int min_inside_pts = 4);
    bool insertNewTrajectory(vector<PclPoint> &pt_container);
    
	void setColorMode(TrajectoryColorMode color_mode);
    void toggleRenderMode() { render_mode_ = !render_mode_;}
    
    void selectNearLocation(float x, float y, float max_distance = 25.0f);
    
	string getInformation(void) const;
	TrajectoryColorMode getColorMode(void) const {return color_mode_;}
    void setSelectionMode(bool mode) {selection_mode_ = mode;}
    void clearData(void);
    
protected:
	bool loadTXT(const string &filename);
    bool loadPBF(const string &filename);
    bool savePBF(const string &filename);
    
    void drawSelectedTrajectories(vector<int> &traj_indices);
    
private:
	PclPointCloud::Ptr                data_;
	PclSearchTree::Ptr                tree_;
	vector<vector<int> >			trajectories_;
    
    // Below are for trajectory display
    QVector4D                               bound_box_; // [min_easting, max_easting, min_northing, max_northing]
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
#ifndef TRAJECTORIES_H_
#define TRAJECTORIES_H_

#include <QObject>

#include <pcl/search/search.h>
#include <pcl/common/common.h>
#include "common.h"
#include "color_map.h"
#include "pcl_wrapper_types.h"

#include "renderable.h"

typedef RichPoint                       PclPoint;
typedef pcl::PointCloud<PclPoint>       PclPointCloud;
typedef pcl::search::Search<PclPoint>   PclSearchTree;

struct Vertex{
    float x, y, z;
    Vertex(float tx, float ty, float tz) : x(tx), y(ty), z(tz){}
};



class Trajectories : public Renderable
{
public:
	explicit Trajectories(QObject *parent);
    ~Trajectories();
    
    void draw();
    
	PclPointCloud::Ptr& data(void) {return data_;}
	PclSearchTree::Ptr& tree(void) {return tree_;}
    
    const std::vector<Vertex>& normalized_vertices(void) const {return normalized_vertices_;}
	const std::vector<std::vector<unsigned> >& trajectories(void) const {return trajectories_;}
    
	bool load(const std::string& filename);
	bool save(const std::string& filename);
    
	void setColorMode(TrajectoryColorMode color_mode);
    void toggleRenderMode() { render_mode_ = !render_mode_;}
    
	std::string getInformation(void) const;
	TrajectoryColorMode getColorMode(void) const {return color_mode_;}
    
protected:
	bool loadTXT(const std::string& filename);
	//bool loadPCD(const std::string& filename);
    
private:
	PclPointCloud::Ptr                data_;
	PclSearchTree::Ptr                tree_;
	std::vector<std::vector<unsigned> >			trajectories_;
    
    // Below is for trajectory display
    Eigen::Vector3d                         center_;
    float                                   min_x_;
    float                                   min_y_;
    float                                   max_x_;
    float                                   max_y_;
    float                                   scaling_;
    float                       point_size_;
    float                       line_width_;
    std::vector<Vertex>         normalized_vertices_;
    std::vector<Color>          vertex_colors_const_;
    std::vector<Color>          vertex_colors_individual_;
	TrajectoryColorMode         color_mode_;
    bool                        render_mode_;
    
};

#endif // TRAJECTORIES_H_
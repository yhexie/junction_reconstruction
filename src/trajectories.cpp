#include <fstream>
#include <boost/filesystem.hpp>
#include "color_map.h"
#include "trajectories.h"

#include <pcl/common/centroid.h>
#include <pcl/search/impl/flann_search.hpp>

Trajectories::Trajectories(QObject *parent) : Renderable(parent), data_(new PclPointCloud), tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    center_.x() = 0.0f;
    center_.y() = 0.0f;
    min_x_ = 1e10f;
    min_y_ = 1e10f;
    max_x_ = -1e10f;
    max_y_ = -1e10f;
    point_size_ = 5.0f;
    line_width_ = 2.0f;
    render_mode_ = false;
    trajectories_.clear();
}

Trajectories::~Trajectories(){
}

void Trajectories::draw(){
    if(!render_mode_){
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
            element_buffer.allocate(&(trajectories_[i][0]), trajectories_[i].size()*sizeof(unsigned));
            glLineWidth(line_width_);
            glDrawElements(GL_LINE_STRIP, trajectories_[i].size(), GL_UNSIGNED_INT, 0);
            glPointSize(point_size_);
            glDrawElements(GL_POINTS, trajectories_[i].size(), GL_UNSIGNED_INT, 0);
        }
    }
}

bool Trajectories::loadTXT(const std::string& filename)
{
	std::ifstream fin(filename);
	if (!fin.good())
		return false;
    
	PclPoint point;
	point.setNormal(0, 0, 1);
    
	data_->clear();
	trajectories_.clear();
	int num_trajectory = 0;
	fin >> num_trajectory;
	trajectories_.resize(num_trajectory, std::vector<unsigned>());
	for (int id_trajectory = 0; id_trajectory < num_trajectory; ++ id_trajectory)
	{
		int num_samples;
		fin >> num_samples;
		std::vector<unsigned>& trajectory = trajectories_[id_trajectory];
		trajectory.resize(num_samples);
		for (int id_sample = 0; id_sample < num_samples; ++ id_sample)
		{
			double x, y, t;
			int heavy;
			fin >> x >> y >> t >> heavy;
            
            if (x < min_x_) {
                min_x_ = x;
            }
            if (x > max_x_){
                max_x_ = x;
            }
            if (y < min_y_) {
                min_y_ = y;
            }
            if (y > max_y_){
                max_y_ = y;
            }
            
			point.setCoordinate(x, y, 0.0);
			point.t = t;
			point.id_trajectory = id_trajectory;
			point.id_sample = id_sample;
            
			trajectory[id_sample] = data_->size();
			data_->push_back(point);
            center_.x() = 0.5*(min_x_ + max_x_);
            center_.y() = 0.5*(min_y_ + max_y_);
		}
	}
    
	return true;
}

bool Trajectories::load(const std::string& filename)
{
	data_->clear();
	trajectories_.clear();
	bool success = false;
	std::cout << "Loading Trajectories from " << filename << "...";
	std::string extension = boost::filesystem::path(filename).extension().string();
	if (extension == ".txt")
		success = loadTXT(filename);
	else if (extension == ".pcd"){
		//success = loadPCD(filename);
    }
	std::cout << "Done!" << std::endl;
    
	if (success)
	{
		tree_->setInputCloud(data_);
        
        float delta_x = max_x_ - min_x_;
        float delta_y = max_y_ - min_y_;
        scaling_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
        
        for (size_t i=0; i<data_->size(); ++i) {
            const PclPoint &point = data_->at(i);
            float n_x = (point.x - center_.x()) / scaling_;
            float n_y = (point.y - center_.y()) / scaling_;
            normalized_vertices_.push_back(Vertex(n_x, n_y, 0.0));
        }
        Color light_blue = ColorMap::getInstance().getNamedColor(ColorMap::LIGHT_BLUE);
        vertex_colors_const_.resize(normalized_vertices_.size(), light_blue);
        
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
	}
	
	return true;
}
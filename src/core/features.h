//
//  features.h
//  junction_reconstruction
//
//  Created by Chen Chen on 1/8/15.
//
//

#ifndef __junction_reconstruction__features__
#define __junction_reconstruction__features__

#include <stdio.h>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <cmath>
#include "renderable.h"
#include "common.h"
#include "trajectories.h"
#include "openstreetmap.h"

#include <dlib/svm_threaded.h>
#include <dlib/rand.h>

typedef dlib::matrix<double, 48, 1> query_init_sample_type;
typedef dlib::radial_basis_kernel<query_init_sample_type> query_init_rbf_kernel;
typedef dlib::one_vs_one_trainer<dlib::any_trainer<query_init_sample_type> > query_init_ovo_trainer;
typedef dlib::one_vs_one_decision_function<query_init_ovo_trainer, dlib::decision_function<query_init_rbf_kernel> > query_init_decision_function;

typedef dlib::matrix<double, 40, 1> query_q_sample_type;
typedef dlib::radial_basis_kernel<query_q_sample_type> query_q_rbf_kernel;
//typedef dlib::svm_c_trainer<query_q_rbf_kernel> query_q_trainer;
typedef dlib::svm_nu_trainer<query_q_rbf_kernel> query_q_trainer;
typedef dlib::decision_function<query_q_rbf_kernel> query_q_decision_function;

using namespace std;

enum QueryInitLabel{
    ONEWAY_ROAD = 0,
    TWOWAY_ROAD = 1,
    NON_OBVIOUS_ROAD = 2
};

enum QueryQLabel{
    R_GROW = 0,
    R_BRANCH = 1
};

void computeQueryInitFeatureAt(float radius,
                               PclPoint& point,
                               Trajectories* trajectories,
                               query_init_sample_type& feature,
                               float canonical_heading);

void trainQueryInitClassifier(vector<query_init_sample_type>& samples,
                                 vector<int>& labels,
                                 query_init_decision_function& df);

bool computeQueryQFeatureAt(float radius,
                            Trajectories* trajectories,
                            query_q_sample_type& new_feature,
                            vector<Vertex>& center_line,
                            bool is_oneway = true,
                            bool grow_backward = false);

void trainQueryQClassifier(vector<query_q_sample_type>& samples,
                              vector<int>& labels,
                              query_init_decision_function& df);

class QueryInitFeatureSelector : public Renderable{
public:
    QueryInitFeatureSelector(QObject *parent, Trajectories* trajectories = NULL);
    ~QueryInitFeatureSelector();
    
    void setTrajectories(Trajectories* new_trajectories);
    
    void computeLabelAt(float radius, PclPoint& point, int& label);
    
    void extractTrainingSamplesFromMap(float radius, OpenStreetMap *osmMap);
    bool loadTrainingSamples(const string& filename);
    bool saveTrainingSamples(const string& filename);
    
    bool trainClassifier();
    bool saveClassifier(const string& filename);
    
    int nFeatures() const { return features_.size(); }
    int nLabels() const { return labels_.size(); }
    
    vector<int>& labels() { return labels_; }
    
    void draw();
    
    void clearVisibleFeatures();
    void clear();
    
private:
    Trajectories*                       trajectories_;
    OpenStreetMap*                      osmMap_;
    
    vector<query_init_sample_type>      features_;
    vector<int>                         labels_;
    bool                                decision_function_valid_;
    query_init_decision_function        df_;
    
    // For visualization
    vector<Vertex>                      feature_vertices_;
    vector<Color>                       feature_colors_;
};

class QueryQFeatureSelector : public Renderable
{
public:
    QueryQFeatureSelector(QObject *parent, Trajectories* trajectories = NULL);
    ~QueryQFeatureSelector();
    
    void setTrajectories(Trajectories* new_trajectories);
    void addFeatureAt(vector<float> &loc, QueryQLabel type);
    
    void extractTrainingSamplesFromMap(float radius, OpenStreetMap *osmMap);
    void computeLabelAt(float radius,
                        PclPoint& point,
                        int& label,
                        bool grow_backward = false);
    bool saveTrainingSamples(const string& filename);
    bool loadTrainingSamples(const string& filename);
    
    bool trainClassifier();
    bool saveClassifier(const string& filename);
    
    int nFeatures() const { return features_.size(); }
    int nLabels() const { return labels_.size(); }
    
    void draw();
    
    void clearVisibleFeatures();
    void clear();
    
private:
    Trajectories*                       trajectories_;
    OpenStreetMap*                      osmMap_;
    
    int                                 tmp_;
    float                               ARROW_LENGTH_;
    vector<query_q_sample_type>         features_;
    vector<Vertex>                      feature_locs_;
    vector<bool>                        feature_is_front_;
    vector<int>                         labels_;
    bool                                decision_function_valid_;
    query_q_decision_function           df_;
    
    vector<Vertex>                      feature_vertices_;
    vector<Color>                       feature_colors_;
    vector<Vertex>                      feature_headings_;
    vector<Color>                       feature_heading_colors_;
};

#endif /* defined(__junction_reconstruction__features__) */
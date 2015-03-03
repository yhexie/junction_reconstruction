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
typedef dlib::matrix<double, 40, 1> query_q_sample_type;

using namespace std;

enum FeatureType{
    NONE,
    BRANCH_FEATURE,
    QUERY_INIT_FEATURE,
    QUERY_Q_FEATURE
};

enum QueryInitLabel{
    ONEWAY_ROAD = 0,
    TWOWAY_ROAD = 1,
    NON_OBVIOUS_ROAD = 2
};

enum QueryQLabel{
    R_GROW = 0,
    R_BRANCH = 1
};

void computeQueryInitFeatureAt(float radius, PclPoint& point, Trajectories* trajectories, vector<float>& feature, float canonical_heading);

void computeQueryQFeatureAt(PclPoint& point, Trajectories* trajectories, vector<float>& feature, float heading, bool is_oneway = true, bool is_reverse_dir = false);

void computeQueryQFeatureAt(PclPoint& point, Trajectories* trajectories, vector<float>& feature, set<int>& allowable_trajs, map<int, float>& allowable_traj_min_ts, map<int, float>& allowable_traj_max_ts, vector<int>& tmp, float heading, bool is_oneway = true, bool is_reverse_dir = false);

class QueryInitFeatureSelector : public Renderable{
public:
    QueryInitFeatureSelector(QObject *parent, Trajectories* trajectories = NULL);
    ~QueryInitFeatureSelector();
    
    void setTrajectories(Trajectories* new_trajectories);
    
    void computeQueryInitLabelAt(float radius, PclPoint& point, int& label);
    
    void computeQueryInitFeaturesFromMap(float radius, OpenStreetMap *osmMap);
    
    bool save(const string& filename);
    bool exportFeatures(float radius, const string& filename);
    bool loadPrediction(const string& filename);
    
    int nCurrentFeatures() const { return features_.size(); }
    int nYes() const { return n_yes_; }
    int nNo() const { return features_.size() - n_yes_; }
    
    vector<int>& labels() { return labels_; }
    
    void draw();
    
    void clearVisibleFeatures();
    void clear();
    
private:
    Trajectories*           trajectories_;
    OpenStreetMap*          osmMap_;
    vector<vector<float>>   features_; //heading distribution + distance distribution
    vector<int>             labels_;
    int                     n_yes_;
    
    // For visualization
    vector<Vertex>          feature_vertices_;
    vector<Color>           feature_colors_;
};

class QueryQFeatureSelector : public Renderable
{
public:
    QueryQFeatureSelector(QObject *parent, Trajectories* trajectories = NULL);
    ~QueryQFeatureSelector();
    
    void setTrajectories(Trajectories* new_trajectories);
    void addFeatureAt(vector<float> &loc, QueryQLabel type);
    
    int nCurrentFeatures(void) const { return features_.size(); }
    
    void computeQueryQFeaturesFromMap(OpenStreetMap *osmMap);
    void computeQueryQLabelAt(PclPoint& point, QueryQLabel& label, bool is_reverse_dir = false);
    
    bool save(const string& filename);
    bool exportFeatures(const string& filename);
    bool loadPrediction(const string& filename);
    
    void draw();
    
    void clearVisibleFeatures();
    void clear();
    
private:
    Trajectories*           trajectories_;
    OpenStreetMap*          osmMap_;
    
    vector<vector<float>>   features_;
    vector<QueryQLabel>     labels_;
    
    int                     tmp_;
    
    vector<Vertex>          feature_vertices_;
    vector<Color>           feature_colors_;
};




#endif /* defined(__junction_reconstruction__features__) */

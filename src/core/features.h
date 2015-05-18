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

using namespace std;

float branchFitting(const vector<RoadPt>&                centerline,
                    PclPointCloud::Ptr&                  points,
                    PclSearchTree::Ptr&                  search_tree,
                    std::shared_ptr<Trajectories>&       trajectories,
                    vector<vector<RoadPt>>&              branches,
                    bool                                 grow_backward = false);

bool branchPrediction(const vector<RoadPt>&                centerline,
                      set<int>&                            candidate_set,
                      std::shared_ptr<Trajectories>&       trajectories,
                      RoadPt&                              junction_loc,
                      vector<vector<RoadPt>>&              branches,
                      bool                                 grow_backward = false);

#endif /* defined(__junction_reconstruction__features__) */

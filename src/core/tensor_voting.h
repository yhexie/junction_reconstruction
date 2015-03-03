//
//  tensor_voting.h
//  junction_reconstruction
//
//  Created by Chen Chen on 2/20/15.
//
//

#ifndef __junction_reconstruction__tensor_voting__
#define __junction_reconstruction__tensor_voting__

#include <stdio.h>
#include "common.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void tensor_decomposition(const Matrix2d& T,
                          Vector2d& e1,
                          Vector2d& e2,
                          double& lambda1,
                          double& lambda2);

/*
 Compute unit stick voting.
    Args:
        - dir: normalized normal direction (e1 of the corresponding tensor).
        - v: a vector from voter to receiver.
        - sigma: neighborhood size.
        - T: the resulting stick tensor vote.
 */
void compute_unit_stick_vote(const Vector2d& dir,
                             const Vector2d& v,
                             float sigma,
                             Matrix2d& T);

/*
 Compute unit ball voting.
 Args:
 - v: a vector from voter to receiver.
 - sigma: neighborhood size.
 - T: the resulting stick tensor vote.
 */
void compute_unit_ball_vote(const Vector2d& v,
                            float sigma,
                            Matrix2d& T);

#endif /* defined(__junction_reconstruction__tensor_voting__) */

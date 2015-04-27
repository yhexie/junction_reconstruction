//
//  tensor_voting.cpp
//  junction_reconstruction
//
//  Created by Chen Chen on 2/20/15.
//
//

#include "tensor_voting.h"

/*

void tensor_decomposition(const Matrix2d& T,
                          Vector2d& e1,
                          Vector2d& e2,
                          double& lambda1,
                          double& lambda2){
    double k11 = T(0, 0);
    double k12 = T(0, 1);
    double k22 = T(1, 1);
    
    double t = (k11 + k22) / 2;
    double a = k11 - t;
    double b = k12;
    
    double ab2 = sqrt(a*a + b*b);
    double l1 = t + ab2;
    double l2 = t - ab2;
    
    double y = ab2 - a;
    double x = b;
    double r = sqrt(x*x + y*y);
    
    double theta = 0.0f;
    if (r > 1e-5) {
        theta = acos(x / r);
    }
    else{
        lambda1 = 0.0f;
        lambda2 = 0.0f;
    }
    
    if (abs(l1) > abs(l2)) {
        e1[0] = cos(theta);
        e1[1] = sin(theta);
        e2[0] = -sin(theta);
        e2[1] = cos(theta);
        lambda1 = l1;
        lambda2 = l2;
    }
    else{
        e2[0] = cos(theta);
        e2[1] = sin(theta);
        e1[0] = -sin(theta);
        e1[1] = cos(theta);
        lambda1 = l2;
        lambda2 = l1;
    }
}

void compute_unit_stick_vote(const Vector2d& dir,
                             const Vector2d& v,
                             float sigma,
                             Matrix2d& T){
    T = Matrix2d::Zero();
    
    Matrix2d rotation;
    rotation << dir[0], dir[1],
                -dir[1], dir[0];
    double c = -16.0f * log(0.1) * (sigma - 1.0) / PI / PI;
    
    Vector2d r_v = rotation * v;
    double r_v_length = r_v.norm();
    
    if (r_v_length < 1e-5) {
        return;
    }
    
    double theta = acos(r_v[0] / r_v_length);
    if (r_v[1] < 1e-5) {
        theta = 2 * PI - theta;
    }
    
    double thetas = acos(abs(r_v[0]) / r_v_length);
    double delta_angle = thetas - PI / 4.0f + 1e-3;
    if (delta_angle > 0.0f) {
        return;
    }
    
    double s = r_v_length;
    if (sin(thetas) > 1e-5) {
        s = thetas * r_v_length / sin(thetas);
    }
    
    double kappa = 2.0f * sin(thetas) / r_v_length;
    double DF = exp(-1.0f * (s*s + c*kappa*kappa) / sigma / sigma);
    if (DF < 1e-6) {
        return;
    }
   
    Vector2d vec(-sin(2.0f * theta), cos(2.0f * theta));
    // Rotate vec back
    Vector2d tmp_vec = rotation.transpose() * vec;
    
    T = DF * tmp_vec * tmp_vec.transpose();
}

void compute_unit_ball_vote(const Vector2d& v,
                            float sigma,
                            Matrix2d& T){
    T = Matrix2d::Zero();
    
    double r = v.norm();
    double attenuation = exp(-1.0f * r * r / sigma / sigma);
    if (attenuation < 1e-6) {
        return;
    }

    if (r < 1e-5) {
        return;
    }
    
    double theta = acos(v[0] / r);
    if (v[1] < 0.0f) {
        theta = 2.0f * PI - theta;
    }
    
    Vector2d vec(cos(theta), sin(theta));
    T = attenuation * (Matrix2d::Identity() - vec * vec.transpose());
}

*/
//
//  road_generator.cpp
//  junction_reconstruction
//
//  Created by Chen Chen on 1/2/15.
//
//

#include "road_generator.h"
#include <fstream>
#include <ctime>
#include "tensor_voting.h"
#include <algorithm>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-deprecated-register"
#include <dlib/svm.h>
#pragma clang diagnostic pop

void polygonalFitting(vector<Vertex>& pts,
                      vector<Vertex>& results,
                      float& avg_dist){
    vector<Vertex> cur_polyline;
    float r;
    if(results.size() == 0){
        // Initialization using PCA
        int n_pt = pts.size();
        Eigen::MatrixXd matA(n_pt, 2);
        
        for (size_t i = 0; i < pts.size(); ++i) {
            matA(i, 0) = pts[i].x;
            matA(i, 1) = pts[i].y;
        }
        
        Eigen::Vector2d col_mean = matA.colwise().mean();
        for (size_t i = 0; i < pts.size(); ++i){
            matA.row(i) -= col_mean;
        }
        
        Eigen::Matrix2d M = matA.transpose() * matA;
        
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        Eigen::Vector2d first_direction = svd.matrixU().col(0);
        
        Eigen::VectorXd projection = matA * first_direction;
        
        float max_coeff = projection.maxCoeff();
        float min_coeff = projection.minCoeff();
        r = (max_coeff - min_coeff) * 0.5f;
        
        Eigen::Vector2d start_point = projection.minCoeff() * first_direction + col_mean;
        Eigen::Vector2d end_point = projection.maxCoeff() * first_direction + col_mean;
        
        // Initialiation
        cur_polyline.push_back(Vertex(start_point.x(), start_point.y(), 0.0f));
        cur_polyline.push_back(Vertex(end_point.x(), end_point.y(), 0.0f));
    }
    else{
        float min_x = 1e6;
        float max_x = -1e6;
        float min_y = 1e6;
        float max_y = -1e6;
        for (size_t i = 0; i < results.size(); ++i) {
            cur_polyline.push_back(Vertex(results[i].x, results[i].y, 0.0f));
            if (results[i].x < min_x) {
                min_x = results[i].x;
            }
            if (results[i].x > max_x) {
                max_x = results[i].x;
            }
            if (results[i].y < min_y) {
                min_y = results[i].y;
            }
            if (results[i].y > max_y) {
                max_y = results[i].y;
            }
        }
        float delta_x = max_x - min_x;
        float delta_y = max_y - min_y;
        r = sqrt(delta_x * delta_x + delta_y * delta_y);
    }
    
    int cur_k_vertices = cur_polyline.size();
    int MAX_K = 20;
    float LAMBDA = 0.05f;
    while(cur_k_vertices <= MAX_K){
        // Projection
        for(int s = 0; s < 3; ++s){
            vector<vector<int>> assignment;
            pointsToPolylineProjection(pts, cur_polyline, assignment);
            
            // Vertex Optimization
                // Compute gradient
            int count = 0;
            while (count < 100) {
                float cum_sqr_change = 0.0f;
                for (int i = 0; i < cur_k_vertices; ++i) {
    //                Eigen::Vector2d grad_dir;
    //                computeGnGradientAt(i, pts, cur_polyline, assignment, LAMBDA, r, grad_dir);
                    Eigen::Vector2d grad_dir1;
                    computeGnApproxiGradientAt(i, pts, cur_polyline, assignment, LAMBDA, r, grad_dir1);
                    
    //                cout << grad_dir << endl;
    //                cout << grad_dir1 << endl;
    //                cout << endl;
                    float delta_x = 0.1 * grad_dir1.x();
                    float delta_y = 0.1 * grad_dir1.y();
                    cur_polyline[i].x -= delta_x;
                    cur_polyline[i].y -= delta_y;
                    cum_sqr_change += sqrt(delta_x * delta_x + delta_y * delta_y);
                }
                if (cum_sqr_change < 0.1) {
                    cout << "Break earlier at " << count << endl;
                    break;
                }
                count ++;
            }
        }
        
        vector<vector<int>> assignment;
        pointsToPolylineProjection(pts, cur_polyline, assignment);
        
        // Add a new vertex
            // Find the biggest segment
        int max_count = -1;
        int max_idx = -1;
        for (int i = cur_k_vertices; i < assignment.size(); ++i) {
            int tmp_n = assignment[i].size();
            if (tmp_n > max_count) {
                max_count = assignment[i].size();
                max_idx = i;
            }
        }
        int insert_idx = max_idx - cur_k_vertices;
        
        vector<Vertex> new_polyline;
        for (int i = 0; i < cur_polyline.size(); ++i) {
            new_polyline.push_back(cur_polyline[i]);
            if (i == insert_idx) {
                Vertex mid_pt;
                mid_pt.x = 0.5 * (cur_polyline[i].x + cur_polyline[i+1].x);
                mid_pt.y = 0.5 * (cur_polyline[i].y + cur_polyline[i+1].y);
                new_polyline.push_back(mid_pt);
            }
        }
       
        cur_polyline.clear();
       
        for (int i = 0; i < new_polyline.size(); ++i) {
            cur_polyline.push_back(new_polyline[i]);
        }
        
        cur_k_vertices += 1;
    }
    
    results.clear();
    for (int i = 0; i < cur_polyline.size(); ++i) {
        results.push_back(cur_polyline[i]);
    }
}

void initialRoadFitting(vector<Vertex>& pts,
                        vector<Vertex>& results,
                        int max_n_vertices){
    vector<Vertex> cur_polyline;
    float r;
    if(results.size() == 0){
        // Initialization using PCA
        int n_pt = pts.size();
        Eigen::MatrixXd matA(n_pt, 2);
        
        for (size_t i = 0; i < pts.size(); ++i) {
            matA(i, 0) = pts[i].x;
            matA(i, 1) = pts[i].y;
        }
        
        Eigen::Vector2d col_mean = matA.colwise().mean();
        for (size_t i = 0; i < pts.size(); ++i){
            matA.row(i) -= col_mean;
        }
        
        Eigen::Matrix2d M = matA.transpose() * matA;
        
        Eigen::JacobiSVD<Eigen::Matrix2d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        Eigen::Vector2d first_direction = svd.matrixU().col(0);
        
        Eigen::VectorXd projection = matA * first_direction;
        
        float max_coeff = projection.maxCoeff();
        float min_coeff = projection.minCoeff();
        r = (max_coeff - min_coeff) * 0.5f;
        
        Eigen::Vector2d start_point = projection.minCoeff() * first_direction + col_mean;
        Eigen::Vector2d end_point = projection.maxCoeff() * first_direction + col_mean;
        
        // Initialiation
        cur_polyline.push_back(Vertex(start_point.x(), start_point.y(), 0.0f));
        cur_polyline.push_back(Vertex(end_point.x(), end_point.y(), 0.0f));
    }
    else{
        float min_x = 1e6;
        float max_x = -1e6;
        float min_y = 1e6;
        float max_y = -1e6;
        for (size_t i = 0; i < results.size(); ++i) {
            cur_polyline.push_back(Vertex(results[i].x, results[i].y, 0.0f));
            if (results[i].x < min_x) {
                min_x = results[i].x;
            }
            if (results[i].x > max_x) {
                max_x = results[i].x;
            }
            if (results[i].y < min_y) {
                min_y = results[i].y;
            }
            if (results[i].y > max_y) {
                max_y = results[i].y;
            }
        }
        float delta_x = max_x - min_x;
        float delta_y = max_y - min_y;
        r = sqrt(delta_x * delta_x + delta_y * delta_y);
    }
    
    int cur_k_vertices = cur_polyline.size();
    float LAMBDA = 0.05f;
    int n_inner_loop = 3;
    while(cur_k_vertices <= max_n_vertices){
        for(int i_inner = 0; i_inner < n_inner_loop; ++i_inner){
            // Projection
            vector<vector<int>> assignment;
            pointsToPolylineProjection(pts, cur_polyline, assignment);
            
            // Vertex Optimization
            // Compute gradient
            int count = 0;
            while (count < 100) {
                float cum_sqr_change = 0.0f;
                for (int i = 0; i < cur_k_vertices; ++i) {
                    //                Eigen::Vector2d grad_dir;
                    //                computeGnGradientAt(i, pts, cur_polyline, assignment, LAMBDA, r, grad_dir);
                    Eigen::Vector2d grad_dir1;
                    computeGnApproxiGradientAt(i, pts, cur_polyline, assignment, LAMBDA, r, grad_dir1);
                    grad_dir1.normalize();
                    
                    //                cout << grad_dir << endl;
                    //                cout << grad_dir1 << endl;
                    //                cout << endl;
                    float delta_x = 1.0 * grad_dir1.x();
                    float delta_y = 1.0 * grad_dir1.y();
                    cur_polyline[i].x -= delta_x;
                    cur_polyline[i].y -= delta_y;
                    cum_sqr_change += sqrt(delta_x * delta_x + delta_y * delta_y);
                }
                if (cum_sqr_change < 10.0) {
//                    cout << "Break earlier at " << count << endl;
                    break;
                }
                count ++;
            }
        }
        
        vector<vector<int>> assignment;
        pointsToPolylineProjection(pts, cur_polyline, assignment);
        
        // Add a new vertex
        // Find the biggest segment
        int max_count = -1;
        int max_idx = -1;
        for (int i = cur_k_vertices; i < assignment.size(); ++i) {
            int tmp_n = assignment[i].size();
            if (tmp_n > max_count) {
                max_count = assignment[i].size();
                max_idx = i;
            }
        }
        int insert_idx = max_idx - cur_k_vertices;
        
        vector<Vertex> new_polyline;
        for (int i = 0; i < cur_polyline.size(); ++i) {
            new_polyline.push_back(cur_polyline[i]);
            if (i == insert_idx) {
                Vertex mid_pt;
                mid_pt.x = 0.5 * (cur_polyline[i].x + cur_polyline[i+1].x);
                mid_pt.y = 0.5 * (cur_polyline[i].y + cur_polyline[i+1].y);
                new_polyline.push_back(mid_pt);
            }
        }
        
        cur_polyline.clear();
        
        for (int i = 0; i < new_polyline.size(); ++i) {
            cur_polyline.push_back(new_polyline[i]);
        }
        
        cur_k_vertices += 1;
    }
    
    results.clear();
    for (int i = 0; i < cur_polyline.size(); ++i) {
        Eigen::Vector2d forward_heading(0.0f, 0.0f);
        Eigen::Vector2d backward_heading(0.0f, 0.0f);
        if (i > 0) {
            backward_heading += Eigen::Vector2d(cur_polyline[i].x - cur_polyline[i-1].x,
                                                cur_polyline[i].y - cur_polyline[i-1].y);
            float length = backward_heading.norm();
            if (length > 1e-5) {
                backward_heading /= length;
            }
            else{
                backward_heading *= 0.0f;
            }
        }
        if (i < cur_polyline.size() - 1) {
            // Look ahead
            forward_heading += Eigen::Vector2d(cur_polyline[i+1].x - cur_polyline[i].x,
                                               cur_polyline[i+1].y - cur_polyline[i].y);
            float length = forward_heading.norm();
            if (length > 1e-5) {
                forward_heading /= length;
            }
            else{
                forward_heading *= 0.0f;
            }
        }
        
        Eigen::Vector2d vertice_heading = forward_heading + backward_heading;
        float length = vertice_heading.norm();
        if (length > 1e-5) {
            vertice_heading /= length;
        }
        else{
            vertice_heading *= 0.0f;
        }
       
        float heading = acos(vertice_heading[0]) * 180.0f / PI;
        if (vertice_heading[1] < 0) {
            heading = 360.0f - heading;
        }
        
        results.push_back(Vertex(cur_polyline[i].x, cur_polyline[i].y, heading));
    }
}

void pointsToPolylineProjection(vector<Vertex>& pts,
                                vector<Vertex>& centerline,
                                vector<vector<int>>& assignments){
    /*
     Given a center line of n points, this function devices a bunch of points into (2n+1) sets. The first n sets are vertices, and the last n-1 sets are the edges.
     */
    if (centerline.size() < 2) {
        assignments.clear();
        printf("Warning from dividePointsIntoSets: centerline has less than two points!\n");
        return;
    }
    
    int n_pts = centerline.size();
    assignments.resize(2*n_pts-1, vector<int>());
    for (int iit = 0; iit < pts.size(); ++iit) {
        float min_dist = 1e6;
        int min_dist_bin = -1;
        for (int i = 0; i < n_pts - 1; ++i) {
            // i to i+1, corresponding segment is (n + i)
            Eigen::Vector2d first_pt(centerline[i].x, centerline[i].y);
            Eigen::Vector2d second_pt(centerline[i+1].x, centerline[i+1].y);
            Eigen::Vector2d vec = second_pt - first_pt;
            Eigen::Vector2d vec1 = Eigen::Vector2d(pts[iit].x, pts[iit].y) - first_pt;
            float vec_length = vec.norm();
            if (vec_length < 0.1) {
                continue;
            }
            vec /= vec_length;
            float dot_value = vec.dot(vec1);
            if (dot_value < 0) {
                // this pt may belongs to point i
                float dist = vec1.norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i;
                }
            }
            else if (dot_value > vec_length){
                // this pt may belongs to point i+1
                Eigen::Vector2d vec2 = Eigen::Vector2d(pts[iit].x, pts[iit].y) - second_pt;
                float dist = vec2.norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + 1;
                }
            }
            else{
                // this pt may belongs to segment i->(i+1)
                float v1_length = vec1.norm();
                float dist = sqrt(v1_length*v1_length - dot_value*dot_value);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + n_pts;
                }
            }
        }
        // insert pt into corresponding set
        if (min_dist_bin != -1) {
            assignments[min_dist_bin].push_back(iit);
        }
    }
}

void pointsToPolylineProjectionWithHeading(vector<Vertex>& pts,
                                           vector<Vertex>& centerline,
                                           vector<vector<int>>& assignments){
    /*
     Given a center line of n points, this function devices a bunch of points into (2n+1) sets. The first n sets are vertices, and the last n-1 sets are the edges.
     */
    if (centerline.size() < 2) {
        assignments.clear();
        printf("Warning from dividePointsIntoSets: centerline has less than two points!\n");
        return;
    }
    
    int n_pts = centerline.size();
    assignments.resize(2*n_pts-1, vector<int>());
    for (int iit = 0; iit < pts.size(); ++iit) {
        float min_dist = 1e6;
        int min_dist_bin = -1;
        float pt_heading_in_radius = pts[iit].z / 180.0f * PI;
        Eigen::Vector2d pt_heading(cos(pt_heading_in_radius), sin(pt_heading_in_radius));
        for (int i = 0; i < n_pts - 1; ++i) {
            // i to i+1, corresponding segment is (n + i)
            Eigen::Vector2d first_pt(centerline[i].x, centerline[i].y);
            Eigen::Vector2d second_pt(centerline[i+1].x, centerline[i+1].y);
            Eigen::Vector2d vec = second_pt - first_pt;
            Eigen::Vector2d vec1 = Eigen::Vector2d(pts[iit].x, pts[iit].y) - first_pt;
            float vec_length = vec.norm();
            if (vec_length < 0.1) {
                continue;
            }
            vec /= vec_length;
            
            Eigen::Vector2d first_pt_heading(vec);
            Eigen::Vector2d second_pt_heading(vec);
            if (i > 0) {
                Eigen::Vector2d prev_seg_heading(centerline[i].x - centerline[i-1].x,
                                                 centerline[i].y - centerline[i-1].y);
                float p_length = prev_seg_heading.norm();
                if (p_length > 1e-5) {
                    prev_seg_heading /= p_length;
                }
                else{
                    prev_seg_heading *= 0.0f;
                }
                
                first_pt_heading += prev_seg_heading;
                p_length = first_pt_heading.norm();
                if (p_length > 1e-5) {
                    first_pt_heading /= p_length;
                }
                else{
                    first_pt_heading *= 0.0f;
                }
            }
            
            if (i < n_pts - 2) {
                Eigen::Vector2d nxt_seg_heading(centerline[i+1].x - centerline[i].x,
                                                centerline[i+1].y - centerline[i].y);
                float p_length = nxt_seg_heading.norm();
                if (p_length > 1e-5){
                    nxt_seg_heading /= p_length;
                }
                else{
                    nxt_seg_heading *= 0.0f;
                }
                
                second_pt_heading += nxt_seg_heading;
                p_length = second_pt_heading.norm();
                if (p_length > 1e-5){
                    second_pt_heading /= p_length;
                }
                else{
                    second_pt_heading *= 0.0f;
                }
            }
            
            float dot_value = vec.dot(vec1);
            if (dot_value < 0) {
                // this pt may belongs to point i
                float dot_value = abs(pt_heading.dot(first_pt_heading));
                float dist = vec1.norm() * (1.2 - dot_value);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i;
                }
            }
            else if (dot_value > vec_length){
                // this pt may belongs to point i+1
                Eigen::Vector2d vec2 = Eigen::Vector2d(pts[iit].x, pts[iit].y) - second_pt;
                float dot_value = abs(pt_heading.dot(second_pt_heading));
                float dist = vec2.norm() * (1.2 - dot_value);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + 1;
                }
            }
            else{
                // this pt may belongs to segment i->(i+1)
                float v1_length = vec1.norm();
                float dot_value = abs(pt_heading.dot(vec));
                float dist = sqrt(v1_length*v1_length - dot_value*dot_value) * (1.2f - dot_value);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + n_pts;
                }
            }
        }
        // insert pt into corresponding set
        if (min_dist_bin != -1) {
            assignments[min_dist_bin].push_back(iit);
        }
    }
}

float computeGnScoreAt(int idx,
                        vector<Vertex>& data,
                        vector<Vertex>& vertices, // the vertices of the poly line
                        vector<vector<int>>& assignment,
                        float lambda,
                        float r){
    float square_dist = 0.0f;
    float smoothness = 0.0f;
    float heading_term = 0.0f;
    
    // Point assigned to this vertex
    for (vector<int>::iterator it = assignment[idx].begin(); it != assignment[idx].end(); ++it) {
        float delta_x = data[*it].x - vertices[idx].x;
        float delta_y = data[*it].y - vertices[idx].y;
        square_dist += (delta_x * delta_x + delta_y * delta_y);
    }
    
    // Relevant point to segment distance
    if (idx > 0) {
        // Look back, sigma minus
        int seg_idx = idx - 1 + vertices.size();
        Eigen::Vector2d vec(vertices[idx].x - vertices[idx-1].x,
                            vertices[idx].y - vertices[idx-1].y);
        vec.normalize();
        for (vector<int>::iterator it = assignment[seg_idx].begin(); it != assignment[seg_idx].end(); ++it) {
            Eigen::Vector2d vec1(data[*it].x - vertices[idx-1].x,
                                 data[*it].y - vertices[idx-1].y);
            float dot_value = vec.dot(vec1);
            float vec1_norm = vec1.norm();
            square_dist += (vec1_norm * vec1_norm - dot_value * dot_value);
        }
    }
    if (idx < vertices.size() - 1) {
        // Look ahead, sigma plus
        int seg_idx = idx + vertices.size();
        Eigen::Vector2d vec(vertices[idx+1].x - vertices[idx].x,
                            vertices[idx+1].y - vertices[idx].y);
        vec.normalize();
        for (vector<int>::iterator it = assignment[seg_idx].begin(); it != assignment[seg_idx].end(); ++it) {
            Eigen::Vector2d vec1(data[*it].x - vertices[idx].x,
                                 data[*it].y - vertices[idx].y);
            float dot_value = vec.dot(vec1);
            float vec1_norm = vec1.norm();
            square_dist += (vec1_norm * vec1_norm - dot_value * dot_value);
        }
    }
    
    square_dist /= data.size();
//    printf("sqre dist: %.5f\n", square_dist);
    
    // Heading term
    Eigen::Vector2d forward_heading(0.0f, 0.0f);
    Eigen::Vector2d backward_heading(0.0f, 0.0f);
    if (idx > 0) {
        backward_heading += Eigen::Vector2d(vertices[idx].x - vertices[idx-1].x,
                                            vertices[idx].y - vertices[idx-1].y);
        float length = backward_heading.norm();
        if (length > 1e-5) {
            backward_heading /= length;
        }
        else{
            backward_heading *= 0.0f;
        }
    }
    if (idx < vertices.size() - 1) {
        // Look ahead
        forward_heading += Eigen::Vector2d(vertices[idx+1].x - vertices[idx].x,
                                           vertices[idx+1].y - vertices[idx].y);
        float length = forward_heading.norm();
        if (length > 1e-5) {
            forward_heading /= length;
        }
        else{
            forward_heading *= 0.0f;
        }
    }
    
    Eigen::Vector2d vertice_heading = forward_heading + backward_heading;
    float length = vertice_heading.norm();
    if (length > 1e-5) {
        vertice_heading /= length;
    }
    else{
        vertice_heading *= 0.0f;
    }
    
    for (vector<int>::iterator it = assignment[idx].begin(); it != assignment[idx].end(); ++it) {
        float pt_heading_in_radius = data[*it].z / 180.0f * PI;
        Eigen::Vector2d pt_heading(cos(pt_heading_in_radius), sin(pt_heading_in_radius));
        float dot_value = abs(pt_heading.dot(vertice_heading));
        heading_term -= dot_value;
    }
    
    if (idx > 0) {
        // Look back, sigma minus
        int seg_idx = idx - 1 + vertices.size();
        Eigen::Vector2d vec(vertices[idx].x - vertices[idx-1].x,
                            vertices[idx].y - vertices[idx-1].y);
        vec.normalize();
        for (vector<int>::iterator it = assignment[seg_idx].begin(); it != assignment[seg_idx].end(); ++it) {
            float pt_heading_in_radius = data[*it].z / 180.0f * PI;
            Eigen::Vector2d pt_heading(cos(pt_heading_in_radius), sin(pt_heading_in_radius));
            float dot_value = abs(pt_heading.dot(backward_heading));
            heading_term -= dot_value;
        }
    }
    if (idx < vertices.size() - 1) {
        // Look ahead, sigma plus
        int seg_idx = idx + vertices.size();
        Eigen::Vector2d vec(vertices[idx+1].x - vertices[idx].x,
                            vertices[idx+1].y - vertices[idx].y);
        vec.normalize();
        for (vector<int>::iterator it = assignment[seg_idx].begin(); it != assignment[seg_idx].end(); ++it) {
            float pt_heading_in_radius = data[*it].z / 180.0f * PI;
            Eigen::Vector2d pt_heading(cos(pt_heading_in_radius), sin(pt_heading_in_radius));
            float dot_value = abs(pt_heading.dot(forward_heading));
            heading_term -= dot_value;
        }
    }
    
    heading_term /= data.size();
    
    // Smoothness term
    if (idx == 0) {
        if (idx + 2 < vertices.size()) {
            Eigen::Vector2d vec1(vertices[idx+2].x - vertices[idx+1].x,
                                 vertices[idx+2].y - vertices[idx+1].y);
            Eigen::Vector2d vec2(vertices[idx].x - vertices[idx+1].x,
                                 vertices[idx].y - vertices[idx+1].y);
            vec1.normalize();
            vec2.normalize();
            float term = 1.0f + vec1.dot(vec2);
            smoothness += (r * r * term);
        }
        
        Eigen::Vector2d vec(vertices[idx+1].x - vertices[idx].x,
                            vertices[idx+1].y - vertices[idx].y);
        float length = vec.norm();
        smoothness += (length * length);
    }
    else if (idx == vertices.size() - 1){
        if (idx - 2 >= 0) {
            Eigen::Vector2d vec1(vertices[idx-2].x - vertices[idx-1].x,
                                 vertices[idx-2].y - vertices[idx-1].y);
            Eigen::Vector2d vec2(vertices[idx].x - vertices[idx-1].x,
                                 vertices[idx].y - vertices[idx-1].y);
            vec1.normalize();
            vec2.normalize();
            float term = 1.0f + vec1.dot(vec2);
            smoothness += (r * r * term);
        }
        
        Eigen::Vector2d vec(vertices[idx-1].x - vertices[idx].x,
                            vertices[idx-1].y - vertices[idx].y);
        float length = vec.norm();
        smoothness += (length * length);
    }
    else{
        Eigen::Vector2d vec1(vertices[idx-1].x - vertices[idx].x,
                             vertices[idx-1].y - vertices[idx].y);
        Eigen::Vector2d vec2(vertices[idx+1].x - vertices[idx].x,
                             vertices[idx+1].y - vertices[idx].y);
        vec1.normalize();
        vec2.normalize();
        float term = 1.0f + vec1.dot(vec2);
        smoothness += (r * r * term);
        
        if(idx + 2 < vertices.size()){
            Eigen::Vector2d vec3(vertices[idx+2].x - vertices[idx+1].x,
                                 vertices[idx+2].y - vertices[idx+1].y);
            Eigen::Vector2d vec4(vertices[idx].x - vertices[idx+1].x,
                                 vertices[idx].y - vertices[idx+1].y);
            vec3.normalize();
            vec4.normalize();
            float term1 = 1.0f + vec3.dot(vec4);
            smoothness += (r * r * term1);
        }
        
        if (idx - 2 >= 0) {
            Eigen::Vector2d vec5(vertices[idx-2].x - vertices[idx-1].x,
                                 vertices[idx-2].y - vertices[idx-1].y);
            Eigen::Vector2d vec6(vertices[idx].x - vertices[idx-1].x,
                                 vertices[idx].y - vertices[idx-1].y);
            vec5.normalize();
            vec6.normalize();
            float term2 = 1.0f + vec5.dot(vec6);
            smoothness += (r * r * term2);
        }
    }
    int k = vertices.size() - 1;
    smoothness /= (k+1);
    return (square_dist + heading_term + lambda * smoothness);
//    printf("smoothness: %.5f\n", smoothness);
}

void computeGnApproxiGradientAt(int idx,
                                vector<Vertex>& data,
                                vector<Vertex>& vertices, // the vertices of the poly line
                                vector<vector<int>>& assignment,
                                float lambda,
                                float r, // the scale of the data
                                Eigen::Vector2d& grad_dir){
    float score0 = computeGnScoreAt(idx, data, vertices, assignment, lambda, r);
    
    float dx = 1e-6 * r;
    float dy = dx;
    vector<Vertex> new_vertices1;
    for(size_t i = 0; i < vertices.size(); ++i){
        new_vertices1.push_back(Vertex(vertices[i].x, vertices[i].y, vertices[i].z));
    }
    new_vertices1[idx].x += dx;
    float score1 = computeGnScoreAt(idx, data, new_vertices1, assignment, lambda, r);
    
    float delta_gn_x = score1 - score0;
    float dg_dx = delta_gn_x/ dx;
    
    vector<Vertex> new_vertices2;
    for(size_t i = 0; i < vertices.size(); ++i){
        new_vertices2.push_back(Vertex(vertices[i].x, vertices[i].y, vertices[i].z));
    }
    new_vertices2[idx].y += dy;
   
    float score2 = computeGnScoreAt(idx, data, new_vertices2, assignment, lambda, r);
    float delta_gn_y = score2 - score0;
    
//    cout << "score 0: " << score_0 <<endl;
//    cout << "score 1: " << score_1 <<endl;
//    cout << "score 2: " << score_2 <<endl;
//    cout << endl;
    
    float dg_dy = delta_gn_y / dy;
    
    grad_dir[0] = dg_dx;
    grad_dir[1] = dg_dy;
}

void computeGnGradientAt(int idx,
                         vector<Vertex>& data,
                         vector<Vertex>& vertices,
                         vector<vector<int>>& assignment,
                         float lambda,
                         float R,
                         Eigen::Vector2d& grad_dir){
    int n = data.size();
    int k = vertices.size() - 1;
   
    Eigen::Vector2d d_sigma_plus(0.0f, 0.0f);
    Eigen::Vector2d d_sigma_minus(0.0f, 0.0f);
    Eigen::Vector2d d_vvi(0.0f, 0.0f);
    
    Eigen::Vector2d vec_forward;
    Eigen::Vector2d vec_backward;
    
    if(idx < k){
        vec_forward[0] = vertices[idx].x - vertices[idx+1].x;
        vec_forward[1] = vertices[idx].y - vertices[idx+1].y;
    }
    
    if(idx > 0){
        vec_backward[0] = vertices[idx].x - vertices[idx-1].x;
        vec_backward[1] = vertices[idx].y - vertices[idx-1].y;
    }
    
    float vec_forward_norm = vec_forward.norm();
    float vec_backward_norm = vec_backward.norm();
    
    if(idx < k && idx != 0){
        // Compute d_sigma_plus
        int sigma_plus_idx = idx + k + 1;
        vector<int>& Si = assignment[sigma_plus_idx];
        for (vector<int>::iterator it = Si.begin(); it != Si.end(); ++it) {
            Eigen::Vector2d vec(vertices[idx].x - data[*it].x,
                                vertices[idx].y - data[*it].y);
            float quant = -1.0f * vec[0] * vec_forward[1] + vec[1] * vec_forward[0];
            float tmp_x = 2.0f * quant * (vertices[idx+1].y - data[*it].y) / pow(vec_forward_norm, 2) - 2.0f * quant * quant * vec_forward[0] / pow(vec_forward_norm, 4);
            float tmp_y = 2.0f * quant * (data[*it].x - vertices[idx+1].x) / pow(vec_forward_norm, 2) - 2.0f * pow(quant, 2) * vec_forward[1] / pow(vec_forward_norm, 4);
            d_sigma_plus += Eigen::Vector2d(tmp_x, tmp_y);
        }
    }
    
    if(idx > 0 && idx != k){
        // Compute d_sigma_minus
        int sigma_minus_idx = idx + k;
        vector<int>& Si_minus_1 = assignment[sigma_minus_idx];
        for (vector<int>::iterator it = Si_minus_1.begin(); it != Si_minus_1.end(); ++it) {
            Eigen::Vector2d vec(vertices[idx].x - data[*it].x,
                                vertices[idx].y - data[*it].y);
            float quant = -1.0f * vec[0] * vec_backward[1] + vec[1] * vec_backward[0];
            float tmp_x = 2.0f * quant * (vertices[idx-1].y - data[*it].y) / pow(vec_backward_norm, 2) - 2.0f * pow(quant, 2) * vec_backward[0] / pow(vec_backward_norm, 4);
            float tmp_y = 2.0f * quant * (data[*it].x - vertices[idx-1].x) / pow(vec_backward_norm, 2) - 2.0f * pow(quant, 2) * vec_backward[1] / pow(vec_backward_norm, 4);
            d_sigma_minus += Eigen::Vector2d(tmp_x, tmp_y);
        }
    }
    
    vector<int>& Vi = assignment[idx];
    for (vector<int>::iterator it = Vi.begin(); it != Vi.end(); ++it) {
        d_vvi += 2.0f * Eigen::Vector2d(vertices[idx].x - data[*it].x,
                                        vertices[idx].y - data[*it].y);
    }
    
    Eigen::Vector2d d_dn_f = d_sigma_plus + d_sigma_minus + d_vvi;
    d_dn_f /= n;
    
    // Compute d_pv
    Eigen::Vector2d d_pv_vi_minus_1(0.0f, 0.0f);
    Eigen::Vector2d d_pv_vi(0.0f, 0.0f);
    Eigen::Vector2d d_pv_vi_plus_1(0.0f, 0.0f);
    
    if (idx < k) {
        // Compute d_pv_vi_plus_1
        computeDpPlusOneAt(idx, vertices, R, d_pv_vi_plus_1);
    }
    
    if (idx > 0){
        // Compute d_pv_vi_minus_1
        computeDpMinusOneAt(idx, vertices, R, d_pv_vi_minus_1);
    }
    
    computeDpAt(idx, vertices, R, d_pv_vi);
    
    Eigen::Vector2d d_pv = d_pv_vi + d_pv_vi_plus_1 + d_pv_vi_minus_1;
    d_pv /= (k+1);
    
//    if (idx == 0 || idx == k) {
//        d_pv *= 0.0f;
//    }
    
    grad_dir = d_dn_f + lambda * d_pv;
}

void computeDpPlusOneAt(int idx,
                        vector<Vertex>& vertices,
                        float R,
                        Eigen::Vector2d& d_pv_vi_plus_1){
    int idx_plus_one = idx + 1;
    
    if(idx_plus_one == vertices.size() - 1){
        d_pv_vi_plus_1 = 2.0f * Eigen::Vector2d(vertices[idx].x - vertices[idx+1].x,
                                                vertices[idx].y - vertices[idx+1].y);
        return;
    }
    
    Eigen::Vector2d vec2(vertices[idx+2].x - vertices[idx+1].x,
                         vertices[idx+2].y - vertices[idx+1].y);
    Eigen::Vector2d vec1(vertices[idx].x - vertices[idx+1].x,
                         vertices[idx].y - vertices[idx+1].y);
    
    float DOT = vec1.dot(vec2);
    float vec1_norm = vec1.norm();
    float vec2_norm = vec2.norm();
    
    d_pv_vi_plus_1 = Eigen::Vector2d(vertices[idx+2].x - vertices[idx+1].x,
                                     vertices[idx+2].y - vertices[idx+1].y);
    d_pv_vi_plus_1 -= (DOT * vec1 / vec1_norm / vec1_norm);
    
    d_pv_vi_plus_1 *= (R * R / vec1_norm / vec2_norm);
}

void computeDpMinusOneAt(int idx,
                         vector<Vertex>& vertices,
                         float R,
                         Eigen::Vector2d& d_pv_vi_minus_1){
    int idx_minus_one = idx - 1;
    
    if(idx_minus_one == 0){
        d_pv_vi_minus_1 = 2.0f * Eigen::Vector2d(vertices[idx].x - vertices[idx-1].x,
                                                vertices[idx].y - vertices[idx-1].y);
        return;
    }
    
    Eigen::Vector2d vec2(vertices[idx].x - vertices[idx-1].x,
                         vertices[idx].y - vertices[idx-1].y);
    Eigen::Vector2d vec1(vertices[idx-2].x - vertices[idx-1].x,
                         vertices[idx-2].y - vertices[idx-1].y);
    
    float DOT = vec1.dot(vec2);
    float vec1_norm = vec1.norm();
    float vec2_norm = vec2.norm();
    
    d_pv_vi_minus_1 = Eigen::Vector2d(vertices[idx-2].x - vertices[idx-1].x,
                                      vertices[idx-2].y - vertices[idx-1].y);
    d_pv_vi_minus_1 -= (DOT * vec2 / vec2_norm / vec2_norm);
    
    d_pv_vi_minus_1 *= (R * R / vec1_norm / vec2_norm);
}

void computeDpAt(int idx,
                 vector<Vertex>& vertices,
                 float R,
                 Eigen::Vector2d& d_pv_vi){
    if (idx == 0) {
        d_pv_vi = 2.0f * Eigen::Vector2d(vertices[idx].x - vertices[idx+1].x,
                                         vertices[idx].y - vertices[idx+1].y);
        return;
    }
    
    if (idx == vertices.size() - 1) {
        d_pv_vi = 2.0f * Eigen::Vector2d(vertices[idx].x - vertices[idx-1].x,
                                         vertices[idx].y - vertices[idx-1].y);
        return;
    }
    
    Eigen::Vector2d vec_forward(vertices[idx].x - vertices[idx+1].x,
                                vertices[idx].y - vertices[idx+1].y);
    Eigen::Vector2d vec_backward(vertices[idx].x - vertices[idx-1].x,
                                 vertices[idx].y - vertices[idx-1].y);
    float vec_forward_norm = vec_forward.norm();
    float vec_backward_norm = vec_backward.norm();
    
    d_pv_vi = 2.0f * Eigen::Vector2d(vertices[idx].x, vertices[idx].y)
                - Eigen::Vector2d(vertices[idx+1].x, vertices[idx+1].y)
                - Eigen::Vector2d(vertices[idx-1].x, vertices[idx-1].y)
                - vec_forward.dot(vec_backward) * (vec_forward / pow(vec_forward_norm, 2) + vec_backward / pow(vec_backward_norm, 2));
    d_pv_vi *= (R * R / vec_forward_norm / vec_backward_norm);
}

void dividePointsIntoSets(vector<Vertex>& pts,
                          vector<Vertex>& centerline,
                          vector<vector<int>>& assignments,
                          map<int, float>& dists){
    /*
     Given a center line of n points, this function devices a bunch of points into (2n+1) sets. The first n sets are vertices, and the last n-1 sets are the edges.
     */
    if (centerline.size() < 2) {
        assignments.clear();
        dists.clear();
        printf("Warning from dividePointsIntoSets: centerline has less than two points!\n");
        return;
    }
    
    int n_pts = centerline.size();
    assignments.resize(2*n_pts+1, vector<int>());
    for (int iit = 0; iit < pts.size(); ++iit) {
        float min_dist = 1e6;
        int min_dist_bin = -1;
        for (int i = 0; i < n_pts - 1; ++i) {
            // i to i+1, corresponding segment is (n + i)
            Eigen::Vector2d first_pt(centerline[i].x, centerline[i].y);
            Eigen::Vector2d second_pt(centerline[i+1].x, centerline[i+1].y);
            Eigen::Vector2d vec = second_pt - first_pt;
            Eigen::Vector2d vec1 = Eigen::Vector2d(pts[iit].x, pts[iit].y) - first_pt;
            float vec_length = vec.norm();
            if (vec_length < 0.1) {
                continue;
            }
            vec /= vec_length;
            float dot_value = vec.dot(vec1);
            if (dot_value < 0) {
                // this pt may belongs to point i
                float dist = vec1.norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i;
                }
            }
            else if (dot_value > vec_length){
                // this pt may belongs to point i+1
                Eigen::Vector2d vec2 = Eigen::Vector2d(pts[iit].x, pts[iit].y) - second_pt;
                float dist = vec2.norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + 1;
                }
            }
            else{
                // this pt may belongs to segment i->(i+1)
                float dist = sqrt(vec_length*vec_length - dot_value*dot_value);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + n_pts;
                }
            }
        }
        // insert pt into corresponding set
        if (min_dist_bin != -1) {
            assignments[min_dist_bin].push_back(iit);
            dists[iit] = min_dist;
        }
    }
}

bool dividePointsIntoSets(set<int>& pts,
                          PclPointCloud::Ptr data,
                          vector<Vertex>& centerline,
                          vector<vector<int>>& assignments,
                          map<int, float>& dists,
                          bool with_checking,
                          bool is_oneway,
                          float dist_threshold,
                          float heading_threshold,
                          int tolerance){
    /*
     Given a center line of n points, this function devices a bunch of points into (2n+1) sets. The first n sets are vertices, and the last n-1 sets are the edges.
     */
    if (centerline.size() < 2) {
        assignments.clear();
        dists.clear();
        printf("Warning from dividePointsIntoSets: centerline has less than two points!\n");
        return false;
    }
    
    int n_pts = centerline.size();
    assignments.resize(2*n_pts+1, vector<int>());
    int n_violating_pts = 0;
    for (set<int>::iterator it = pts.begin(); it != pts.end(); ++it) {
        PclPoint& pt = data->at(*it);
        float min_dist = 1e6;
        int min_dist_bin = -1;
        for (int i = 0; i < n_pts - 1; ++i) {
            // i to i+1, corresponding segment is (n + i)
            Eigen::Vector2d first_pt(centerline[i].x, centerline[i].y);
            Eigen::Vector2d second_pt(centerline[i+1].x, centerline[i+1].y);
            Eigen::Vector2d vec = second_pt - first_pt;
            Eigen::Vector2d vec1 = Eigen::Vector2d(pt.x, pt.y) - first_pt;
            float vec_length = vec.norm();
            if (vec_length < 0.1) {
                continue;
            }
            vec /= vec_length;
            float dot_value = vec.dot(vec1);
            if (dot_value < 0) {
                // this pt may belongs to point i
                float dist = vec1.norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i;
                }
            }
            else if (dot_value > vec_length){
                // this pt may belongs to point i+1
                Eigen::Vector2d vec2 = Eigen::Vector2d(pt.x, pt.y) - second_pt;
                float dist = vec2.norm();
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + 1;
                }
            }
            else{
                // this pt may belongs to segment i->(i+1)
                float dist = sqrt(vec_length*vec_length - dot_value*dot_value);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_bin = i + n_pts;
                }
            }
        }
        // insert pt into corresponding set
        if (min_dist_bin != -1) {
            assignments[min_dist_bin].push_back(*it);
            dists[*it] = min_dist;
            if (with_checking) {
                if(min_dist_bin != 0 && min_dist_bin != n_pts - 1){
                    if(min_dist > dist_threshold){
                        n_violating_pts++;
                        continue;
                    }
                    // Check heading
                    if (min_dist_bin < n_pts) {
                        float delta_heading = abs(deltaHeading1MinusHeading2(data->at(*it).head, centerline[min_dist_bin].z));
                        if(!is_oneway){
                            if(delta_heading > 90.0f){
                                delta_heading = 180.0f - delta_heading;
                            }
                        }
                        if(delta_heading > heading_threshold){
                            n_violating_pts++;
                            continue;
                        }
                    }
                    else{
                        int first_pt = min_dist_bin - n_pts;
                        int second_pt = min_dist_bin - n_pts + 1;
                        float delta_heading1 = abs(deltaHeading1MinusHeading2(data->at(*it).head, centerline[first_pt].z));
                        float delta_heading2 = abs(deltaHeading1MinusHeading2(data->at(*it).head, centerline[second_pt].z));
                        if(!is_oneway){
                            if(delta_heading1 > 90.0f){
                                delta_heading1 = 180.0f - delta_heading1;
                            }
                            if(delta_heading2 > 90.0f){
                                delta_heading2 = 180.0f - delta_heading2;
                            }
                        }
                        float delta_heading = (delta_heading1 < delta_heading2) ? delta_heading1 : delta_heading2;
                        if(delta_heading > heading_threshold){
                            n_violating_pts++;
                            continue;
                        }
                    }
                }
                if(n_violating_pts > tolerance){
                    assignments.clear();
                    dists.clear();
                    return false;
                }
            }
        }
    }
    return true;
}

RoadGenerator::RoadGenerator(QObject *parent, Trajectories* trajectories) : Renderable(parent), point_cloud_(new PclPointCloud), search_tree_(new pcl::search::FlannSearch<PclPoint>(false))
{
    trajectories_ = trajectories;
    has_been_covered_.clear();
    if(trajectories_ != NULL){
        if(trajectories_->data()->size() > 0){
            has_been_covered_.resize(trajectories_->data()->size(), false);
        }
    }
    production_string_.clear();
    current_feature_type_ = NONE;
    feature_properties_.clear();
}

RoadGenerator::~RoadGenerator(){
    if(production_string_.size() != 0){
        for (size_t i = 0; i < production_string_.size(); ++i) {
            delete production_string_[i];
        }
    }
    
    if(graph_nodes_.size() > 0){
        for (map<vertex_t, Symbol*>::iterator it = graph_nodes_.begin(); it != graph_nodes_.end(); ++it) {
            delete it->second;
        }
    }
}

void RoadGenerator::sparseVoting(float sigma){
    PclPointCloud::Ptr samples = trajectories_->samples();
    PclSearchTree::Ptr sample_tree = trajectories_->sample_tree();
    vector<Matrix2d> init_tensors(samples->size(), Matrix2d::Identity());
    tensor_votes_.clear();
    tensor_votes_.resize(samples->size(), Matrix2d::Zero());
    float search_radius = 5.0f * sigma;
    for (size_t i = 0; i < samples->size(); ++i) {
        PclPoint& pt = samples->at(i);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        sample_tree->radiusSearch(pt, search_radius, k_indices, k_dist_sqrs);
        // Decompose current tensor
        float heading_in_radius = samples->at(i).head * PI / 180.0f;
        Vector2d e1(-sin(heading_in_radius), cos(heading_in_radius));
        Vector2d dir(cos(heading_in_radius), sin(heading_in_radius));
        
        double lambda1, lambda2;
//        tensor_decomposition(init_tensors[i],
//                             e1,
//                             e2,
//                             lambda1,
//                             lambda2);
        lambda1 = 1.0f;
        lambda2 = 0.0f;
        
        for(size_t j = 0; j < k_indices.size(); ++j){
            if (j == i) {
                continue;
            }
            
            // Check heading compatibility
            float delta_heading = abs(deltaHeading1MinusHeading2(samples->at(i).head, samples->at(j).head));
            if(delta_heading > 45.0f){
                continue;
            }
            
            int nb_pt_idx = k_indices[j];
            
            Vector2d v(samples->at(nb_pt_idx).x - samples->at(i).x,
                       samples->at(nb_pt_idx).y - samples->at(i).y);
            
            if(lambda1 > lambda2){
                // Cast stick vote with magnitude (lambda1 - lambda2)
                Matrix2d Ts;
                compute_unit_stick_vote(e1,
                                        v,
                                        sigma,
                                        Ts);
                tensor_votes_[nb_pt_idx] += (lambda1 - lambda2) * Ts;
            }
            
            if(lambda2 > 0){
                // Cast ball vote with magnitude lambda2
                Matrix2d Tb;
                compute_unit_ball_vote(v,
                                       sigma,
                                       Tb);
                tensor_votes_[nb_pt_idx] += lambda2 * Tb;
            }
        }
    }
    
    return;
    // Visualization
    feature_vertices_.clear();
    feature_colors_.clear();
    float min_threshold_for_noise = 0.1f;
    for (size_t i = 0; i < samples->size(); ++i) {
        PclPoint& pt = samples->at(i);
        // Decompose current tensor
        Vector2d e1, e2;
        double lambda1, lambda2;
        tensor_decomposition(tensor_votes_[i],
                             e1,
                             e2,
                             lambda1,
                             lambda2);
        float stick_scale = lambda1 - lambda2;
        float ball_scale = lambda2;
        
        if(ball_scale < min_threshold_for_noise && stick_scale < min_threshold_for_noise){
            // This is noise
            feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
        }
        else{
            if (stick_scale > 1.05 * ball_scale) {
                // This is curve
                feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
            }
            else{
                feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
                feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::RED));
            }
        }
    }
}

void RoadGenerator::denseVoting(float sigma){
    
}

void RoadGenerator::tracing(float sigma){
    PclPointCloud::Ptr samples = trajectories_->samples();
    PclSearchTree::Ptr sample_tree = trajectories_->sample_tree();
    
    // Trace road as local maximum
    traced_curves_.clear();
    vector<double> curve_map(tensor_votes_.size(), 0.0f);
    vector<Vector2d> curve_normals(tensor_votes_.size(), Vector2d::Zero());
    vector<double> junction_map(tensor_votes_.size(), 0.0f);
   
    for(size_t i = 0; i < tensor_votes_.size(); ++i){
        // Decompose the tensor
        Vector2d e1, e2;
        double lambda1, lambda2;
        tensor_decomposition(tensor_votes_[i],
                             e1,
                             e2,
                             lambda1,
                             lambda2);
        curve_normals[i] = e1;
        curve_map[i] = lambda1 - lambda2;
        junction_map[i] = lambda2;
    }
    
    // local maximum
    vector<bool> is_curve_local_maximum(tensor_votes_.size(), false);
    float search_radius = 1.2f * sigma;
    for(size_t i = 0; i < tensor_votes_.size(); ++i){
        PclPoint& pt = samples->at(i);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        bool is_local_maximum = true;
        sample_tree->radiusSearch(pt, search_radius, k_indices, k_dist_sqrs);
        if(k_indices.size() < 2){
            is_local_maximum = false;
        }
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            if(*it == i){
                continue;
            }
            
            if (curve_map[i] < curve_map[*it]) {
                is_local_maximum = false;
                break;
            }
        }
        is_curve_local_maximum[i] = is_local_maximum;
    }
    
    // Visualization
    feature_vertices_.clear();
    feature_colors_.clear();
    for (size_t i = 0; i < samples->size(); ++i) {
        PclPoint& pt = samples->at(i);
        // This is curve
        if (is_curve_local_maximum[i]) {
            feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::GREEN));
        }
        else{
//            feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
//            feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
        }
    }
}

void RoadGenerator::applyRules(){
    float delta_growth = 10.0f;
    float search_radius = 15.0f;
    float heading_threshold = 30.0f;
    PclPoint point;
    for(map<vertex_t, Symbol*>::iterator it = graph_nodes_.begin(); it != graph_nodes_.end(); ++it){
        if (it->second->type() == ROAD) {
            RoadSymbol* road = dynamic_cast<RoadSymbol*>(it->second);
            if (road->startState() == QUERY_Q) {
                // Grow start
                Vertex& front_pt = road->center().front();
                point.setCoordinate(front_pt.x, front_pt.y, 0.0f);
                float heading = front_pt.z;
                float heading_in_radius = heading * PI / 180.0f;
                Vector3d dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                trajectories_->tree()->radiusSearch(point, search_radius, k_indices, k_dist_sqrs);
                
                float min_dot_value = 0.0f;
                float avg_vert_dist = 0.0f;
                float n_count = 0;
                Vector2d avg_heading(0.0f, 0.0f);
                for(vector<int>::iterator iit = k_indices.begin(); iit != k_indices.end(); ++iit){
                    PclPoint& nearby_pt = trajectories_->data()->at(*iit);
                    Vector3d vec(nearby_pt.x - point.x,
                                 nearby_pt.y - point.y,
                                 0.0f);
                    float dot_value = dir.dot(vec);
                    if(dot_value > 0.0f){
                        continue;
                    }
                    float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, heading));
                    if (!road->isOneway()){
                        if(delta_heading > 90.0f){
                            delta_heading = 180.0f - delta_heading;
                        }
                    }
                    
                    if(delta_heading > heading_threshold){
                        continue;
                    }
                    
                    float pt_heading_in_radius = nearby_pt.head * PI / 180.0f;
                    Vector2d pt_heading_dir(cos(pt_heading_in_radius),
                                            sin(pt_heading_in_radius));
                    Vector2d dir_2d(dir[0], dir[1]);
                    if (pt_heading_dir.dot(dir_2d) < 0) {
                        pt_heading_dir*= -1.0f;
                    }
                    
                    avg_heading += pt_heading_dir;
                    
                    n_count++;
                    if (min_dot_value > dot_value) {
                        min_dot_value = dot_value;
                    }
                    
                    float cross_value = dir.cross(vec)[2];
                    avg_vert_dist += cross_value;
                }
                if (n_count >= 2){
                    avg_vert_dist /= n_count;
                    avg_heading /= n_count;
                    avg_heading.normalize();
                    Vector2d delta(min_dot_value, avg_vert_dist);
                    delta.normalize();
                    delta *= delta_growth;
                    
                    float dx = delta[0] * cos(heading_in_radius) - delta[1] * sin(heading_in_radius);
                    float dy = delta[0] * sin(heading_in_radius) + delta[1] * cos(heading_in_radius);
                    float angle = acos(avg_heading[0]) * 180.0f / PI;
                    if(avg_heading[1] < 0.0f){
                        angle = 360.0f - angle;
                    }
                    road->center().insert(road->center().begin(), Vertex(dx + point.x,
                                                                         dy + point.y,
                                                                         angle));
                }
                else{
                    road->startState() = TERMINAL;
                }
            }
            if (road->endState() == QUERY_Q){
                // Grow end
                Vertex& back_pt = road->center().back();
                point.setCoordinate(back_pt.x, back_pt.y, 0.0f);
                float heading = back_pt.z;
                float heading_in_radius = heading * PI / 180.0f;
                Vector3d dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                trajectories_->tree()->radiusSearch(point, search_radius, k_indices, k_dist_sqrs);
                
                float max_dot_value = 0.0f;
                float avg_vert_dist = 0.0f;
                float n_count = 0;
                Vector2d avg_heading(0.0f, 0.0f);
                for(vector<int>::iterator iit = k_indices.begin(); iit != k_indices.end(); ++iit){
                    PclPoint& nearby_pt = trajectories_->data()->at(*iit);
                    Vector3d vec(nearby_pt.x - point.x,
                                 nearby_pt.y - point.y,
                                 0.0f);
                    float dot_value = dir.dot(vec);
                    if(dot_value < 0.0f){
                        continue;
                    }
                    float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, heading));
                    if (!road->isOneway()){
                        if(delta_heading > 90.0f){
                            delta_heading = 180.0f - delta_heading;
                        }
                    }
                    
                    if(delta_heading > heading_threshold){
                        continue;
                    }
                    
                    float pt_heading_in_radius = nearby_pt.head * PI / 180.0f;
                    Vector2d pt_heading_dir(cos(pt_heading_in_radius),
                                            sin(pt_heading_in_radius));
                    Vector2d dir_2d(dir[0], dir[1]);
                    if (pt_heading_dir.dot(dir_2d) < 0) {
                        pt_heading_dir*= -1.0f;
                    }
                    
                    avg_heading += pt_heading_dir;
                    
                    n_count++;
                    if (max_dot_value < dot_value) {
                        max_dot_value = dot_value;
                    }
                    
                    float cross_value = dir.cross(vec)[2];
                    avg_vert_dist += cross_value;
                }
                if (n_count >= 2){
                    avg_vert_dist /= n_count;
                    avg_heading /= n_count;
                    avg_heading.normalize();
                    Vector2d delta(max_dot_value, avg_vert_dist);
                    delta.normalize();
                    delta *= delta_growth;
                    
                    float dx = delta[0] * cos(heading_in_radius) - delta[1] * sin(heading_in_radius);
                    float dy = delta[0] * sin(heading_in_radius) + delta[1] * cos(heading_in_radius);
                    float angle = acos(avg_heading[0]) * 180.0f / PI;
                    if(avg_heading[1] < 0.0f){
                        angle = 360.0f - angle;
                    }
                    road->center().push_back(Vertex(dx + point.x,
                                                    dy + point.y,
                                                    angle));
                }
                else{
                    road->endState() = TERMINAL;
                }
            }
        }
    }
}

bool RoadGenerator::exportQueryInitFeatures(float radius, const string &filename){
    // Compute features for each sample points
    current_feature_type_ = QUERY_INIT_FEATURE;
    vector<vector<float>> features;
    
    PclPointCloud::Ptr& samples = trajectories_->samples();
    
    features.clear();
    feature_vertices_.clear();
    feature_colors_.clear();
    feature_properties_.clear();
    for (size_t i = 0; i < samples->size(); ++i) {
        PclPoint& pt = samples->at(i);
        feature_properties_.push_back(Vertex(pt.x, pt.y, pt.head));
        
        vector<float> new_feature;
        computeQueryInitFeatureAt(radius, pt, trajectories_, new_feature, pt.head);
        features.push_back(new_feature);
        feature_vertices_.push_back(SceneConst::getInstance().normalize(pt.x, pt.y, Z_FEATURES));
        feature_colors_.push_back(ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY));
    }
    
    ofstream output;
    output.open(filename);
    if (output.fail()) {
        return false;
    }
    
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < features[i].size()-1; ++j) {
            output << features[i][j] << ", ";
        }
        output << features[i].back() << endl;
    }
    output.close();
    return true;
}

bool RoadGenerator::loadQueryInitFeatures(const string &filename){
    if(current_feature_type_ != QUERY_INIT_FEATURE){
        printf("Road Generator Warning: current feature is not query init!\n");
        return false;
    }
    
    ifstream fin(filename);
    if (!fin.good())
        return false;
    labels_.clear();
    
    int n_samples = 0;
    fin >> n_samples;
    for (size_t i = 0; i < n_samples; ++i) {
        int label;
        fin >> label;
        labels_.push_back(label);
        if (label == NON_OBVIOUS_ROAD) {
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::RED);
        }
//        else if (label == 1.0f){
//            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::GREEN);
//        }
        else{
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
        }
    }
    
    if (labels_.size() != feature_properties_.size()) {
        labels_.clear();
        printf("WARNING: labels do not match with features!\n");
    }
    
    fin.close();
    
    return true;
}

bool RoadGenerator::addQueryInitToString(){
    /*
        At each sample location, determine road type (oneway or twoway), guess road heading, road width
     */
    PclPointCloud::Ptr samples = trajectories_->samples();
    PclSearchTree::Ptr sample_tree = trajectories_->sample_tree();
    
    if (labels_.size() != samples->size()) {
        printf("WARNING: labels do not match query init features!\n");
        return false;
    }
    
    float search_radius = 25.0f;
    float sigma_n = 35.0f;
    float sigma_s = 7.5f;
    float delta_bin = 2.5f;
    float heading_threshold = 15.0f; // in degrees
    float minimum_high_speed = 15.0f; // in meters
    int N_BIN = ceil(2.0f * search_radius / delta_bin);
    int half_window_size = floor(sigma_s / delta_bin);
    
    vector<bool> is_visited(samples->size(), false);
    point_cloud_->clear();
    for (size_t i = 0; i < samples->size(); ++i) {
        if(labels_[i] == NON_OBVIOUS_ROAD){
            continue;
        }
        
        is_visited[i] = true;
        PclPoint& sample_pt = samples->at(i);
        
        // Search nearby
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        trajectories_->tree()->radiusSearch(sample_pt,
                                  search_radius,
                                  k_indices,
                                  k_dist_sqrs);
        
        float heading_in_radius = sample_pt.head * PI / 180.0f;
        Vector2d dir(cos(heading_in_radius), sin(heading_in_radius));
        Vector3d dir_3d(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
        Vector3d perp_dir(-sin(heading_in_radius), cos(heading_in_radius), 0.0f);
        
        // Determine direction
        bool is_oneway = false;
        int fw_pt_count = 0;
        int bw_pt_count = 0;
        float fw_dist = 0.0f;
        float bw_dist = 0.0f;
        float fw_speed = 0.0f;
        float bw_speed = 0.0f;
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = trajectories_->data()->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, sample_pt.head));
            if (delta_heading < heading_threshold) {
                // Consistent with forward direction
                fw_pt_count++;
                Vector3d vec(nb_pt.x - sample_pt.x,
                             nb_pt.y - sample_pt.y,
                             0.0f);
                fw_dist += vec.cross(dir_3d)[2];
                fw_speed += (static_cast<float>(nb_pt.speed) / 100.0f);
            }
            else if (delta_heading > 180.0f - heading_threshold){
                // Consistent with backward direction
                bw_pt_count++;
                Vector3d vec(nb_pt.x - sample_pt.x,
                             nb_pt.y - sample_pt.y,
                             0.0f);
                bw_dist += vec.cross(dir_3d)[2];
                bw_speed += (static_cast<float>(nb_pt.speed) / 100.0f);
            }
        }
        
        if (fw_pt_count > 0) {
            fw_dist /= fw_pt_count;
            fw_speed /= fw_pt_count;
        }
        
        if(bw_pt_count > 0){
            bw_dist /= bw_pt_count;
            bw_speed /= bw_pt_count;
        }
        
        int pt_count_threshold = 2;
        if(fw_pt_count > 30 && fw_speed > 0.5 * minimum_high_speed){
            pt_count_threshold = floor(0.1 * fw_pt_count);
        }
        
        
        if(fw_speed > minimum_high_speed && fw_pt_count > 30){
            is_oneway = true;
        }
       
        if (bw_pt_count < pt_count_threshold) {
            is_oneway = true;
        }
        
        float dist_threshold = 10.0f;
        if(fw_speed < minimum_high_speed){
            dist_threshold *= 2.0f;
        }
        
        if (fw_dist - bw_dist > dist_threshold) {
            is_oneway = true;
        }
        
        if(!is_oneway){
            feature_colors_[i] = ColorMap::getInstance().getNamedColor(ColorMap::GREEN);
        }
        
        // Estimate road at this site
        vector<float> distribution(N_BIN, 0.0f);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = trajectories_->data()->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, sample_pt.head));
            
            if (!is_oneway) {
                if (delta_heading > 90.0f) {
                    delta_heading = 180.0f - delta_heading;
                }
            }
            
            float nb_pt_heading_in_radius = nb_pt.head * PI / 180.0f;
            Vector2d velocity(cos(nb_pt_heading_in_radius), sin(nb_pt_heading_in_radius));
            velocity *= (nb_pt.speed / 100.0f);
            
            float velo_dot_value = abs(velocity.dot(dir)) / minimum_high_speed;
            
            if (delta_heading < heading_threshold) {
                Vector3d vec(nb_pt.x - sample_pt.x,
                             nb_pt.y - sample_pt.y,
                             0.0f);
                float projection = dir_3d.cross(vec)[2] + search_radius;
                int bin_idx = floor(projection / delta_bin);
                
                float vote_base = velo_dot_value * exp(-1.0f * projection * projection / sigma_n / sigma_n);
                
                for (int v_idx = bin_idx - half_window_size; v_idx < bin_idx + half_window_size; ++v_idx) {
                    if (v_idx < 0 || v_idx >= N_BIN) {
                        continue;
                    }
                    
                    float t_bin_center = (v_idx) * delta_bin;
                    float t_delta = (projection - t_bin_center);
                    float delta_vote = (vote_base * exp(-1.0f * t_delta * t_delta / sigma_s / sigma_s));
                    if(delta_vote < 1e-5){
                        delta_vote = 0.0f;
                    }
                    distribution[v_idx] += delta_vote;
                }
            }
        }
        
        // Detect peak
        float max_val = 0.0f;
        int max_idx = -1;
        for(size_t s = 0; s < N_BIN; ++s){
            if(distribution[s] > max_val){
                max_val = distribution[s];
                max_idx = s;
            }
        }
        
        // Get window
        float threshold = 0.25;
        int left_idx = max_idx;
        for (int s = max_idx; s >= 0; --s) {
            if (distribution[s] > threshold * max_val) {
                left_idx = s;
            }
        }
        int right_idx = max_idx;
        for (int s = max_idx; s < N_BIN; ++s) {
            if(distribution[s] > threshold * max_val){
                right_idx = s;
            }
        }
        float estimated_width = (right_idx - left_idx) * delta_bin;
        int n_lanes = floor(estimated_width / 3.7f);
        if(n_lanes == 0){
            n_lanes = 1;
        }
        
        float avg_speed = 0.0f;
        if(is_oneway){
            avg_speed = fw_speed;
        }
        else{
            avg_speed = 0.5 * (fw_speed + bw_speed);
        }
        
        if (avg_speed < minimum_high_speed) {
            n_lanes /= 2;
            if(n_lanes == 0){
                n_lanes = 1;
            }
        }
       
        if(is_oneway){
            if (n_lanes > 6){
                n_lanes = 6;
            }
        }
        else{
            if (n_lanes > 8){
                n_lanes = 8;
            }
        }
        
        float max_bin_center = (max_idx ) * delta_bin - search_radius;
        Vector2d loc(sample_pt.x, sample_pt.y);
        Vector2d perp_dir_2d(perp_dir.x(), perp_dir.y());
        loc += (max_bin_center * perp_dir_2d);
        PclPoint pt;
        pt.setCoordinate(loc[0], loc[1], 0.0f);
        pt.head = sample_pt.head;
        pt.id_sample = n_lanes; // just for storage, the name "id_sample" means nothing.
        pt.id_trajectory = is_oneway; // just for storage, the name "id_trajecotry" means nothing
        
        point_cloud_->push_back(pt);
//        cout << loc[0] << ", " << loc[1] <<", " << n_lanes<< ", " << is_oneway << endl;
    }
    
    search_tree_->setInputCloud(point_cloud_);
    trace_road();
    
//    for(size_t i = 0; i < samples->size(); ++i){
//        if(labels_[i] == NON_OBVIOUS_ROAD){
//            continue;
//        }
//        cout << samples->at(i).x << ", " << samples->at(i).y << endl;
//    }
    
    
    return true;
}

void RoadGenerator::trace_road(){
    cleanUp();
    vector<bool> is_marked(point_cloud_->size(), false);
    vector<int> random_sequence(point_cloud_->size(), 0);
    for(int i = 0; i < point_cloud_->size(); ++i){
        random_sequence[i] = i;
    }
    
    random_shuffle(random_sequence.begin(), random_sequence.end());
    
    vector<vector<RoadPt>> roads;
    for(size_t i = 0; i < point_cloud_->size(); ++i){
        int start_idx = random_sequence[i];
        if(is_marked[start_idx]){
            continue;
        }
        
        // Grow a road from this location
        vector<RoadPt> road;
        
        extend_road(start_idx, road, is_marked, true);
        extend_road(start_idx, road, is_marked, false);
        roads.push_back(road);
    }
    
    // Create initial roads and junctions by traversing these raw guesses
    float search_radius = 15.0f;
    float heading_threshold = 15.0f;
    PclPoint point;
    for (size_t i = 0; i < roads.size(); ++i) {
        vector<RoadPt>& road = roads[i];
        if (road.size() < 2) {
            continue;
        }
        
        RoadSymbol* new_road = new RoadSymbol;
        new_road->center().clear();
        new_road->isOneway() = road[0].is_oneway;
        new_road->startState() = QUERY_Q;
        new_road->endState() = QUERY_Q;
        
        float cum_road_width = 0.0f;
        // Get nearby points
        for(size_t j = 0; j < road.size(); ++j){
            // Fit another point
            bool to_break = false;
            if (new_road->center().size() >= 1){
                Vertex& last_pt = new_road->center().back();
                float delta_x = road[j].x - last_pt.x;
                float delta_y = road[j].y - last_pt.y;
                float delta_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                
                if(delta_dist < 10.0f){
                    continue;
                }
                int n_vertice_to_add = floor(delta_dist / 10.0f);
                
                Vector2d dir0(delta_x, delta_y);
                Vector2d start_loc(last_pt.x, last_pt.y);
                float length = dir0.norm();
                dir0.normalize();
                float delta_increase = length / n_vertice_to_add;
                for (int s = 0; s < n_vertice_to_add; ++s){
                    Vector2d loc = start_loc + (s + 1) * delta_increase * dir0;
                    // Add a new point
                    point.setCoordinate(loc[0], loc[1], 0.0f);
                    float heading = road[j].head;
                    float heading_in_radius = heading * PI / 180.0f;
                    Vector3d dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
                    
                    vector<int> k_indices;
                    vector<float> k_dist_sqrs;
                    trajectories_->tree()->radiusSearch(point, search_radius, k_indices, k_dist_sqrs);
                    
                    vector<int> nearby_pt;
                    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
                        PclPoint& nb_pt = trajectories_->data()->at(*it);
                        // Check heading compatibility
                        float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, heading));
                        if (!road[j].is_oneway) {
                            if(delta_heading > 90.0f){
                                delta_heading = 180.0f - delta_heading;
                            }
                        }
                        
                        if(delta_heading > heading_threshold){
                            continue;
                        }
                        
                        Vector3d vec(nb_pt.x - point.x,
                                     nb_pt.y - point.y,
                                     0.0f);
                        float perp_dist = abs(vec.cross(dir)[2]);
                        if(perp_dist > road[j].n_lanes * 3.7f){
                            continue;
                        }
                        
                        nearby_pt.push_back(*it);
                    }
                    
                    float cum_x = 0.0f;
                    float cum_y = 0.0f;
                    for(vector<int>::iterator it = nearby_pt.begin(); it != nearby_pt.end(); ++it){
                        PclPoint& nb_pt = trajectories_->data()->at(*it);
                        cum_x += nb_pt.x;
                        cum_y += nb_pt.y;
                    }
                    
                    if(nearby_pt.size() >= 1){
                        cum_x /= nearby_pt.size();
                        cum_y /= nearby_pt.size();
                        float strength = nearby_pt.size() * 1.0f / 20.0f;
                        float pt_x = (strength * cum_x + point.x) / (strength + 1.0f);
                        float pt_y = (strength * cum_y + point.y) / (strength + 1.0f);
                        Vertex new_v(pt_x, pt_y, road[j].head);
                        new_road->center().push_back(new_v);
                    }
                    else{
                        Vertex new_v(point.x, point.y, road[j].head);
                        new_road->center().push_back(new_v);
                    }
                    cum_road_width += 3.7 * road[j].n_lanes;
                    
                    if (new_road->center().size() >= 3) {
                        // Junction is allowed to be inserted only after the road is long enough (has at least 3 points).
                        PclPoint pt;
                        pt.setCoordinate(new_road->center().back().x, new_road->center().back().y, 0.0f);
                        pt.head = floor(new_road->center().back().z);
                        float pt_head_in_radius = pt.head * PI / 180.0f;
                        Vector3d t_dir(cos(pt_head_in_radius), sin(pt_head_in_radius), 0.0f);
                        vector<int> k_ind;
                        vector<float> k_dists;
                        trajectories_->tree()->radiusSearch(point, 10.0f, k_ind, k_dists);
                        bool has_left_branch = false;
                        int has_left_branch_count = 0;
                        Vector2d left_branch_dir;
                        bool has_right_branch = false;
                        int has_right_branch_count = 0;
                        Vector2d right_branch_dir;
                        
                        float avg_speed = 0.0f;
                        int count = 0;
                        for(size_t tt = 0; tt < k_ind.size(); ++tt){
                            int iit = k_ind[tt];
                            if (k_dists[tt] > 25.0f) {
                                continue;
                            }
                            PclPoint& nb_gps_pt = trajectories_->data()->at(iit);
                            avg_speed += nb_gps_pt.speed * 1.0f / 100.0f;
                            count += 1;
                        }
                        
                        if (count > 0) {
                            avg_speed /= count;
                        }
                        
                        if (avg_speed > 15.0f) {
                            // High speed
                            continue;
                        }
                        
                        for(vector<int>::iterator iit = k_ind.begin(); iit != k_ind.end(); ++iit){
                            PclPoint& nb_gps_pt = trajectories_->data()->at(*iit);
                            Vector3d t_vec(nb_gps_pt.x - pt.x,
                                           nb_gps_pt.y - pt.y,
                                           0.0f);
                            float dot_value = t_dir.dot(t_vec);
                            if(dot_value < -1.0f){
                                continue;
                            }
                            
                            float delta_heading = deltaHeading1MinusHeading2(nb_gps_pt.head, pt.head);
                            
                            if (delta_heading > 45.0f) {
                                // on left side
                                if(delta_heading > 45.0f){
                                    has_left_branch = true;
                                    has_left_branch_count++;
                                    left_branch_dir = Vector2d(t_vec[0], t_vec[1]);
                                }
                            }
                            
                            if (delta_heading < -45.0f){
                                // on right side
                                if(delta_heading > 45.0f){
                                    has_right_branch_count++;
                                    right_branch_dir = Vector2d(t_vec[0], t_vec[1]);
                                }
                            }
                        }
                        
                        int n_branch = 1;
                        if(has_left_branch_count > 10){
                            has_left_branch = true;
                        }
                        if(has_right_branch_count > 10){
                            has_right_branch = true;
                        }
                        
                        if (has_left_branch) {
                            n_branch++;
                        }
                        if (has_right_branch){
                            n_branch++;
                        }
                        
                        if(n_branch >= 100){
                            to_break = true;
                            // Insert current road to graph
                            cum_road_width /= new_road->center().size();
                            
                            int n_lanes = floor(cum_road_width / 3.7f);
                            if (new_road->isOneway()) {
                                if(n_lanes < 1){
                                    n_lanes = 1;
                                }
                                if(n_lanes > 6){
                                    n_lanes = 6;
                                }
                            }
                            else{
                                if(n_lanes < 2){
                                    n_lanes = 2;
                                }
                                if(n_lanes > 8){
                                    n_lanes = 8;
                                }
                            }
                            
                            new_road->nLanes() = n_lanes;
                            new_road->endState() = TERMINAL;
                            vertex_t u = add_vertex(symbol_graph_);
                            graph_nodes_[u] = new_road;
                            new_road->vertex_descriptor() = u;
                            
                            // Add junction
                            JunctionSymbol* new_junction = new JunctionSymbol;
                            new_junction->loc()[0] = pt.x;
                            new_junction->loc()[1] = pt.y;
                            vertex_t junc_v = add_vertex(symbol_graph_);
                            new_junction->vertex_descriptor() = junc_v;
                            graph_nodes_[junc_v] = new_junction;
                            add_edge(u, junc_v, symbol_graph_);
                            
                            // Add new branch road
                            if(has_left_branch){
                                RoadSymbol* left_road = new RoadSymbol;
                                left_road->startState() = TERMINAL;
                                left_road->endState() = QUERY_INIT;
                                left_branch_dir.normalize();
                                
                                if(left_branch_dir[0] > 1.0f){
                                    left_branch_dir[0] = 1.0f;
                                }
                                if(left_branch_dir[0] < -1.0f){
                                    left_branch_dir[0] = -1.0f;
                                }
                                
                                float angle = acos(left_branch_dir[0]) * 180.0f / PI;
                                if (left_branch_dir[1] < 0.0f) {
                                    angle = 360.0f - angle;
                                }
                                
                                left_road->center().push_back(Vertex(pt.x, pt.y, angle));
                                
                                vertex_t left_u = add_vertex(symbol_graph_);
                                left_road->vertex_descriptor() = left_u;
                                graph_nodes_[left_u] = left_road;
                                add_edge(junc_v, left_u, symbol_graph_);
                            }
                            if(has_right_branch){
                                RoadSymbol* right_road = new RoadSymbol;
                                right_road->startState() = TERMINAL;
                                right_road->endState() = QUERY_INIT;
                                right_branch_dir.normalize();
                                if(right_branch_dir[0] > 1.0f){
                                    right_branch_dir[0] = 1.0f;
                                }
                                if(right_branch_dir[0] < -1.0f){
                                    right_branch_dir[0] = -1.0f;
                                }
                               
                                float angle = acos(right_branch_dir[0]) * 180.0f / PI;
                                if (right_branch_dir[1] < 0.0f) {
                                    angle = 360.0f - angle;
                                }
                                
                                right_road->center().push_back(Vertex(pt.x, pt.y, angle));
                                
                                vertex_t right_u = add_vertex(symbol_graph_);
                                right_road->vertex_descriptor() = right_u;
                                graph_nodes_[right_u] = right_road;
                                add_edge(junc_v, right_u, symbol_graph_);
                            }
                            
                            // Continue current branch
                            new_road = new RoadSymbol;
                            new_road->startState() = TERMINAL;
                            new_road->endState() = QUERY_Q;
                            new_road->isOneway() = road[j].is_oneway;
                            Vertex new_v(pt.x, pt.y, road[j].head);
                            new_road->center().push_back(new_v);
                            cum_road_width = 0.0f;
                            cum_road_width += 3.7 * road[j].n_lanes;
//                            vertex_t new_u = add_vertex(symbol_graph_);
//                            new_road->vertex_descriptor() = new_u;
//                            graph_nodes_[new_u] = new_road;
//                            add_edge(junc_v, new_u, symbol_graph_);
                        }
                    }
                    
                    if(to_break){
                        break;
                    }
                }
            }
            else{
                // Get the first point
                point.setCoordinate(road[j].x, road[j].y, 0.0f);
                float heading = road[j].head;
                float heading_in_radius = heading * PI / 180.0f;
                Vector3d dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
                
                vector<int> k_indices;
                vector<float> k_dist_sqrs;
                trajectories_->tree()->radiusSearch(point, search_radius, k_indices, k_dist_sqrs);
                
                vector<int> nearby_pt;
                for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
                    PclPoint& nb_pt = trajectories_->data()->at(*it);
                    // Check heading compatibility
                    float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, heading));
                    if (!road[j].is_oneway) {
                        if(delta_heading > 90.0f){
                            delta_heading = 180.0f - delta_heading;
                        }
                    }
                    
                    if(delta_heading > heading_threshold){
                        continue;
                    }
                    
                    Vector3d vec(nb_pt.x - point.x,
                                 nb_pt.y - point.y,
                                 0.0f);
                    float perp_dist = abs(vec.cross(dir)[2]);
                    if(perp_dist > road[j].n_lanes * 3.7f){
                        continue;
                    }
                    
                    nearby_pt.push_back(*it);
                }
                
                float cum_x = 0.0f;
                float cum_y = 0.0f;
                for(vector<int>::iterator it = nearby_pt.begin(); it != nearby_pt.end(); ++it){
                    PclPoint& nb_pt = trajectories_->data()->at(*it);
                    cum_x += nb_pt.x;
                    cum_y += nb_pt.y;
                }
                
                if(nearby_pt.size() > 0){
                    cum_x /= nearby_pt.size();
                    cum_y /= nearby_pt.size();
                    float strength = nearby_pt.size() * 1.0f / 20.0f;
                    float pt_x = (strength * cum_x + road[j].x) / (strength + 1.0f);
                    float pt_y = (strength * cum_y + road[j].y) / (strength + 1.0f);
                    Vertex new_v(pt_x, pt_y, road[j].head);
                    new_road->center().push_back(new_v);
                }
                else{
                    Vertex new_v(road[j].x, road[j].y, road[j].head);
                    new_road->center().push_back(new_v);
                }
                cum_road_width += 3.7 * road[j].n_lanes;
            }
        }
        
        if (new_road->center().size() > 2) {
            cum_road_width /= new_road->center().size();
            
            int n_lanes = floor(cum_road_width / 3.7f);
            if (new_road->isOneway()) {
                if(n_lanes < 1){
                    n_lanes = 1;
                }
                if(n_lanes > 6){
                    n_lanes = 6;
                }
            }
            else{
                if(n_lanes < 2){
                    n_lanes = 2;
                }
                if(n_lanes > 8){
                    n_lanes = 8;
                }
            }
            
            new_road->nLanes() = n_lanes;
            
            vertex_t u = add_vertex(symbol_graph_);
            new_road->vertex_descriptor() = u;
            graph_nodes_[u] = new_road;
        }
        else{
            delete new_road;
        }
    }
    
    ofstream output;
    output.open("test_roads.txt");
    if (output.fail()) {
        printf("Warning: cannot create test_curve.txt!\n");
    }

    output << roads.size() << endl;
    for(size_t i = 0; i < roads.size(); ++i){
        vector<RoadPt>& road = roads[i];
        output << road.size() << endl;
        
        for(size_t j = 0; j < road.size(); ++j){
            output << road[j].x << ", " << road[j].y << endl;
        }
    }
    
    output.close();
}

void RoadGenerator::extend_road(int r_idx, vector<RoadPt> &road, vector<bool>& mark_list, bool forward){
   
    if (road.size() == 0) {
        // Add initial points
        PclPoint& pt = point_cloud_->at(r_idx);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        search_tree_->radiusSearch(pt, 10.0f, k_indices, k_dist_sqrs);
        vector<int> pt_set;
        float heading_in_radius = pt.head * PI / 180.0f;
        Vector3d dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
        
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            if(r_idx == *it){
                continue;
            }
            PclPoint &nb_pt = point_cloud_->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
            if(pt.id_trajectory == 0 || nb_pt.id_trajectory == 0){
                if(delta_heading > 90.0f){
                    delta_heading = 180.0f - delta_heading;
                }
            }
            
            if(delta_heading > 15.0f){
                continue;
            }
            
            Vector3d vec(nb_pt.x - pt.x,
                         nb_pt.y - pt.y,
                         0.0f);
            float perp_dist = abs(dir.dot(vec));
            float vertical_dist = abs(dir.cross(vec)[2]);
            if (vertical_dist < 5.0f && perp_dist < 5.0f) {
                pt_set.push_back(*it);
                mark_list[*it] = true;
            }
        }
    
        float cum_x = pt.x;
        float cum_y = pt.y;
        int n_for_oneway = 0;
        int n_for_twoway = 0;
        int n_oneway_lanes = 0;
        int n_twoway_lanes = 0;
        if(pt.id_trajectory == 1){
            // oneway
            n_for_oneway++;
            n_oneway_lanes += pt.id_sample;
        }
        else{
            n_for_twoway++;
            n_twoway_lanes += pt.id_sample;
        }
        for(vector<int>::iterator it = pt_set.begin(); it != pt_set.end(); ++it){
            PclPoint& nb_pt = point_cloud_->at(*it);
            cum_x += nb_pt.x;
            cum_y += nb_pt.y;
            if(nb_pt.id_trajectory == 1){
                // oneway
                n_for_oneway++;
                n_oneway_lanes += nb_pt.id_sample;
            }
            else{
                n_for_twoway++;
                n_twoway_lanes += nb_pt.id_sample;
            }
        }
        
        RoadPt r_pt;
        r_pt.x = cum_x / (pt_set.size() + 1);
        r_pt.y = cum_y / (pt_set.size() + 1);
        if(n_for_oneway > n_for_twoway){
            r_pt.is_oneway = true;
            r_pt.n_lanes = n_oneway_lanes / n_for_oneway;
        }
        else{
            r_pt.is_oneway = false;
            r_pt.n_lanes = n_twoway_lanes / n_for_twoway;
        }
        r_pt.head = pt.head;
        road.push_back(r_pt);
    }
    
    float search_radius = 50.0f;
    
    RoadPt cur_pt;
    if(forward){
        cur_pt.x = road.back().x;
        cur_pt.head = road.back().head;
        cur_pt.y = road.back().y;
        cur_pt.is_oneway = road.back().is_oneway;
        cur_pt.n_lanes = road.back().n_lanes;
    }
    else{
        cur_pt.x = road.front().x;
        cur_pt.head = road.front().head;
        cur_pt.y = road.front().y;
        cur_pt.is_oneway = road.front().is_oneway;
        cur_pt.n_lanes = road.front().n_lanes;
    }
    
    while(true){
        PclPoint pt;
        pt.setCoordinate(cur_pt.x, cur_pt.y, 0.0f);
        pt.head = cur_pt.head;
        float head_in_radius = pt.head * PI / 180.0f;
        Vector3d dir(cos(head_in_radius), sin(head_in_radius), 0.0f);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        vector<int> compatible_set;
        map<int, float> compatible_set_parallel_dist;
        search_tree_->radiusSearch(pt, search_radius, k_indices, k_dist_sqrs);
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            if(mark_list[*it]){
                continue;
            }
            
            // Check compatibility
            PclPoint& nb_pt = point_cloud_->at(*it);
            if(cur_pt.is_oneway && nb_pt.id_trajectory == 0){
                continue;
            }
            if(!cur_pt.is_oneway && nb_pt.id_trajectory == 1){
                continue;
            }
            
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
            if(nb_pt.id_trajectory == 0 || !cur_pt.is_oneway){
                if(delta_heading > 90.0f){
                    delta_heading = 180.0f - delta_heading;
                }
            }
            if(delta_heading > 30.0f){
                continue;
            }
            
            Vector3d vec(nb_pt.x - pt.x,
                         nb_pt.y - pt.y,
                         0.0f);
            float dot_value = dir.dot(vec);
            if (forward && dot_value <= 0) {
                continue;
            }
            if (!forward && dot_value >= 0) {
                continue;
            }
            
            float vec_length = vec.norm();
            float abs_dot_value = abs(dot_value / vec_length);
            float perp_dist = abs(dir.cross(vec)[2]);
            float perp_threshold = cur_pt.n_lanes * 3.7f;
            if (!cur_pt.is_oneway){
                perp_threshold *= 3.0f;
            }
            if (perp_dist < perp_threshold && abs(dot_value) < 10.0f){
                compatible_set.push_back(*it);
                compatible_set_parallel_dist[*it] = abs(dot_value);
                continue;
            }
            
            if (abs_dot_value < 0.9f) {
                continue;
            }
            compatible_set.push_back(*it);
            compatible_set_parallel_dist[*it] = abs(dot_value);
        }
        if (compatible_set.size() == 0) {
            break;
        }
        
        // Find the dist value for first qualifying pt
        float tmp_dist = -1.0f;
        for(map<int, float>::iterator d_it = compatible_set_parallel_dist.begin(); d_it != compatible_set_parallel_dist.end(); ++d_it){
            if(d_it->second > 10.0f){
                if(tmp_dist < 0.0f){
                    tmp_dist = d_it->second;
                }
                else{
                    if(tmp_dist > d_it->second){
                        tmp_dist = d_it->second;
                    }
                }
            }
        }
        if(tmp_dist < 0.0f){
            break;
        }
        
        // Mark and add
        for(vector<int>::iterator t_it = compatible_set.begin(); t_it != compatible_set.end(); ++t_it){
            if(compatible_set_parallel_dist[*t_it] < tmp_dist){
                mark_list[*t_it] = true;
            }
            else{
                if(compatible_set_parallel_dist[*t_it] - tmp_dist < 1e-3){
                    mark_list[*t_it] = true;
                    // Add new point
                    PclPoint& target_pt = point_cloud_->at(*t_it);
                    RoadPt new_pt;
                    new_pt.x = target_pt.x;
                    new_pt.y = target_pt.y;
                    if(target_pt.id_trajectory == 1){
                        new_pt.is_oneway = true;
                    }
                    else{
                        new_pt.is_oneway = false;
                    }
                    new_pt.n_lanes = target_pt.id_sample;
                   
                    float d_head = abs(deltaHeading1MinusHeading2(target_pt.head, cur_pt.head));
                    if(d_head > 90.0f){
                        new_pt.head = (target_pt.head + 180) % 360;
                    }
                    else{
                        new_pt.head = target_pt.head;
                    }
                    
                    if(forward){
                        road.push_back(new_pt);
                    }
                    else{
                        road.insert(road.begin(), new_pt);
                    }
                    
                    cur_pt.x = new_pt.x;
                    cur_pt.y = new_pt.y;
                    cur_pt.head = new_pt.head;
                    cur_pt.is_oneway = new_pt.is_oneway;
                    cur_pt.n_lanes = new_pt.n_lanes;
                }
            }
        }
    }
}


//bool RoadGenerator::addQueryInitToString(){
//    if(current_feature_type_ != QUERY_INIT_FEATURE){
//        printf("Road Generator Warning: current feature is not query init!\n");
//        return false;
//    }
//    
//    float SEARCH_RADIUS = 25.0f;
//    float NON_ROAD_SAFETY_RADIUS = 15.0f;
//    float HEADING_THRESHOLD = 7.5f;
//    
//    vector<int> assigned_road_info(trajectories_->data()->size(), -1);
//    vector<float> assigned_road_heading(trajectories_->data()->size(), 0.0f);
//    // Transport sample label to GPS points
//    PclPoint pt;
//   
//    for (size_t i = 0; i < trajectories_->samples()->size(); ++i){
//        if (labels_[i] != NON_OBVIOUS_ROAD) {
//            continue;
//        }
//        PclPoint& sample_pt = trajectories_->samples()->at(i);
//        vector<int> k_indices;
//        vector<float> k_dists;
//        trajectories_->sample_tree()->radiusSearch(sample_pt, SEARCH_RADIUS, k_indices, k_dists);
//        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//            labels_[*it] = NON_OBVIOUS_ROAD;
//        }
//    }
//    
//    for (size_t i = 0; i < trajectories_->samples()->size(); ++i) {
//        PclPoint& sample_pt = trajectories_->samples()->at(i);
//        vector<int> k_indices;
//        vector<float> k_dists;
//        switch (labels_[i]) {
//            case NON_OBVIOUS_ROAD:
//                trajectories_->tree()->radiusSearch(sample_pt, NON_ROAD_SAFETY_RADIUS, k_indices, k_dists);
//                for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//                    assigned_road_info[*it] = NON_OBVIOUS_ROAD;
//                }
//                break;
//            case ONEWAY_ROAD:
//                trajectories_->tree()->radiusSearch(sample_pt, SEARCH_RADIUS, k_indices, k_dists);
//                for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//                    if (assigned_road_info[*it] != NON_OBVIOUS_ROAD) {
//                        // Check heading
//                        float delta_heading = abs(deltaHeading1MinusHeading2(trajectories_->data()->at(*it).head, sample_pt.head));
//                        if (delta_heading <= HEADING_THRESHOLD) {
//                            assigned_road_info[*it] = ONEWAY_ROAD;
//                            assigned_road_heading[*it] = sample_pt.head;
//                        }
//                    }
//                }
//                break;
//            case TWOWAY_ROAD:
//                trajectories_->tree()->radiusSearch(sample_pt, SEARCH_RADIUS, k_indices, k_dists);
//                for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//                    if (assigned_road_info[*it] != NON_OBVIOUS_ROAD) {
//                        // Check heading
//                        float delta_heading = abs(deltaHeading1MinusHeading2(trajectories_->data()->at(*it).head, sample_pt.head));
//                        if (delta_heading > 90) {
//                            delta_heading = 180.0f - delta_heading;
//                        }
//                        if (delta_heading <= HEADING_THRESHOLD) {
//                            assigned_road_info[*it] = TWOWAY_ROAD;
//                            assigned_road_heading[*it] = sample_pt.head;
//                        }
//                    }
//                }
//                break;
//                
//            default:
//                break;
//        }
//    }
//    
//    // Fit road
//    float RADIUS = 15.0f;
//    float BIN_SIZE = 5.0f;
//    vector<bool> data_marked(trajectories_->data()->size(), false);
//    for (size_t i = 0; i < trajectories_->samples()->size(); ++i) {
//        if (labels_[i] == NON_OBVIOUS_ROAD) {
//            continue;
//        }
//        
//        PclPoint& sample_pt = trajectories_->samples()->at(i);
//        float heading_in_radius = sample_pt.head * PI / 180.0f;
//        Eigen::Vector3f dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
//        
//        set<int> nearby_pts;
//        for (int j = -4; j <= 4; ++j) {
//            Eigen::Vector2f v = j * RADIUS * Eigen::Vector2f(dir.x(), dir.y());
//            pt.setCoordinate(sample_pt.x + v.x(), sample_pt.y + v.y(), 0.0f);
//            vector<int> k_indices;
//            vector<float> k_dists;
//            trajectories_->tree()->radiusSearch(pt, RADIUS, k_indices, k_dists);
//            for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
//                if (data_marked[*it]){
//                    continue;
//                }
//                if (nearby_pts.find(*it) != nearby_pts.end()) {
//                    continue;
//                }
//                
//                if (assigned_road_info[*it] != -1 && assigned_road_info[*it] != labels_[i]) {
//                    continue;
//                }
//                
//                if (assigned_road_info[*it] != -1) {
//                    float delta_assigned_heading = abs(deltaHeading1MinusHeading2(assigned_road_heading[*it], sample_pt.head));
//                    if (labels_[i] == TWOWAY_ROAD && delta_assigned_heading > 90.0f) {
//                        delta_assigned_heading = 180.0f - delta_assigned_heading;
//                    }
//                    
//                    if (delta_assigned_heading > HEADING_THRESHOLD) {
//                        continue;
//                    }
//                }
//                
//                // Check heading
//                PclPoint& nb_pt = trajectories_->data()->at(*it);
//                float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, sample_pt.head));
//                if (labels_[i] == TWOWAY_ROAD && delta_heading > 90.0f) {
//                    delta_heading = 180.0f - delta_heading;
//                }
//                if (delta_heading < HEADING_THRESHOLD) {
//                    nearby_pts.insert(*it);
//                }
//            }
//        }
//        
//        // Project points in nearby_pts to coordinate system where x axis is dir.
//        if (nearby_pts.size() == 0) {
//            continue;
//        }
//        
//        map<int, int> projected_x;
//        map<int, int> projected_y;
//        map<int, float> projected_x_value;
//        map<int, float> projected_y_value;
//        map<int, bool> is_active;
//        float min_x = 1e6;
//        float min_y = 1e6;
//        float max_x = -1e6;
//        float max_y = -1e6;
//        for (set<int>::iterator it = nearby_pts.begin(); it != nearby_pts.end(); ++it) {
//            Eigen::Vector3f vec(trajectories_->data()->at(*it).x - sample_pt.x,
//                                trajectories_->data()->at(*it).y - sample_pt.y,
//                                0.0f);
//            float x_proj = vec.dot(dir);
//            float y_proj = dir.cross(vec)[2];
//            
//            projected_x_value[*it] = x_proj;
//            projected_y_value[*it] = y_proj;
//            if (x_proj < min_x) {
//                min_x = x_proj;
//            }
//            if (x_proj > max_x) {
//                max_x = x_proj;
//            }
//            if (y_proj < min_y) {
//                min_y = y_proj;
//            }
//            if (y_proj > max_y) {
//                max_y = y_proj;
//            }
//            is_active[*it] = true;
//        }
//        // Shift projected_x and projected_y by min_x and min_y
//        for (map<int, float>::iterator it = projected_x_value.begin(); it != projected_x_value.end(); ++it) {
//            int key = it->first;
//            projected_x[key] = floor((projected_x_value[key] - min_x) / BIN_SIZE);
//            projected_y[key] = floor((projected_y_value[key] - min_y) / BIN_SIZE);
//        }
//        
//        // Iterative x and y axis projection to get the road parameter
//        int n_x_bins = ceil((max_x - min_x) / BIN_SIZE)+1;
//        int n_y_bins = ceil((max_y - min_y) / BIN_SIZE)+1;
//        
//        int shifted_x_orig = floor((0.0f - min_x) / BIN_SIZE);
//        int shifted_y_orig = floor((0.0f - min_y) / BIN_SIZE);
//        int MAX_N_X_ESCAPE = 2; // the length direction
//        int MAX_N_Y_ESCAPE = 1; // the width direction
//        int x_left, x_right;
//        int y_left, y_right;
//        bool early_escape = false;
//        for (int l = 0; l < 4; ++l) {
//            // x-axis projection
//            int active_pts_this_iter = 0;
//            vector<int> x_hist(n_x_bins, 0);
//            for (set<int>::iterator it = nearby_pts.begin(); it != nearby_pts.end(); ++it) {
//                if (!is_active[*it]) {
//                    continue;
//                }
//                active_pts_this_iter++;
//                x_hist[projected_x[*it]]++;
//            }
//            
//            if (active_pts_this_iter < 3) {
//                early_escape = true;
//                break;
//            }
//           
//            // Find peak
//            float max_value = -1e6;
//            int max_bin_idx = -1;
//            for (int bin_idx = 0; bin_idx < n_x_bins; ++bin_idx) {
//                int dist_to_x_orig = abs(bin_idx - shifted_x_orig) + 1;
//                float cur_value = static_cast<float>(x_hist[bin_idx]) / dist_to_x_orig;
//                if (cur_value > max_value) {
//                    max_value = cur_value;
//                    max_bin_idx = bin_idx;
//                }
//            }
//           
//            int dist_to_x_orig = abs(max_bin_idx - shifted_x_orig);
//            if (dist_to_x_orig * BIN_SIZE > RADIUS) {
//                early_escape = true;
//                break;
//            }
//            // Extend
//            float x_threshold = x_hist[max_bin_idx] * 0.3f + 1;
//            if (x_threshold > 10) {
//                x_threshold = 10;
//            }
//            x_left = max_bin_idx;
//            x_right = max_bin_idx;
//            int left_escape_count = 0;
//            while (x_left >= 0) {
//                if (x_hist[x_left] < x_threshold) {
//                    left_escape_count++;
//                }
//                if (left_escape_count > MAX_N_X_ESCAPE) {
//                    x_left++;
//                    break;
//                }
//                x_left--;
//            }
//            int right_escape_count = 0;
//            while (x_right < n_x_bins){
//                if (x_hist[x_right] < x_threshold) {
//                    right_escape_count++;
//                }
//                if (right_escape_count > MAX_N_X_ESCAPE) {
//                    x_right--;
//                    break;
//                }
//                x_right++;
//            }
//            
//            // Eliminate non-qualifying
//            for (set<int>::iterator it = nearby_pts.begin(); it != nearby_pts.end(); ++it) {
//                if(!is_active[*it]){
//                    continue;
//                }
//                
//                if (projected_x[*it] <= x_left) {
//                    is_active[*it] = false;
//                }
//                if (projected_x[*it] >= x_right) {
//                    is_active[*it] = false;
//                }
//            }
//            
//            // y-axis projection
//            vector<int> y_hist(n_y_bins, 0);
//            for (set<int>::iterator it = nearby_pts.begin(); it != nearby_pts.end(); ++it) {
//                if (!is_active[*it]) {
//                    continue;
//                }
//                y_hist[projected_y[*it]]++;
//            }
//            
//            // Find peak
//            max_value = -1e6;
//            max_bin_idx = -1;
//            for (int bin_idx = 0; bin_idx < n_y_bins; ++bin_idx) {
//                int dist_to_y_orig = abs(bin_idx - shifted_y_orig) + 1;
//                float cur_value = static_cast<float>(y_hist[bin_idx]) / dist_to_y_orig;
//                if (cur_value > max_value) {
//                    max_value = cur_value;
//                    max_bin_idx = bin_idx;
//                }
//            }
//            
//            int dist_to_y_orig = abs(max_bin_idx - shifted_y_orig);
//            if (dist_to_y_orig * BIN_SIZE > 20.0f) {
//                early_escape = true;
//                break;
//            }
//            
//            // Extend
//            float y_threshold = y_hist[max_bin_idx] * 0.4f + 2;
//            y_left = max_bin_idx;
//            y_right = max_bin_idx;
//            left_escape_count = 0;
//            while (y_left >= 0) {
//                if (y_hist[y_left] < y_threshold) {
//                    left_escape_count++;
//                }
//                if (left_escape_count > MAX_N_Y_ESCAPE) {
//                    y_left++;
//                    break;
//                }
//                y_left--;
//            }
//            right_escape_count = 0;
//            while (y_right < n_y_bins){
//                if (y_hist[y_right] < y_threshold) {
//                    right_escape_count++;
//                }
//                if (right_escape_count > MAX_N_Y_ESCAPE) {
//                    y_right--;
//                    break;
//                }
//                y_right++;
//            }
//            
//            // Eliminate non-qualifying
//            for (set<int>::iterator it = nearby_pts.begin(); it != nearby_pts.end(); ++it) {
//                if(!is_active[*it]){
//                    continue;
//                }
//                if (projected_y[*it] <= y_left) {
//                    is_active[*it] = false;
//                }
//                if (projected_y[*it] >= y_right) {
//                    is_active[*it] = false;
//                }
//            }
//        }
//        
//        if (early_escape) {
//            continue;
//        }
//        
//        float y_value = 0.5f * (y_left + y_right);
//        float yloc = (y_value + 0.5f) * BIN_SIZE + min_y;
//        float xmin_loc, xmax_loc;
//        xmin_loc = (x_left + 0.5f) * BIN_SIZE + min_x;
//        xmax_loc = (x_right + 0.5f) * BIN_SIZE + min_x;
//        float delta_x = xmin_loc * dir[0] - yloc * dir[1];
//        float delta_y = xmin_loc * dir[1] + yloc * dir[0];
//        Eigen::Vector2f center_start(delta_x, delta_y);
//        center_start += Eigen::Vector2f(sample_pt.x, sample_pt.y);
//        
//        delta_x = xmax_loc * dir[0] - yloc * dir[1];
//        delta_y = xmax_loc * dir[1] + yloc * dir[0];
//        Eigen::Vector2f center_end(delta_x, delta_y);
//        center_end += Eigen::Vector2f(sample_pt.x, sample_pt.y);
//        
//        // Add a road
//        RoadSymbol* road = new RoadSymbol;
//        road->center().push_back(Vertex(center_start.x(), center_start.y(), sample_pt.head));
//        road->center().push_back(Vertex(center_end.x(), center_end.y(), sample_pt.head));
//        int d_y = (y_right - y_left > 1) ? (y_right - y_left - 1) : 1;
//        float road_width = d_y * BIN_SIZE;
//        road->nLanes() = floor(road_width / road->laneWidth());
//        if (labels_[i] == ONEWAY_ROAD) {
//            road->isOneway() = true;
//        }
//        else{
//            road->isOneway() = false;
//            if(road->nLanes() %2 != 0){
//                road->nLanes() ++;
//            }
//        }
//        
//        // Mark data point
//        int n_marked_data = 0;
//        for (set<int>::iterator it = nearby_pts.begin(); it != nearby_pts.end(); ++it) {
//            if (projected_x[*it] > x_left && projected_x[*it] < x_right) {
//                if (projected_y[*it] > y_left - 1 && projected_y[*it] < y_right + 1) {
//                    PclPoint& data_point = trajectories_->data()->at(*it);
//                    if(road->coveredPts().emplace(*it).second){
//                        if (road->coveredTrajs().emplace(data_point.id_trajectory).second) {
//                            road->coveredTrajMinTs()[data_point.id_trajectory] = data_point.t;
//                            road->coveredTrajMaxTs()[data_point.id_trajectory] = data_point.t;
//                        }
//                        else{
//                            if (data_point.t < road->coveredTrajMinTs()[data_point.id_trajectory]) {
//                                road->coveredTrajMinTs()[data_point.id_trajectory] = data_point.t;
//                            }
//                            if (data_point.t > road->coveredTrajMaxTs()[data_point.id_trajectory]) {
//                                road->coveredTrajMaxTs()[data_point.id_trajectory] = data_point.t;
//                            }
//                        }
//                        
//                    }
//                    else{
//                        if (data_point.t < road->coveredTrajMinTs()[data_point.id_trajectory]) {
//                            road->coveredTrajMinTs()[data_point.id_trajectory] = data_point.t;
//                        }
//                        if (data_point.t > road->coveredTrajMaxTs()[data_point.id_trajectory]) {
//                            road->coveredTrajMaxTs()[data_point.id_trajectory] = data_point.t;
//                        }
//                    }
//                }
//                data_marked[*it] = true;
//                n_marked_data++;
//            }
//        }
//        
//        int n_x = x_right - x_left;
//        int n_y = y_right - y_left;
//        if (n_x <= 2 && n_y <= 2) {
//            delete road;
//            continue;
//        }
//        
//        production_string_.push_back(road);
//    }
//    
//    // Debuging, change GPS colors
//    //vector<Color>& gps_point_colors = trajectories_->vertex_colors();
////    for(size_t i = 0; i < assigned_road_info.size(); ++i){
////        if(assigned_road_info[i] == NON_OBVIOUS_ROAD){
////            gps_point_colors[i] = ColorMap::getInstance().getNamedColor(ColorMap::DARK_GRAY);
////        }
////        else if (assigned_road_info[i] == ONEWAY_ROAD){
////            gps_point_colors[i] = ColorMap::getInstance().getNamedColor(ColorMap::RED);
////        }
////        else if (assigned_road_info[i] == TWOWAY_ROAD){
////            gps_point_colors[i] = ColorMap::getInstance().getNamedColor(ColorMap::GREEN);
////        }
////        else{
////            gps_point_colors[i] = ColorMap::getInstance().getNamedColor(ColorMap::BLUE);
////        }
////    }
//    
//    return true;
//}

void RoadGenerator::detectOverlappingRoadSeeds(vector<RoadSymbol*>& road_list, vector<vector<int>>& result){
    result.clear();
    for (size_t i = 0; i < road_list.size(); ++i) {
        // Find an overlapping group
        vector<Vertex>& cur_centerline = road_list[i]->center();
        int cur_n_pt = cur_centerline.size();
        Eigen::Vector2d cur_start_pt(cur_centerline[0].x, cur_centerline[0].y);
        Eigen::Vector2d cur_end_pt(cur_centerline[cur_n_pt-1].x, cur_centerline[cur_n_pt-1].y);
        Eigen::Vector2d cur_vec = cur_end_pt - cur_start_pt;
        cur_vec.normalize();
      
        bool has_cluster = false;
        int cluster_idx = -1;
        for (int j = 0; j < result.size(); ++j) {
            vector<int>& cluster = result[j];
            bool is_overlapping = false;
            for (int k = 0; k < cluster.size(); ++k) {
                int road_idx = cluster[k];
                vector<Vertex>& centerline = road_list[road_idx]->center();
                float road_width = 0.5 * road_list[road_idx]->nLanes() * road_list[road_idx]->laneWidth();
                int n_pt = centerline.size();
                Eigen::Vector2d start_pt(centerline[0].x, centerline[0].y);
                Eigen::Vector2d end_pt(centerline[n_pt-1].x, centerline[n_pt-1].y);
                Eigen::Vector2d vec = end_pt - start_pt;
                float length = vec.norm();
                vec /= length;
              
                Eigen::Vector2d vec1 = cur_start_pt - start_pt;
                float vec1_length = vec1.norm();
                float dot_value = vec.dot(vec1);
                if (dot_value >= 0.0f && dot_value < length) {
                    float dist = sqrt(vec1_length * vec1_length - dot_value * dot_value);
                    if (dist < road_width) {
                        is_overlapping = true;
                        break;
                    }
                }
                
                Eigen::Vector2d vec2 = cur_end_pt - start_pt;
                float vec2_length = vec2.norm();
                float dot_value2 = vec.dot(vec2);
                if (dot_value2 >= 0.0f && dot_value2 < length) {
                    float dist = sqrt(vec2_length * vec2_length - dot_value2 * dot_value2);
                    if (dist < road_width) {
                        is_overlapping = true;
                        break;
                    }
                }
            }
            if (is_overlapping) {
                has_cluster = true;
                cluster_idx = j;
                break;
            }
        }
        if (has_cluster) {
            result[cluster_idx].push_back(i);
        }
        else{
            vector<int> new_cluster;
            new_cluster.push_back(i);
            result.push_back(new_cluster);
        }
    }
   
    cout << "Merging result: "<<endl;
    for (size_t i = 0; i < result.size(); ++i) {
        cout << "\t";
        for (size_t j = 0; j < result[i].size(); ++j) {
            cout << result[i][j] << ",";
        }
        cout << endl;
    }
}

bool RoadGenerator::exportQueryQFeatures(const string &filename){
    return true;
}

bool RoadGenerator::loadQueryQPredictions(const string &filename){
    return true;
}

void RoadGenerator::production(){
    
}

void RoadGenerator::resolveQuery(){
}

void RoadGenerator::localAdjustment(){
    // Detect overlapping
    vector<RoadSymbol*> new_roads;
    vector<bool> symbol_is_active(production_string_.size(), true);
    map<int, int> road_map_to;
    
    refit();
    
//    int n_pt = 100;
//    
//    float X_SCALE = 10.0f;
//    float Y_SCALE = 10.0f;
//    float NOISE_R = 0.02f;
//    
//    // Clean data
//    float x3 = 1.0f;
//    float x2 = 1.0f;
//    float x1 = -1.0f;
//    float x0 = 1.0f;
//    float y3 = 0.0f;
//    float y2 = -1.0f;
//    float y1 = -1.0f;
//    float y0 = 1.0f;
//    float t_start = -1.5f;
//    float t_end = 1.5f;
//    vector<Vertex> clean_data;
//    for (int i = 0; i < n_pt; ++i) {
//        float t = i * (t_end - t_start) / n_pt + t_start;
//        float x = x3 * pow(t, 3) + x2 * pow(t, 2) + x1 * t + x0;
//        float y = y3 * pow(t, 3) + y2 * pow(t, 2) + y1 * t + y0;
//        Vertex a_pt;
//        a_pt.x = X_SCALE * x;
//        a_pt.y = Y_SCALE * y;
//        clean_data.push_back(a_pt);
//    }
//    
//    // Compute heading and add noise
//    vector<Vertex> pts;
//    for (int i = 0; i < n_pt; ++i) {
//        float delta_x = 0.0f;
//        float delta_y = 0.0f;
//        if (i > 0) {
//            delta_x = clean_data[i].x - clean_data[i-1].x;
//            delta_y = clean_data[i].y - clean_data[i-1].y;
//        }
//        else{
//            delta_x = clean_data[i+1].x - clean_data[i].x;
//            delta_y = clean_data[i+1].y - clean_data[i].y;
//        }
//        float length = sqrt(delta_x*delta_x + delta_y*delta_y);
//        float cos_value = delta_x / length;
//        float heading = acos(cos_value) * 180.0f / PI;
//        if (delta_y < 0) {
//            heading = 360.0f - heading;
//        }
//        int delta_angle = rand() % 360;
//        float delta_angle_radius = static_cast<float>(delta_angle) * PI / 180.0f;
//        float noise_x = NOISE_R * cos(delta_angle_radius);
//        float noise_y = NOISE_R * sin(delta_angle_radius);
//        Vertex a_pt;
//        a_pt.x = clean_data[i].x + noise_x;
//        a_pt.y = clean_data[i].y + noise_y;
//        a_pt.z = heading;
//        pts.push_back(a_pt);
//    }
//    
//    ofstream output;
//    output.open("test_curve.txt");
//    if (output.fail()) {
//        printf("Warning: cannot create test_curve.txt!\n");
//    }
//    
//    for (int i = 0; i < n_pt; ++i) {
//        output << pts[i].x << ", " << pts[i].y << ", " << pts[i].z << endl;
//    }
//    
//    output.close();
//    
//    // Fit polygonal line
//    clock_t begin = clock();
//    
//    vector<Vertex> results;
//    int delta_d = n_pt / 5;
//    int idx = 0;
//    while(idx < n_pt){
//        results.push_back(pts[idx]);
//        idx += delta_d;
//    }
//    results.push_back(pts[n_pt-1]);
//    
//    float avg_dist = 0.0f;
//    polygonalFitting(pts, results, avg_dist);
//    
//    clock_t end = clock();
//    float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
//    printf("For %d points, fitting took %.2f sec.\n", n_pt, elapsed_secs);
//    
//    output.open("fitted_curve.txt");
//    if (output.fail()) {
//        printf("Warning: cannot create fitted_curve.txt!\n");
//    }
//    
//    for (int i = 0; i < results.size(); ++i) {
//        output << results[i].x << ", " << results[i].y << ", " << results[i].z << endl;
//    }
//    
//    output.close();
    
//    // Rebuild search tree
//    updatePointCloud();
//    
//    // Detect overlapping road
//    vector<vector<int>> roads_to_merge;
//    detectRoadsToMerge(roads_to_merge);
//    mergeRoads(roads_to_merge);
}

void RoadGenerator::mergeRoads(vector<vector<int>>& roads_to_merge){
    // Generate merged roads
    for (size_t i = 0; i < roads_to_merge.size(); ++i) {
        vector<int>& roads = roads_to_merge[i];
        RoadSymbol* new_road = new RoadSymbol;
        
        getMergedRoadsFrom(roads, new_road);
    }
    
    // Update production_string_
}

void RoadGenerator::getMergedRoadsFrom(vector<int>& candidate_roads, RoadSymbol* new_road){
    // Vote for the direction of the final road
    int n_pt_for_oneway = 0;
    int n_pt_for_twoway = 0;
    int largest_road_count = 0;
    int largest_road_idx = -1;
    for (size_t i = 0; i < candidate_roads.size(); ++i) {
        RoadSymbol *road = dynamic_cast<RoadSymbol*>(production_string_[candidate_roads[i]]);
        int n_pts = road->coveredPts().size();
        if (largest_road_count < n_pts) {
            largest_road_count = n_pts;
            largest_road_idx = i;
        }
        if (road->isOneway()) {
            n_pt_for_oneway += road->coveredPts().size();
        }
        else{
            n_pt_for_twoway += road->coveredPts().size();
        }
    }
    bool is_oneway = true;
    if (n_pt_for_oneway < n_pt_for_twoway) {
        is_oneway = false;
    }
    
    vector<Vertex> current_center;
    RoadSymbol *starting_road = dynamic_cast<RoadSymbol*>(production_string_[candidate_roads[largest_road_idx]]);
    for (size_t i = 0; i < starting_road->center().size(); ++i) {
        current_center.push_back(starting_road->center()[i]);
    }
    
    if (is_oneway) {
        for (int i = 0; i < candidate_roads.size(); ++i) {
            if (i == largest_road_idx) {
                continue;
            }
            
            RoadSymbol *a_road = dynamic_cast<RoadSymbol*>(production_string_[candidate_roads[i]]);
            vector<vector<int>> assignments;
            map<int, float> dists;
            dividePointsIntoSets(a_road->center(), current_center, assignments, dists);
            
            vector<Vertex> updated_center;
            int n_current_pts = current_center.size();
            for(int j = 0; j < current_center.size(); ++j){
                if (j == 0) {
                    if (assignments[j].size() > 0) {
                        vector<int>& assignment = assignments[j];
                        for (int k = 0; k < assignment.size(); ++k) {
                            updated_center.push_back(a_road->center()[assignment[k]]);
                        }
                    }
                    updated_center.push_back(current_center[0]);
                    
                    for (vector<int>::iterator it = assignments[j + n_current_pts].begin(); it != assignments[j + n_current_pts].end(); ++it) {
                        updated_center.push_back(a_road->center()[*it]);
                    }
                    continue;
                }
                else if(j == current_center.size() - 1){
                    updated_center.push_back(current_center[j]);
                    for (vector<int>::iterator it = assignments[j].begin(); it != assignments[j].end(); ++it) {
                        updated_center.push_back(a_road->center()[*it]);
                    }
                }
                else{
                    float avg_x = current_center[j].x;
                    float avg_y = current_center[j].y;
                    int n = 1;
                    for (vector<int>::iterator it = assignments[j].begin(); it != assignments[j].end(); ++it) {
                        avg_x += a_road->center()[*it].x;
                        avg_y += a_road->center()[*it].y;
                        n++;
                    }
                    avg_x /= n;
                    avg_y /= n;
                    updated_center.push_back(Vertex(avg_x, avg_y, 0.0f));
                    for (vector<int>::iterator it = assignments[j + n_current_pts].begin(); it != assignments[j + n_current_pts].end(); ++it) {
                        updated_center.push_back(a_road->center()[*it]);
                    }
                }
            }
            current_center.clear();
            for (size_t i = 0; i < updated_center.size(); ++i) {
                current_center.push_back(updated_center[i]);
            }
            updated_center.clear();
        }
    }
    else{
        
    }
}

void RoadGenerator::updatePointCloud(){
    point_cloud_->clear();
    PclPoint pt;
    for (size_t i = 0; i < production_string_.size(); ++i) {
        if (production_string_[i]->type() == ROAD) {
            RoadSymbol *road = dynamic_cast<RoadSymbol*>(production_string_[i]);
            if (road->startState() == QUERY_INIT) {
                continue;
            }
            
            // Insert the center location into the point cloud
            for (size_t j = 0; j < road->center().size(); ++j) {
                pt.setCoordinate(road->center()[j].x, road->center()[j].y, road->center()[j].z);
                pt.id_trajectory = i;
                point_cloud_->push_back(pt);
            }
        }
    }
    search_tree_->setInputCloud(point_cloud_);
}

void RoadGenerator::detectRoadsToMerge(vector<vector<int>>& roads_to_merge){
    if (has_been_covered_.size() != trajectories_->data()->size()) {
        printf("ERROR! Something is wrong! Trajectory point cloud and has_been_covered GPS points does not match!\n");
        clear();
        return;
    }
    
    roads_to_merge.clear();
    
    vector<vector<int>> roads_cover_same_pts;
    for(size_t i = 0; i < has_been_covered_.size(); ++i){
        if(!has_been_covered_[i]){
            continue;
        }
        // Go over current roads
        vector<int> roads_cover_this_pts;
        for (size_t j = 0; j < production_string_.size(); ++j) {
            if (production_string_[j]->type() == ROAD) {
                RoadSymbol *road = dynamic_cast<RoadSymbol*>(production_string_[j]);
                if (road->coveredPts().find(i) != road->coveredPts().end()) {
                    // This road does cover this point
                    roads_cover_this_pts.push_back(j);
                }
            }
        }
        if (roads_cover_this_pts.size() >= 1) {
            roads_cover_same_pts.push_back(roads_cover_this_pts);
        }
    }
    
    map<pair<int, int>, int> cum_bins;
    for (size_t i = 0; i < roads_cover_same_pts.size(); ++i) {
        vector<int>& roads = roads_cover_same_pts[i];
        for (int j = 0; j < roads.size() - 1; ++j) {
            for (int k = j + 1; k < roads.size(); ++k) {
                pair<int, int> key(roads[j], roads[k]);
                map<pair<int,int>, int>::iterator fit = cum_bins.find(key);
                if (fit != cum_bins.end()) {
                    fit->second++;
                }
                else{
                    cum_bins[key] = 1;
                }
            }
        }
    }
    
    int MIN_PTS_FOR_MERGING = 3;
    map<int, int> already_merged_roads;
    for (map<pair<int,int>, int>::iterator it = cum_bins.begin(); it != cum_bins.end(); ++it) {
        if (it->second > MIN_PTS_FOR_MERGING) {
            int first_road_idx = it->first.first;
            int second_road_idx = it->first.second;
            
            // Check the compatibility of the two roads
            int larger_road_idx = first_road_idx;
            int smaller_road_idx = second_road_idx;
            RoadSymbol *first_road = dynamic_cast<RoadSymbol*>(production_string_[first_road_idx]);
            RoadSymbol *second_road = dynamic_cast<RoadSymbol*>(production_string_[second_road_idx]);
            if (second_road->center().size() > first_road->center().size()) {
                larger_road_idx = second_road_idx;
                smaller_road_idx = first_road_idx;
            }
            
            RoadSymbol *larger_road = dynamic_cast<RoadSymbol*>(production_string_[larger_road_idx]);
            RoadSymbol *smaller_road = dynamic_cast<RoadSymbol*>(production_string_[smaller_road_idx]);
            
            vector<Vertex>& r_center = larger_road->center();
            vector<vector<int>> point_assignment;
            set<int>& points_to_check = smaller_road->coveredPts();
            map<int, float> min_dists;
            float larger_road_width = 0.5 * larger_road->nLanes() * larger_road->laneWidth();
            if(dividePointsIntoSets(points_to_check,
                                    trajectories_->data(),
                                    r_center,
                                    point_assignment,
                                    min_dists,
                                    true,
                                    larger_road->isOneway(),
                                    2.5 * larger_road_width)){
                // merge
                map<int, int>::iterator first_merged = already_merged_roads.find(first_road_idx);
                int merge_cluster_idx = -1;
                bool first_is_merged = false;
                if (first_merged != already_merged_roads.end()) {
                    merge_cluster_idx = first_merged->second;
                    first_is_merged = true;
                }
                bool second_is_merged = false;
                map<int, int>::iterator second_merged = already_merged_roads.find(second_road_idx);
                if (second_merged != already_merged_roads.end()) {
                    merge_cluster_idx = second_merged->second;
                    second_is_merged = true;
                }
                if (merge_cluster_idx != -1) {
                    if (!first_is_merged) {
                        roads_to_merge[merge_cluster_idx].push_back(first_road_idx);
                    }
                    if (!second_is_merged) {
                        roads_to_merge[merge_cluster_idx].push_back(second_road_idx);
                    }
                }
                else{
                    vector<int> new_cluster;
                    new_cluster.push_back(first_road_idx);
                    new_cluster.push_back(second_road_idx);
                    already_merged_roads[first_road_idx] = roads_to_merge.size();
                    already_merged_roads[second_road_idx] = roads_to_merge.size();
                    roads_to_merge.push_back(new_cluster);
                }
            }
        }
    }
}

void minMaxFinder(vector<float>& values, float& min_value, int& min_idx, float& max_value, int& max_idx){
    min_value = 1e9;
    max_value = -1e9;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] < min_value) {
            min_value = values[i];
            min_idx = i;
        }
        if (values[i] > max_value) {
            max_value = values[i];
            max_idx = i;
        }
    }
}

void windowFinder(vector<float>& values, int start_idx, float& threshold, int& left, int& right){
    left = start_idx;
    for (left = start_idx; left > 0; --left) {
        if (values[left] < threshold * values[start_idx]) {
            break;
        }
    }
    
    for (right = start_idx; right < values.size() - 1; ++right) {
        if (values[right] < threshold * values[start_idx]) {
            break;
        }
    }
}

bool RoadGenerator::fitARoadAt(PclPoint& loc, RoadSymbol* road, bool is_oneway, float heading){
    // Default fitting parameters
    float SEARCH_RADIUS = 15.0f; // in meters, which also means the maximum road patch length is 100m.
    float ANGLE_THRESHOLD = 10.0f; // in degrees
    
    // Search nearby points
    vector<int> k_indices;
    vector<float> k_dists;
    
    trajectories_->tree()->radiusSearch(loc, SEARCH_RADIUS, k_indices, k_dists);
    
    // Filter points by their headings with ANGLE_THRESHOLD
    vector<int> compatible_pts;
    vector<float> compatible_pts_dist_sqrts;
    for (size_t i = 0; i < k_indices.size(); ++i) {
        PclPoint& pt = trajectories_->data()->at(k_indices[i]);
        bool is_compatible = true;
        float delta_angle = deltaHeading1MinusHeading2(pt.head, heading);
        if (is_oneway) {
            if (abs(delta_angle) > ANGLE_THRESHOLD) {
                is_compatible = false;
            }
        }
        else{
            float abs_delta_angle = abs(delta_angle);
            if (abs_delta_angle > 90.0f) {
                abs_delta_angle = 180.0f - abs_delta_angle;
            }
            if (abs_delta_angle > ANGLE_THRESHOLD) {
                is_compatible = false;
            }
        }
        
        if (is_compatible) {
            compatible_pts.push_back(k_indices[i]);
            compatible_pts_dist_sqrts.push_back(k_dists[i]);
        }
    }
    
    // Project compatible points into the canonical coordinate system
    float DELTA_BIN = 1.0f;
    int N_BINS = ceil(2 * SEARCH_RADIUS / 2.5f);
    DELTA_BIN = 2 * SEARCH_RADIUS / N_BINS;
    float heading_in_radius = heading / 180.0f * PI;
    Eigen::Vector3d canonical_dir(cos(heading_in_radius), sin(heading_in_radius), 0.0f);
    
    vector<float> y_distribution(N_BINS, 0.0f);
    vector<float> x_distribution(N_BINS, 0.0f);
    set<pair<int, int>> occupied_cells;
    for (size_t i = 0; i < compatible_pts.size(); ++i) {
        PclPoint& compatible_pt = trajectories_->data()->at(compatible_pts[i]);
        Eigen::Vector3d vec(compatible_pt.x - loc.x, compatible_pt.y - loc.y, 0.0f);
        float x_projection = vec.dot(canonical_dir) + SEARCH_RADIUS;
        float y_projection = canonical_dir.cross(vec)[2] + SEARCH_RADIUS;
        int x_bin_idx = floor(x_projection / DELTA_BIN);
        int y_bin_idx = floor(y_projection / DELTA_BIN);
        occupied_cells.insert(pair<int, int>(x_bin_idx, y_bin_idx));
    }
    
    
    for (set<pair<int, int>>::iterator it = occupied_cells.begin(); it != occupied_cells.end(); ++it) {
        for (int s = it->first - 1; s <= it->first + 1; ++s) {
            if (s < 0 || s >= N_BINS) {
                continue;
            }
            x_distribution[s] += 1.0f;
        }
        for (int s = it->second - 1; s <= it->second + 1; ++s) {
            if (s < 0 || s >= N_BINS) {
                continue;
            }
            y_distribution[s] += 1.0f;
        }
    }
    
    float X_RATIO = 0.1;
    float Y_RATIO = 0.75;
    
    float max_x = 0.0f;
    int max_x_idx = 0;
    for (int i = 0; i < x_distribution.size(); ++i) {
        if (x_distribution[i] >= max_x) {
            max_x_idx = i;
            max_x = x_distribution[i];
        }
    }
   
    int x_left, x_right;
    for(x_left = 0; x_left < max_x_idx - 1; ++x_left){
        if (x_distribution[x_left] > X_RATIO * max_x) {
            break;
        }
    }
    for(x_right = x_distribution.size() - 1; x_right > max_x_idx+1; --x_right){
        if (x_distribution[x_right] > X_RATIO * max_x) {
            break;
        }
    }
    
    float max_y = 0.0f;
    int max_y_idx = 0;
    for (int i = 0; i < y_distribution.size(); ++i) {
        if (y_distribution[i] >= max_y) {
            max_y_idx = i;
            max_y = y_distribution[i];
        }
    }
    
    int y_left, y_right;
    windowFinder(y_distribution, max_y_idx, Y_RATIO, y_left, y_right);
    
    // Convert to road attributes
    float x_start       = (x_left + 0.5) * DELTA_BIN - SEARCH_RADIUS;
    float x_end         = (x_right + 0.5) * DELTA_BIN - SEARCH_RADIUS;
    float y_start       = (y_left + 0.5) * DELTA_BIN - SEARCH_RADIUS;
    float y_end         = (y_right + 0.5) * DELTA_BIN - SEARCH_RADIUS;
    float y_value       = 0.5 * (y_start + y_end);
    
    float width = y_end - y_start;
    
    int estimated_n_lanes = ceil(width / road->laneWidth());
    if (is_oneway) {
        if (estimated_n_lanes < 1) {
            estimated_n_lanes = 1;
        }
    }
    else{
        if (estimated_n_lanes %2 != 0) {
            estimated_n_lanes += 1;
        }
        if (estimated_n_lanes < 2) {
            estimated_n_lanes = 2;
        }
    }
    
    float delta_x_start = x_start * canonical_dir[0] - y_value * canonical_dir[1];
    float delta_y_start = x_start * canonical_dir[1] + y_value * canonical_dir[0];
    float delta_x_end   = x_end * canonical_dir[0] - y_value * canonical_dir[1];
    float delta_y_end   = x_end * canonical_dir[1] + y_value * canonical_dir[0];
  
    Eigen::Vector2d road_start(delta_x_start, delta_y_start);
    Eigen::Vector2d road_end(delta_x_end, delta_y_end);
    Eigen::Vector2d length_vec = road_end - road_start;
    road_start += Eigen::Vector2d(loc.x, loc.y);
    road_end += Eigen::Vector2d(loc.x, loc.y);
    
    // Update road attributes
    road->center().push_back(Vertex(road_start.x(), road_start.y(), heading));
    road->center().push_back(Vertex(road_end.x(), road_end.y(), heading));
    road->isOneway() = is_oneway;
    road->nLanes() = estimated_n_lanes;
    
    // Insert the compatible points into the road
    float road_length = length_vec.norm();
    float half_road_width = 0.5 * estimated_n_lanes * road->laneWidth();
    for (size_t i = 0; i < compatible_pts.size(); ++i) {
        PclPoint& compatible_pt = trajectories_->data()->at(compatible_pts[i]);
        Eigen::Vector3d vec(compatible_pt.x - road_start.x(), compatible_pt.y - road_start.y(), 0.0f);
        float x_projection = vec.dot(canonical_dir);
        float y_projection = canonical_dir.cross(vec)[2];
        if (x_projection > -2.5f && x_projection < road_length + 2.5f) {
            if (abs(y_projection) <= 1.5 * half_road_width) {
                road->coveredPts().insert(compatible_pts[i]);
                if(road->coveredPts().emplace(compatible_pts[i]).second){
                    road->coveredTrajs().insert(compatible_pt.id_trajectory);
                    road->coveredTrajMinTs()[compatible_pt.id_trajectory] = compatible_pt.t;
                    road->coveredTrajMaxTs()[compatible_pt.id_trajectory] = compatible_pt.t;
                }
                else{
                    int id_traj = compatible_pt.id_trajectory;
                    if (road->coveredTrajMinTs()[id_traj] > compatible_pt.t) {
                        road->coveredTrajMinTs()[id_traj] = compatible_pt.t;
                    }
                    if (road->coveredTrajMaxTs()[id_traj] < compatible_pt.t) {
                        road->coveredTrajMaxTs()[id_traj] = compatible_pt.t;
                    }
                }
            }
        }
    }
    
    return true;
}

void RoadGenerator::fitRoadAt(int sample_idx, Eigen::Vector2d& start, Eigen::Vector2d& end, float& start_heading, float& end_heading, int& n_lanes, bool& is_oneway, float& lane_width, set<int>& covered_data){
    covered_data.clear();
    
    float SIGMA_DIST = 7.5f;
    
    PclPoint& sample_pt = trajectories_->samples()->at(sample_idx);
    Eigen::Vector2d sample_loc = Eigen::Vector2d(sample_pt.x, sample_pt.y);
    float sample_heading = sample_pt.head;
    float sample_heading_in_radius = sample_heading / 180.0 * PI;
    Eigen::Vector2d sample_dir = Eigen::Vector2d(cos(sample_heading_in_radius), sin(sample_heading_in_radius));
    Eigen::Vector2d sample_perp_dir = Eigen::Vector2d(sample_dir[1], -1.0 * sample_dir[0]);
    
    // Nearby points
    vector<int> k_indices;
    vector<float> k_dists;
    float SEARCH_RADIUS = 15.0; // in meters
    float ANGLE_THRESHOLD = 15.0f;
    
    trajectories_->tree()->radiusSearch(sample_pt, SEARCH_RADIUS, k_indices, k_dists);
    
    // Compute distributions
    int n_x_bins = 40;
    int n_y_bins = 40;
    
    float delta_x_bin = 2 * SEARCH_RADIUS / n_x_bins;
    float delta_y_bin = 2 * SEARCH_RADIUS / n_y_bins;
    
    int half_x_window_size = ceil(3 * SIGMA_DIST / delta_x_bin);
    
    vector<float> forward_x_dist(n_x_bins, 0.0f); // x projection in the canonical direction
    vector<float> backward_x_dist(n_x_bins, 0.0f); // x projection in the opposite of the canonical direction
    vector<float> forward_y_dist(n_y_bins, 0.0f);
    vector<float> backward_y_dist(n_y_bins, 0.0f);
    
    for (int i = 0; i < k_indices.size(); ++i) {
        PclPoint& pt = trajectories_->data()->at(k_indices[i]);
        float pt_heading = 450.0 - pt.head;
        if (pt_heading > 360.0f) {
            pt_heading -= 360.0f;
        }
        float angle_diff = abs(pt_heading - sample_heading);
        if (angle_diff > 180.0f) {
            angle_diff = 360.0f - angle_diff;
        }
        
        if (angle_diff < ANGLE_THRESHOLD) {
            // conform with the canonical direction
            Eigen::Vector2d v1 = Eigen::Vector2d(pt.x, pt.y) - sample_loc;
            float x_proj = v1.dot(sample_perp_dir);
            float y_proj = v1.dot(sample_dir);
            
            int x_bin_idx = floor((x_proj + SEARCH_RADIUS) / delta_x_bin);
            int y_bin_idx = floor((y_proj + SEARCH_RADIUS) / delta_y_bin);
            
            // Add to x and y histogram
            for (int j = -1.0 * half_x_window_size; j <= half_x_window_size; ++j) {
                int tmp_x_bin_idx = x_bin_idx + j;
                int tmp_y_bin_idx = y_bin_idx + j;
                float tmp_x_bin_center = (tmp_x_bin_idx + 0.5) * delta_x_bin - SEARCH_RADIUS;
                float tmp_y_bin_center = (tmp_y_bin_idx + 0.5) * delta_y_bin - SEARCH_RADIUS;
                float delta_x = tmp_x_bin_center - x_proj;
                float delta_y = tmp_y_bin_center - y_proj;
                
                if (tmp_x_bin_idx >= 0 && tmp_x_bin_idx < n_x_bins) {
                    forward_x_dist[tmp_x_bin_idx] += exp(-1.0f * delta_x * delta_x / 2.0f / SIGMA_DIST / SIGMA_DIST);
                }
                if (tmp_y_bin_idx >= 0 && tmp_y_bin_idx < n_y_bins) {
                    forward_y_dist[tmp_y_bin_idx] += exp(-1.0f * delta_y * delta_y / 2.0f / SIGMA_DIST / SIGMA_DIST);
                }
            }
        }
        else if(angle_diff > 90.0f){
            angle_diff = 180.0f - angle_diff;
            if (angle_diff < ANGLE_THRESHOLD) {
                // in the opposite of the canonical direction
                Eigen::Vector2d v1 = Eigen::Vector2d(pt.x, pt.y) - sample_loc;
                float x_proj = v1.dot(sample_perp_dir);
                float y_proj = v1.dot(sample_dir);
                int x_bin_idx = floor((x_proj + SEARCH_RADIUS) / delta_x_bin);
                int y_bin_idx = floor((y_proj + SEARCH_RADIUS) / delta_y_bin);
                
                // Add to x and y histogram
                for (int j = -1.0 * half_x_window_size; j <= half_x_window_size; ++j) {
                    int tmp_x_bin_idx = x_bin_idx + j;
                    int tmp_y_bin_idx = y_bin_idx + j;
                    float tmp_x_bin_center = (tmp_x_bin_idx + 0.5) * delta_x_bin - SEARCH_RADIUS;
                    float tmp_y_bin_center = (tmp_y_bin_idx + 0.5) * delta_y_bin - SEARCH_RADIUS;
                    float delta_x = tmp_x_bin_center - x_proj;
                    float delta_y = tmp_y_bin_center - y_proj;
                    
                    if (tmp_x_bin_idx >= 0 && tmp_x_bin_idx < n_x_bins) {
                        backward_x_dist[tmp_x_bin_idx] += exp(-1.0f * delta_x * delta_x / 2.0f / SIGMA_DIST / SIGMA_DIST);
                    }
                    if (tmp_y_bin_idx >= 0 && tmp_y_bin_idx < n_y_bins) {
                        backward_y_dist[tmp_y_bin_idx] += exp(-1.0f * delta_y * delta_y / 2.0f / SIGMA_DIST / SIGMA_DIST);
                    }
                }
            }
        }
    }
   
    // Fitting based on distribution
    float forward_min_x_val = 0.0f;
    int forward_min_x_idx = -1;
    float forward_max_x_val = 0.0f;
    int forward_max_x_idx = -1;
    minMaxFinder(forward_x_dist, forward_min_x_val, forward_min_x_idx, forward_max_x_val, forward_max_x_idx);
    int forward_x_left = -1;
    int forward_x_right = -1;
    float threshold;
    float X_RATIO = 0.75;
    float Y_RATIO = 0.1;
    if (forward_max_x_val > 1.0) {
        threshold = forward_min_x_val + X_RATIO * (forward_max_x_val - forward_min_x_val);
        windowFinder(forward_x_dist, forward_max_x_idx, threshold, forward_x_left, forward_x_right);
    }
    
    float forward_min_y_val = 0.0f;
    int forward_min_y_idx = -1;
    float forward_max_y_val = 0.0f;
    int forward_max_y_idx = -1;
    minMaxFinder(forward_y_dist, forward_min_y_val, forward_min_y_idx, forward_max_y_val, forward_max_y_idx);
    int forward_y_left = -1;
    int forward_y_right = -1;
    if (forward_max_y_val > 1.0f) {
        threshold = forward_min_y_val + Y_RATIO * (forward_max_y_val - forward_min_y_val);
        windowFinder(forward_y_dist, forward_max_y_idx, threshold, forward_y_left, forward_y_right);
    }
    
    float backward_min_x_val = 0.0f;
    int backward_min_x_idx = -1;
    float backward_max_x_val = 0.0f;
    int backward_max_x_idx = -1;
    int backward_x_left = -1;
    int backward_x_right = -1;
    minMaxFinder(backward_x_dist, backward_min_x_val, backward_min_x_idx, backward_max_x_val, backward_max_x_idx);
    if (backward_max_x_val > 1.0f) {
        threshold = backward_min_x_val + X_RATIO * (backward_max_x_val - backward_min_x_val);
        windowFinder(backward_x_dist, backward_max_x_idx, threshold, backward_x_left, backward_x_right);
    }
    
    float backward_min_y_val = 0.0f;
    int backward_min_y_idx = -1;
    float backward_max_y_val = 0.0f;
    int backward_max_y_idx = -1;
    int backward_y_left = -1;
    int backward_y_right = -1;
    minMaxFinder(backward_y_dist, backward_min_y_val, backward_min_y_idx, backward_max_y_val, backward_max_y_idx);
    if (backward_max_y_val > 1.0f) {
        threshold = backward_min_x_val + Y_RATIO * (backward_max_y_val - backward_min_y_val);
        windowFinder(backward_y_dist, backward_max_y_idx, threshold, backward_y_left, backward_y_right);
    }
    
    float forward_width = (forward_x_right - forward_x_left) * delta_x_bin;
    float backward_width = (backward_x_right - backward_x_left) * delta_x_bin;
    
    is_oneway = false;
    
    if (forward_width < 1.0f || backward_width < 1.0f) {
        is_oneway = true;
    }
    
    if (!is_oneway) {
        if (forward_x_left > backward_x_right) {
            is_oneway = true;
        }
        if (backward_x_right > forward_x_right) {
            is_oneway = true;
        }
        
        if (forward_max_x_val > 10.0 * backward_max_x_val){
            is_oneway = true;
        }
    }
    
    lane_width = 4.0f;
    if (!is_oneway) {
        n_lanes = floor((forward_x_right - backward_x_left) * delta_x_bin / lane_width + 0.5);
        if (n_lanes < 2) {
            n_lanes = 2;
        }
        float x_center = (0.5 * (forward_x_right + backward_x_left) + 0.5) * delta_x_bin - SEARCH_RADIUS;
        int max_y = (forward_y_right > backward_y_right)? forward_y_right : backward_y_right;
        int min_y = (forward_y_left < backward_y_left)? forward_y_left : backward_y_left;
        float y_center = (0.5 * (max_y + min_y) + 0.5) * delta_y_bin - SEARCH_RADIUS;
        float length = (max_y - min_y) * delta_y_bin;
       
        float delta_x = x_center * sample_dir[0] + y_center * sample_dir[1];
        float delta_y = -1 * x_center * sample_dir[1] + y_center * sample_dir[0];
        Eigen::Vector2d new_center = sample_loc + Eigen::Vector2d(delta_x, delta_y);
        start = new_center - 0.5 * length * sample_dir;
        end = new_center + 0.5 * length * sample_dir;
    }
    else{
        n_lanes = floor((forward_x_right - forward_x_left) * delta_x_bin / lane_width + 0.5);
        if (n_lanes < 1) {
            n_lanes = 1;
        }
        float x_center = (0.5 * (forward_x_right + forward_x_left) + 0.5) * delta_x_bin - SEARCH_RADIUS;
        float y_center = (0.5 * (forward_y_right + forward_y_left) + 0.5) * delta_y_bin - SEARCH_RADIUS;
        float length = (forward_y_right - forward_y_left) * delta_y_bin;
        
        float delta_x = x_center * sample_dir[0] - y_center * sample_dir[1];
        float delta_y = x_center * sample_dir[1] + y_center * sample_dir[0];
        Eigen::Vector2d new_center = sample_loc + Eigen::Vector2d(delta_x, delta_y);
        start = new_center - 0.5 * length * sample_dir;
        end = new_center + 0.5 * length * sample_dir;
    }
    
    start_heading = sample_heading;
    end_heading = sample_heading;
}

void RoadGenerator::refit(){
}

//void RoadGenerator::fitRoadAt(int sample_idx, Eigen::Vector2d& start, Eigen::Vector2d& end, float& start_heading, float& end_heading, int& n_lanes, int& n_directions, float& lane_width, set<int>& covered_data){
//    covered_data.clear();
//    
//    float HALF_LENGTH = 15.0; // in meters
//    float SIGMA_DIST = 15.0f;
//    float SIGMA_ANGLE = 15.0f;
//    float LAMBDA = 1;
//    PclPoint& sample_pt = trajectories_->samples()->at(sample_idx);
//    Eigen::Vector2d sample_loc = Eigen::Vector2d(sample_pt.x, sample_pt.y);
//    float sample_heading = trajectories_->sampleHeadings()[sample_idx];
//    float sample_heading_in_radius = sample_heading / 180.0 * PI;
//    Eigen::Vector2d sample_dir = Eigen::Vector2d(cos(sample_heading_in_radius), sin(sample_heading_in_radius));
//    start = sample_loc - HALF_LENGTH * sample_dir;
//    end = sample_loc + HALF_LENGTH * sample_dir;
//    
//    // Nearby points
//    vector<int> k_indices;
//    vector<float> k_dists;
//    trajectories_->tree()->radiusSearch(sample_pt, 1.2*HALF_LENGTH, k_indices, k_dists);
//    
//    // Decide the road directions: one-way or two-way
//    
//    std::default_random_engine generator;
//    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
//    float score_value = -1e9;
//   
//    int n_not_increasing = 0;
//    while (true) {
//        // Propose new parameters
//        float length = (end - start).norm();
//            // new start and new end: uniformly sample using a circle of radius length / 2.0f
//        float r = length / 4.0 * uniform(generator);
//        float angle = 2 * PI * uniform(generator);
//        Eigen::Vector2d new_start = start + Eigen::Vector2d(r*cos(angle), r*sin(angle));
//        r = length / 4.0 * uniform(generator);
//        angle = 2 * PI * uniform(generator);
//        Eigen::Vector2d new_end = end + Eigen::Vector2d(r*cos(angle), r*sin(angle));
//        
//            // new n_lanes: from 1 to 6
//        r = uniform(generator);
//        int new_n_lanes = floor(r * 6) + 1;
//        
//            // new lane_width: from 2.7m to 4.6m
//        r = uniform(generator);
//        float new_lane_width = 2.7f + 1.9f * r;
//        float new_half_road_width = 0.5f * new_n_lanes * new_lane_width;
//        
//        Eigen::Vector2d dir = (new_end - new_start);
//        float new_length = dir.norm();
//        dir /= new_length;
//        
//        // Compute new score
//        float new_score = 0.0f;
//        int np_inside = 0;
//        for (size_t i = 0; i < k_indices.size(); ++i) {
//            PclPoint& pt = trajectories_->data()->at(k_indices[i]);
//            float pt_head = 450.0 - pt.head;
//            if (pt_head > 360.0) {
//                pt_head -= 360.0;
//            }
//            
//            float delta_head = abs(pt_head - sample_heading);
//            if (delta_head > 180.0f) {
//                delta_head = 360 - delta_head;
//            }
//            if (delta_head > 30.0f) {
//                continue;
//            }
//            
//            Eigen::Vector2d v1 = Eigen::Vector2d(pt.x, pt.y) - new_start;
//            float v1_norm = v1.norm();
//            float dist_to_road = 0.0f;
//            float dot_value = v1.dot(dir);
//            float dist = sqrt(v1_norm*v1_norm - dot_value*dot_value);
//            if (dot_value >= 0.0f && dot_value < new_length) {
//                if (dist > new_half_road_width) {
//                    dist_to_road = dist - new_half_road_width;
//                }
//                else{
//                    dist_to_road = 0.0f;
//                    np_inside++;
//                }
//            }
//            else{
//                if (dot_value < 0) {
//                    dist_to_road = sqrt(dot_value*dot_value + dist*dist);
//                }
//                else{
//                    dot_value -= new_length;
//                    dist_to_road = sqrt(dot_value*dot_value + dist*dist);
//                }
//            }
//            
//            
//            float road_heading = acos(dir.dot(Eigen::Vector2d(1.0f, 0.0f))) * 180.0f / PI;
//            if (dir[1] < 0) {
//                road_heading = 360 - road_heading;
//            }
//            
//            float d_heading = abs(pt_head - road_heading);
//            if (d_heading > 180.0f) {
//                d_heading = 360.0 - d_heading;
//            }
//            
//            float data_log_probability = -1.0 * dist_to_road * dist_to_road / 2.0f / SIGMA_DIST / SIGMA_DIST;
//            
//            data_log_probability += -1.0 * d_heading * d_heading / 2.0f / SIGMA_ANGLE / SIGMA_ANGLE;
//            new_score += data_log_probability;
//        }
//        if (np_inside > 0) {
//            new_score -= LAMBDA * new_length * new_half_road_width * 2.0f / np_inside;
//        }
//        else{
//            new_score -= 1e3;
//        }
//        // Update if function is improved
//        if (new_score > score_value) {
//            // Update
//            score_value = new_score;
//            start = new_start;
//            end = new_end;
//            lane_width = new_lane_width;
//            n_lanes = new_n_lanes;
//            n_not_increasing = 0;
//        }
//        else{
//            n_not_increasing++;
//        }
//        if (n_not_increasing > 100) {
//            break;
//        }
//    }
//    
//    // Update covered data points and headings
//    Eigen::Vector2d dir = end - start;
//    float length = dir.norm();
//    float half_road_width = 0.5 * n_lanes * lane_width;
//    dir /= length;
//    for (size_t i = 0; i < k_indices.size(); ++i) {
//        PclPoint& pt = trajectories_->data()->at(k_indices[i]);
//        float pt_head = 450.0 - pt.head;
//        if (pt_head > 360.0) {
//            pt_head -= 360.0;
//        }
//        
//        Eigen::Vector2d v1 = Eigen::Vector2d(pt.x, pt.y) - start;
//        float v1_norm = v1.norm();
//        float dist_to_road = 0.0f;
//        float dot_value = v1.dot(dir);
//        float dist = sqrt(v1_norm*v1_norm - dot_value*dot_value);
//        if (dot_value >= 0.0f && dot_value < length) {
//            if (dist > half_road_width) {
//                dist_to_road = dist - half_road_width;
//            }
//            else{
//                dist_to_road = 0.0f;
//            }
//        }
//        else{
//            if (dot_value < 0) {
//                dist_to_road = sqrt(dot_value*dot_value + dist*dist);
//            }
//            else{
//                dot_value -= length;
//                dist_to_road = sqrt(dot_value*dot_value + dist*dist);
//            }
//        }
//        
//        float data_dist_probability = exp(-1.0 * dist_to_road * dist_to_road / 2.0f / SIGMA_DIST / SIGMA_DIST);
//        float delta_head = abs(pt_head - sample_heading);
//        if (delta_head > 180.0f) {
//            delta_head = 360 - delta_head;
//        }
//        
//        float data_angle_probability = exp(-1.0 * delta_head * delta_head / 2.0f / SIGMA_ANGLE / SIGMA_ANGLE);
//        
//        if (data_dist_probability > 0.7 && data_angle_probability > 0.7) {
//            covered_data.insert(k_indices[i]);
//        }
//    }
//    start_heading = sample_heading;
//    end_heading = sample_heading;
//}

void RoadGenerator::draw(){
    // Draw generated road
    vector<Vertex> junction_vertices;
    for (map<vertex_t, Symbol*>::iterator it = graph_nodes_.begin(); it != graph_nodes_.end(); ++it) {
        feature_vertices_.clear();
        feature_colors_.clear();
        if (it->second->type() == ROAD) {
            RoadSymbol *road = dynamic_cast<RoadSymbol*>(it->second);
            vector<Vertex> v;
            if(road->getDrawingVertices(v)){
                vector<Color> color;
                color.clear();
                if (road->isOneway()) {
                    color.resize(v.size(), Color(1,0,0,1));
                }
                else{
                    color.resize(v.size(), Color(0,1,0,1));
                }
                
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&v[0], 3*v.size()*sizeof(float));
                shadder_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&color[0], 4*color.size()*sizeof(float));
                shadder_program_->setupColorAttributes();
                glDrawArrays(GL_LINE_LOOP, 0, v.size());
            }
            else{
                vector<Color> color;
                color.clear();
                color.resize(v.size(), Color(0,0,1,1));
                
                vertexPositionBuffer_.create();
                vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexPositionBuffer_.bind();
                vertexPositionBuffer_.allocate(&v[0], 3*v.size()*sizeof(float));
                shadder_program_->setupPositionAttributes();
                
                vertexColorBuffer_.create();
                vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
                vertexColorBuffer_.bind();
                vertexColorBuffer_.allocate(&color[0], 4*color.size()*sizeof(float));
                shadder_program_->setupColorAttributes();
                glDrawArrays(GL_LINES, 0, v.size());
            }
        }
        else{
            // Draw junction
            Vertex v;
            
            JunctionSymbol *junction = dynamic_cast<JunctionSymbol*>(it->second);
            junction->getDrawingVertices(v);
            junction_vertices.push_back(v);
        }
    }
    
    if(junction_vertices.size() > 0){
        vector<Color> color;
        color.clear();
        color.resize(junction_vertices.size(), Color(1,1,0,1));
        
        vertexPositionBuffer_.create();
        vertexPositionBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexPositionBuffer_.bind();
        vertexPositionBuffer_.allocate(&junction_vertices[0], 3*junction_vertices.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        vertexColorBuffer_.create();
        vertexColorBuffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);
        vertexColorBuffer_.bind();
        vertexColorBuffer_.allocate(&color[0], 4*color.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        glPointSize(30);
        glDrawArrays(GL_POINTS, 0, junction_vertices.size());
    }
    
    // Draw features
    if (feature_vertices_.size() != 0) {
        QOpenGLBuffer position_buffer;
        position_buffer.create();
        position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        position_buffer.bind();
        position_buffer.allocate(&feature_vertices_[0], 3*feature_vertices_.size()*sizeof(float));
        shadder_program_->setupPositionAttributes();
        
        QOpenGLBuffer color_buffer;
        color_buffer.create();
        color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        color_buffer.bind();
        color_buffer.allocate(&feature_colors_[0], 4*feature_colors_.size()*sizeof(float));
        shadder_program_->setupColorAttributes();
        glPointSize(20);
        //    glLineWidth(5);
        //    glDrawArrays(GL_LINES, 0, feature_vertices_.size());
        glDrawArrays(GL_POINTS, 0, feature_vertices_.size());
    }
}

void RoadGenerator::clear(){
    trajectories_ = NULL;
    
    point_cloud_->clear();
   
    has_been_covered_.clear();
    feature_vertices_.clear();
    feature_colors_.clear();
    
    current_feature_type_ = NONE;
    feature_properties_.clear();
    labels_.clear();
    
    if(production_string_.size() != 0){
        for (size_t i = 0; i < production_string_.size(); ++i) {
            delete production_string_[i];
        }
    }
    production_string_.clear();
    cleanUp();
}

void RoadGenerator::cleanUp(){
    symbol_graph_.clear();
    
    if(graph_nodes_.size() > 0){
        for (map<vertex_t, Symbol*>::iterator it = graph_nodes_.begin(); it != graph_nodes_.end(); ++it) {
            delete it->second;
        }
    }
    
    graph_nodes_.clear();
}
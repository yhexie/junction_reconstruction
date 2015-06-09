#include <sstream>
#include <iomanip>

#include "common.h"
#include <pcl/search/impl/flann_search.hpp>
#include <QDebug>

namespace Common
{
    std::string int2String(int i, int width)
    {
        std::ostringstream ss;
        ss << std::setw(width) << std::setfill('0') << i;
        return ss.str();
    }
    
    void randomK(std::vector<int>& random_k, int k, int N)
    {
        std::vector<int> random_N(N);
        for (int i = 0; i < N; ++ i)
            random_N[i] = i;
        
        for (int i = 0; i < k; ++ i)
        {
            int idx = i + rand()%(N-i);
            std::swap(random_N[i], random_N[idx]);
        }
        
        random_k.assign(k, 0);
        for (int i = 0; i < k; ++ i)
            random_k[i] = random_N[i];
        
        return;
    }
}

bool pairCompare(const pair<int, float>& firstElem, const pair<int, float>& secondElem){
    return firstElem.second < secondElem.second;
}

bool pairCompareDescend(const pair<int, float>& firstElem, const pair<int, float>& secondElem){
    return firstElem.second > secondElem.second;
}

float roadPtDistance(const RoadPt& p1, const RoadPt& p2){
    Eigen::Vector2d vec(p1.x - p2.x,
                        p1.y - p2.y);
    return vec.norm();
}

void findMaxElement(const vector<float> hist, int& max_idx){
    if(hist.size() == 0){
        max_idx = -1;
        return;
    }
    
    float max_value = hist[0];
    max_idx = 0;
    for(size_t i = 0; i < hist.size(); ++i){
        if(hist[i] > max_value){
            max_value = hist[i];
            max_idx = i;
        }
    }
}

void peakDetector(vector<float>& hist,
                  int            window,
                  float          ratio,
                  vector<int>&   peak_idxs,
                  bool           is_closed){
    // Detecting peaks in a histogram
    //is_closed: true - loop around; false: do not loop around.
    // ratio should >= 1.0f
    peak_idxs.clear();

    vector<int> raw_peak_idxs;
    int left_offset = window / 2;
    int right_offset = window - left_offset;
    
    if(is_closed){
        for (int i = 0; i < hist.size(); ++i) {
            int start_idx = i - left_offset;
            int right_idx = i + right_offset;
            
            bool is_max = true;
            float min_value = 1e6;
            float avg_value = 0.0;
            int count = 0;
            for (int j = start_idx; j <= right_idx; ++j) {
                if(j == i) 
                    continue;

                int hist_idx = j;
                if (hist_idx < 0) {
                    hist_idx += hist.size();
                }
                if (hist_idx >= hist.size()) {
                    hist_idx = hist_idx % hist.size();
                }
                if (hist[hist_idx] > hist[i]) {
                    is_max = false;
                    break;
                }
                if (hist[hist_idx] < min_value) {
                    min_value = hist[hist_idx];
                }
                avg_value += hist[hist_idx];
                count += 1;
            }
            
            if (is_max) {
                if (count == 0) {
                    continue;
                }
                avg_value /= count;
                float d1 = hist[i] - min_value;
                float d2 = avg_value - min_value;
                
                if (d1 > ratio * d2) {
                    raw_peak_idxs.push_back(i);
                }
            }
        }
    }
    else{
        for (int i = 0; i < hist.size(); ++i) {
            int start_idx = i - left_offset;
            int right_idx = i + right_offset;
            if (start_idx < 0) {
                start_idx = 0;
            }
            if (right_idx >= hist.size()) {
                right_idx = hist.size() - 1;
            }
            
            bool is_max = true;
            float min_value = 1e6;
            float avg_value = 0.0;
            int count = 0;
            for (int j = start_idx; j <= right_idx; ++j) {
                if(i == j) 
                    continue;

                int hist_idx = j;
                
                if (hist[hist_idx] > hist[i]) {
                    is_max = false;
                    break;
                }
                
                if (hist[hist_idx] < min_value) {
                    min_value = hist[hist_idx];
                }
                avg_value += hist[hist_idx];
                count += 1;
            }
            
            if (is_max) {
                if (count == 0) {
                    continue;
                }
                avg_value /= count;
                float d1 = hist[i] - min_value;
                float d2 = avg_value - min_value;
                
                if (d1 > ratio * d2) {
                    raw_peak_idxs.push_back(i);
                }
            }
        }
    }

    std::sort(raw_peak_idxs.begin(), raw_peak_idxs.end()); 
    for(int i = 0; i < raw_peak_idxs.size(); ++i){
        if(i > 0){
            int delta_to_previous = abs(raw_peak_idxs[i-1] - raw_peak_idxs[i]);
            if(is_closed){
                if(delta_to_previous > 0.5f * hist.size()){
                    delta_to_previous = hist.size() - delta_to_previous;
                }
            }
            if(delta_to_previous > 0.5f * window){
                peak_idxs.emplace_back(raw_peak_idxs[i]);
            }
        }
        else{
            peak_idxs.emplace_back(raw_peak_idxs[i]);
        }
    }

    // Debug
//    if (!is_closed) {
//        cout << "hist: "<<endl;
//        cout << "\t";
//        for (size_t i = 0; i < hist.size(); ++i) {
//            cout << hist[i] << ", ";
//        }
//        cout << endl;
//        cout << "\t";
//        for (size_t i = 0; i < peak_idxs.size(); ++i) {
//            cout << peak_idxs[i] << ", ";
//        }
//        cout << endl;
//        cout << endl;
//    }
}

float deltaHeading1MinusHeading2(float heading1, float heading2){
    float delta_angle = abs(heading1 - heading2);
   
    if (delta_angle > 180.0f) {
        delta_angle = 360.0f - delta_angle;
       
        if(heading1 > heading2){
            return -1.0f * delta_angle;
        }
        else{
            return delta_angle;
        }
    }
    else{
        if(heading1 > heading2){
            return delta_angle;
        }
        else{
            return -1.0 * delta_angle;
        }
    }
    
}

Eigen::Vector2d headingTo2dVector(const int heading){
    if (heading < 0 || heading > 360) {
        cout << "Error when converting heading to 2d vector. Invalid input: " << heading << endl;
        return Eigen::Vector2d(0.0f, 0.0f);
    }
    
    float heading_in_radius = heading * PI / 180.0f;
    
    return Eigen::Vector2d(cos(heading_in_radius),
                           sin(heading_in_radius));
}

Eigen::Vector3d headingTo3dVector(const int heading){
    if (heading < 0 || heading > 360) {
        cout << "Error when converting heading to 3d vector. Invalid input! Input is: " << heading << endl;
        return Eigen::Vector3d(0.0f, 0.0f, 0.0f);
    }
    
    float heading_in_radius = heading * PI / 180.0f;
    
    return Eigen::Vector3d(cos(heading_in_radius),
                           sin(heading_in_radius),
                           0.0f);
}

int vector2dToHeading(const Eigen::Vector2d vec){
    Eigen::Vector2d tmp_vec(vec);
   
    float length = tmp_vec.norm();
    
    if(length < 1e-3){
        // very small vector
        cout << "Warning (from vectorToHeading2d): encountered zero vector when converting to heading." << endl;
        return 0;
    }
    
    tmp_vec /= length;
    float cos_value = tmp_vec[0];
    if (cos_value > 1.0f) {
        cos_value = 1.0f;
    }
    if (cos_value < -1.0f) {
        cos_value = -1.0f;
    }
    
    int angle = floor(acos(cos_value) * 180.0f / PI);
    
    if (tmp_vec[1] < 0) {
        angle = 360 - angle;
    }
    
    return angle;
}

int vector3dToHeading(const Eigen::Vector3d vec){
    Eigen::Vector3d tmp_vec(vec);
    
    float length = tmp_vec.norm();
    
    if(length < 1e-3){
        // very small vector
        cout << "Warning (from vectorToHeading3d): encountered zero vector when converting to heading." << endl;
        return 0;
    }

    tmp_vec /= length;
    float cos_value = tmp_vec[0];
    if (cos_value > 1.0f) {
        cos_value = 1.0f;
    }
    if (cos_value < -1.0f) {
        cos_value = -1.0f;
    }
    
    int angle = floor(acos(cos_value) * 180.0f / PI);
    
    if (tmp_vec[1] < 0) {
        angle = 360 - angle;
    }
    
    return angle;
}

int increaseHeadingBy(int delta_heading,
                      const int orig_heading){
    if (orig_heading < 0 || orig_heading > 360) {
        cout << "Error (from increaseHeadingBy). Invalid input.!" << endl;
        return -1;
    }
    
    return (orig_heading + delta_heading + 360) % 360;
}

int decreaseHeadingBy(int delta_heading,
                      const int orig_heading){
    if (orig_heading < 0 || orig_heading > 360) {
        cout << "Error (from decreaseHeadingBy). Invalid input.!" << endl;
        return -1;
    }
    
    return (orig_heading - delta_heading + 360) % 360;
}

void smoothCurve(vector<RoadPt>& center_line, bool fix_front){
    /*
     This function smooth the road center line represented as a vector of RoadPt.
     */
    if(center_line.size() < 2){
        return;
    }

    // Smooth heading
    int half_window_size = 2;
    float cum_length = 0.0f;
    for(int i = 0; i < center_line.size(); ++i){
        Eigen::Vector2d cum_vec(0.0f, 0.0f);
        for(int j = i - half_window_size; j <= i + half_window_size; ++j){
            if(j < 0 || j >= center_line.size())
                continue;
            cum_vec += headingTo2dVector(center_line[j].head);
        }

        center_line[i].head = vector2dToHeading(cum_vec);

        if(i > 0){
            float delta_x = center_line[i].x - center_line[i-1].x;
            float delta_y = center_line[i].y - center_line[i-1].y;
            cum_length += sqrt(delta_x*delta_x + delta_y*delta_y);
        }
    }

    float delta = 15.0f; // simplify center_line
    int N = ceil(center_line.size() * delta / cum_length);

    vector<RoadPt> tmp_center_line;
    for(int i = 0; i < center_line.size(); i = i+N){
        tmp_center_line.emplace_back(RoadPt(center_line[i]));
    }

    if((center_line.size() - 1) % N != 0){
        tmp_center_line.emplace_back(RoadPt(center_line.back()));
    }

    vector<float> orig_xs;
    vector<float> orig_ys;
    for (int i = 0; i < tmp_center_line.size(); ++i) {
        orig_xs.push_back(tmp_center_line[i].x);
        orig_ys.push_back(tmp_center_line[i].y);
    }
    
    if(fix_front){
        float orig_last_x = tmp_center_line.back().x;
        float orig_last_y = tmp_center_line.back().y;
        int max_iter = 1000;
        int i_iter = 0;
        while(i_iter <= max_iter){
            i_iter++;
            float cum_change = 0.0f;
            for(int i = 1; i < tmp_center_line.size() - 1; ++i){
                RoadPt& cur_pt = tmp_center_line[i];
                RoadPt& prev_pt = tmp_center_line[i-1];
                RoadPt& nxt_pt = tmp_center_line[i+1];
                Eigen::Vector2d prev_dir = headingTo2dVector(prev_pt.head);
                Eigen::Vector2d nxt_dir = headingTo2dVector(nxt_pt.head);
                Eigen::Vector2d prev_perp_dir(-prev_dir[1], prev_dir[0]);
                Eigen::Vector2d nxt_perp_dir(-nxt_dir[1], nxt_dir[0]);
                
                float A1 = 3 + pow(prev_perp_dir[0] + nxt_perp_dir[0], 2);
                float B1 = (prev_perp_dir[0] + nxt_perp_dir[0]) * (prev_perp_dir[1] + nxt_perp_dir[1]);
                float K = prev_perp_dir[0] * prev_pt.x + prev_perp_dir[1] * prev_pt.y + nxt_perp_dir[0] * nxt_pt.x + nxt_perp_dir[1] * nxt_pt.y;
                float C1 = prev_pt.x + nxt_pt.x + orig_xs[i] + (prev_perp_dir[0] + nxt_perp_dir[0]) * K;
                
                float B2 = 3 + pow(prev_perp_dir[1] + nxt_perp_dir[1], 2);
                float A2 = B1;
                float C2 = prev_pt.y + nxt_pt.y + orig_ys[i] + (prev_perp_dir[1] + nxt_perp_dir[1]) * K;
                
                float bottom = A2*B1 - A1*B2;
                if (abs(bottom) > 1e-3) {
                    float new_x = (B1*C2 - B2*C1) / bottom;
                    float new_y = (A2*C1 - A1*C2) / bottom;
                    float delta_x = new_x - cur_pt.x;
                    float delta_y = new_y - cur_pt.y;
                    
                    cum_change += sqrt(delta_x*delta_x + delta_y*delta_y);
                    
                    cur_pt.x = new_x;
                    cur_pt.y = new_y;
                }
            }
            
            // Update last point
            int last_idx = tmp_center_line.size()-1;
            RoadPt& cur_pt = tmp_center_line[last_idx];
            RoadPt& prev_pt = tmp_center_line[last_idx-1];
            Eigen::Vector2d prev_dir = headingTo2dVector(prev_pt.head);
            Eigen::Vector2d cur_dir = headingTo2dVector(cur_pt.head);
            Eigen::Vector2d prev_perp_dir(-prev_dir[1], prev_dir[0]);
            Eigen::Vector2d cur_perp_dir(-cur_dir[1], cur_dir[0]);
            
            float K = (prev_perp_dir[0] + cur_perp_dir[0]) * prev_pt.x
            + (prev_perp_dir[1] + cur_perp_dir[1]) * prev_pt.y;
            
            float A1 = 1.0f + pow(prev_perp_dir[0] + cur_perp_dir[0], 2.0f);
            float B1 = (prev_perp_dir[0] + cur_perp_dir[0]) * (prev_perp_dir[1] + cur_perp_dir[1]);
            float C1 = (prev_perp_dir[0] + cur_perp_dir[0])*K + orig_last_x;
            float B2 = 1.0f + pow(prev_perp_dir[1] + cur_perp_dir[1], 2.0f);
            float A2 = B1;
            float C2 = (prev_perp_dir[1] + cur_perp_dir[1])*K + orig_last_y;
            
            float bottom = A2*B1 - A1*B2;
            if (abs(bottom) > 1e-3) {
                float new_x = (B1*C2 - B2*C1) / bottom;
                float new_y = (A2*C1 - A1*C2) / bottom;
                float delta_x = new_x - cur_pt.x;
                float delta_y = new_y - cur_pt.y;
                
                cum_change += sqrt(delta_x*delta_x + delta_y*delta_y);
                
                cur_pt.x = new_x;
                cur_pt.y = new_y;
            }
            
            if (cum_change < 0.01f) {
                break;
            }
        }
    }
    else{
        float orig_first_x = tmp_center_line.front().x;
        float orig_first_y = tmp_center_line.front().y;
        int max_iter = 100;
        int i_iter = 0;
        while(i_iter <= max_iter){
            i_iter++;
            float cum_change = 0.0f;
            
            for(int i = tmp_center_line.size() - 2; i > 0; --i){
                RoadPt& cur_pt = tmp_center_line[i];
                RoadPt& prev_pt = tmp_center_line[i-1];
                RoadPt& nxt_pt = tmp_center_line[i+1];
                Eigen::Vector2d prev_dir = headingTo2dVector(prev_pt.head);
                Eigen::Vector2d nxt_dir = headingTo2dVector(nxt_pt.head);
                Eigen::Vector2d prev_perp_dir(-prev_dir[1], prev_dir[0]);
                Eigen::Vector2d nxt_perp_dir(-nxt_dir[1], nxt_dir[0]);
                
                float A1 = 3 + pow(prev_perp_dir[0] + nxt_perp_dir[0], 2);
                float B1 = (prev_perp_dir[0] + nxt_perp_dir[0]) * (prev_perp_dir[1] + nxt_perp_dir[1]);
                float K = prev_perp_dir[0] * prev_pt.x + prev_perp_dir[1] * prev_pt.y + nxt_perp_dir[0] * nxt_pt.x + nxt_perp_dir[1] * nxt_pt.y;
                float C1 = prev_pt.x + nxt_pt.x + orig_xs[i] + (prev_perp_dir[0] + nxt_perp_dir[0]) * K;
                
                float B2 = 3 + pow(prev_perp_dir[1] + nxt_perp_dir[1], 2);
                float A2 = B1;
                float C2 = prev_pt.y + nxt_pt.y + orig_ys[i] + (prev_perp_dir[1] + nxt_perp_dir[1]) * K;
                
                float bottom = A2*B1 - A1*B2;
                if (abs(bottom) > 1e-3) {
                    float new_x = (B1*C2 - B2*C1) / bottom;
                    float new_y = (A2*C1 - A1*C2) / bottom;
                    float delta_x = new_x - cur_pt.x;
                    float delta_y = new_y - cur_pt.y;
                    
                    cum_change += sqrt(delta_x*delta_x + delta_y*delta_y);
                    
                    cur_pt.x = new_x;
                    cur_pt.y = new_y;
                }
            }
            
            // Update first point
            RoadPt& cur_pt = tmp_center_line[0];
            RoadPt& nxt_pt = tmp_center_line[1];
            Eigen::Vector2d nxt_dir = headingTo2dVector(nxt_pt.head);
            Eigen::Vector2d cur_dir = headingTo2dVector(cur_pt.head);
            Eigen::Vector2d nxt_perp_dir(-nxt_dir[1], nxt_dir[0]);
            Eigen::Vector2d cur_perp_dir(-cur_dir[1], cur_dir[0]);
            
            float K = (nxt_perp_dir[0] + cur_perp_dir[0]) * nxt_pt.x
            + (nxt_perp_dir[1] + cur_perp_dir[1]) * nxt_pt.y;
            
            float A1 = 1.0f + pow(nxt_perp_dir[0] + cur_perp_dir[0], 2.0f);
            float B1 = (nxt_perp_dir[0] + cur_perp_dir[0]) * (nxt_perp_dir[1] + cur_perp_dir[1]);
            float C1 = (nxt_perp_dir[0] + cur_perp_dir[0])*K + orig_first_x;
            float B2 = 1.0f + pow(nxt_perp_dir[1] + cur_perp_dir[1], 2.0f);
            float A2 = B1;
            float C2 = (nxt_perp_dir[1] + cur_perp_dir[1])*K + orig_first_y;
            
            float bottom = A2*B1 - A1*B2;
            if (abs(bottom) > 1e-3) {
                float new_x = (B1*C2 - B2*C1) / bottom;
                float new_y = (A2*C1 - A1*C2) / bottom;
                float delta_x = new_x - cur_pt.x;
                float delta_y = new_y - cur_pt.y;
                
                cum_change += sqrt(delta_x*delta_x + delta_y*delta_y);
                
                cur_pt.x = new_x;
                cur_pt.y = new_y;
            }
            
            if (cum_change < 0.01f) {
                break;
            }
        }
    }

    center_line.clear();
    // Interpolate
    float resolution = 5.0f; // in meters 
    center_line.emplace_back(tmp_center_line[0]);
    for(int i = 1; i < tmp_center_line.size(); ++i){
        Eigen::Vector2d vec(tmp_center_line[i].x - tmp_center_line[i-1].x,
                                tmp_center_line[i].y - tmp_center_line[i-1].y);
        float delta_d = vec.norm();

        if(delta_d > resolution){
            int n_to_insert = floor(delta_d / resolution);
            float d = delta_d / (n_to_insert + 1);
            vec.normalize();
            for(int j = 0; j < n_to_insert; ++j){
                RoadPt new_pt(tmp_center_line[i-1]);
                new_pt.x = tmp_center_line[i-1].x + (j+1) * d * vec.x();
                new_pt.y = tmp_center_line[i-1].y + (j+1) * d * vec.y();
                center_line.emplace_back(new_pt);
            }
        }

        center_line.emplace_back(tmp_center_line[i]);
    }
}

void sampleGPSPoints(float                     radius,
                     float                     heading_threshold,
                     const PclPointCloud::Ptr& points,
                     const PclSearchTree::Ptr& search_tree,
                     PclPointCloud::Ptr&       new_points,
                     PclSearchTree::Ptr&       new_search_tree){
    if (radius < 1.0f) {
        cout << "WARNING from sampleGPSPoints: radius is too small" << endl;
        return;
    }

    // Sample points
    vector<bool> pt_covered(points->size(), false);
    for (size_t i = 0; i < points->size(); ++i) {
        if (pt_covered[i]) {
            continue;
        }
        
        PclPoint pt = points->at(i);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        search_tree->radiusSearch(pt,
                                  radius,
                                  k_indices,
                                  k_dist_sqrs);
        
        float cum_x   = 0.0f;
        float cum_y   = 0.0f;
        int cum_speed = 0;
        int n_count   = 0;
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = points->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
            if (delta_heading < heading_threshold) {
                cum_x           += nb_pt.x;
                cum_y           += nb_pt.y;
                cum_speed       += nb_pt.speed;
                n_count++;
                pt_covered[*it]  = true;
            }
        }
        
        if (n_count > 0) {
            PclPoint new_pt = pt;
            new_pt.x         = cum_x / n_count;
            new_pt.y         = cum_y / n_count;
            new_pt.id_sample = n_count;
            new_pt.speed     = cum_speed / n_count;
        
            new_points->push_back(new_pt);
        }
    }
    
    if (new_points->size() == 0) {
        return;
    }
    
    new_search_tree->setInputCloud(new_points);
}

void sampleRoadSkeletonPoints(float search_radius,
                              float heading_threshold,
                              float delta_perp_bin,
                              float sigma_perp_s,
                              float delta_head_bin,
                              float sigma_head_s,
                              int   min_n,
                              bool  is_oneway,
                              const PclPointCloud::Ptr& points,
                              const PclSearchTree::Ptr& search_tree,
                              PclPointCloud::Ptr& new_points,
                              PclSearchTree::Ptr& new_search_tree){
    if (search_radius > 250.0f) {
        cout << "Warning: search radius " << search_radius << "m is too big in computeRoadCenterAt()." << endl;
        return;
    }
    
    if(delta_perp_bin < 1e-3){
        cout << "ERROR: perpendicular delta_perp_bin" << delta_perp_bin << "m cannot be negative or too small in sampleRoadSkeletonPoints() ." << endl;
        return;
    }
    
    int N_PERP_BINS = ceil(2.0f * search_radius / delta_perp_bin);
    
    if (N_PERP_BINS > 1000) {
        cout << "Warning: delta_perp_bin = " << delta_perp_bin << "m might be too small with search radius = " << search_radius << "m in sampleRoadSkeletonPoints(). This will lead to " << N_PERP_BINS << " bins in histogram" << endl;
        return;
    }
    
    int perp_half_window = ceil(sigma_perp_s / delta_perp_bin);
    
    int N_HEAD_BINS = ceil(360.0f / delta_head_bin);
    
    // Sample points to speed up
    PclPointCloud::Ptr t_points(new PclPointCloud);
    PclSearchTree::Ptr t_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    vector<bool> pt_covered(points->size(), false);
    for (size_t i = 0; i < points->size(); ++i) {
        if (pt_covered[i]) {
            continue;
        }
        
        PclPoint pt = points->at(i);
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        search_tree->radiusSearch(pt,
                                  5.0f,
                                  k_indices,
                                  k_dist_sqrs);
        
        float cum_x = 0.0f;
        float cum_y = 0.0f;
        int n_count = 0;
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = points->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
            if (delta_heading < 0.5f * heading_threshold) {
                cum_x += nb_pt.x;
                cum_y += nb_pt.y;
                n_count++;
                pt_covered[*it] = true;
            }
        }
        
        if (n_count > 0) {
            PclPoint new_pt = pt;
            new_pt.x = cum_x / n_count;
            new_pt.y = cum_y / n_count;
            new_pt.id_sample = n_count;
            
            vector<int> t_inds;
            vector<float> t_dsqrs;
            search_tree->radiusSearch(new_pt,
                                      10.0f,
                                      t_inds,
                                      t_dsqrs);
            if (t_inds.size() >= 2) {
                t_points->push_back(new_pt);
            }
        }
    }
   
    if (t_points->size() == 0) {
        return;
    }
    
    t_search_tree->setInputCloud(t_points);
    
    // Make a tmp point cloud to corresponding road center
    PclPointCloud::Ptr tmp_points(new PclPointCloud);
    PclSearchTree::Ptr tmp_search_tree(new pcl::search::FlannSearch<PclPoint>(false));
    
    for (size_t i = 0; i < t_points->size(); ++i) {
        PclPoint& pt = t_points->at(i);
       
        Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        t_search_tree->radiusSearch(pt,
                                    0.6f * search_radius,
                                    k_indices,
                                    k_dist_sqrs);
        
        // Vote in the perpendicular direction
        vector<float> perp_votes(N_PERP_BINS, 0.0f);
        vector<float> head_votes(N_HEAD_BINS, 0.0f);
        int n_vote_pts = 0;
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nb_pt = t_points->at(*it);
            
            float d_heading = deltaHeading1MinusHeading2(nb_pt.head, pt.head);
            float delta_heading = abs(d_heading);
            
//            if (!is_oneway && delta_heading > 90.0f) {
//                delta_heading = 180.0f - delta_heading;
//            }
            
            // heading votes
            if(k_dist_sqrs[it - k_indices.begin()] < 100.0f){
                if (delta_heading < 3.0f * heading_threshold) {
                    // vote for heading
                    int modified_head = nb_pt.head;
                    if (!is_oneway && abs(d_heading) > 90.0f) {
                        modified_head = (modified_head + 180) % 360;
                    }
                    int head_bin_idx = floor(static_cast<float>(modified_head) / delta_head_bin + 0.5f);
                    for (int k = head_bin_idx-1; k <= head_bin_idx+1; ++k) {
                        int corresponding_k = k;
                        if (k < 0) {
                            corresponding_k += N_HEAD_BINS;
                        }
                        if (corresponding_k >= N_HEAD_BINS) {
                            corresponding_k %= N_HEAD_BINS;
                        }
                        
                        float head_bin_center = (corresponding_k + 0.5f) * delta_head_bin;
                        float d_head = head_bin_center - modified_head;
                        head_votes[corresponding_k] += nb_pt.id_sample * exp(-1.0f * d_head * d_head / 2.0f / delta_head_bin / delta_head_bin);
                    }
                }
            }
            
            if (delta_heading < heading_threshold) {
                // Perpendicular voting
                Eigen::Vector3d vec(nb_pt.x - pt.x,
                                    nb_pt.y - pt.y,
                                    0.0f);
                float perp_proj = pt_dir.cross(vec)[2];
                int perp_bin_idx = floor((perp_proj + search_radius) / delta_perp_bin + 0.5f);
                
                if (perp_bin_idx < 0 || perp_bin_idx >= N_PERP_BINS) {
                    continue;
                }
                
                n_vote_pts += nb_pt.id_sample;
                
                float base_vote = exp(-1.0f * perp_proj * perp_proj / 2.0f / 10.0f / 10.0f);
                
                for (int s = perp_bin_idx - perp_half_window; s <= perp_bin_idx + perp_half_window; ++s) {
                    if (s < 0 || s >= N_PERP_BINS) {
                        continue;
                    }
                    
                    float delta_bin_center = perp_proj + search_radius - (s + 0.5f) * delta_perp_bin;
                    
                    perp_votes[s] += nb_pt.id_sample * exp(-1.0f * delta_bin_center * delta_bin_center / 2.0f / sigma_perp_s / sigma_perp_s) * base_vote;
                }
            }
        }
        
        if(n_vote_pts >= min_n){
            int head_max_idx = -1;
            findMaxElement(head_votes, head_max_idx);
            int perp_max_idx = -1;
            findMaxElement(perp_votes, perp_max_idx);
            if (head_max_idx != -1 && perp_max_idx != -1) {
                float max_perp_dist = (perp_max_idx + 0.5f) * delta_perp_bin - search_radius;
                int max_head = floor((head_max_idx + 0.5f) * delta_head_bin);
                float d_head = abs(deltaHeading1MinusHeading2(pt.head, max_head));
                if (!is_oneway && d_head > 90.0f) {
                    d_head = 180.0f - d_head;
                }
                
                if (d_head < heading_threshold) {
                    Eigen::Vector2d perp_dir(-pt_dir[1], pt_dir[0]);
                    Eigen::Vector2d new_loc(pt.x, pt.y);
                    new_loc += max_perp_dist * perp_dir;
                    PclPoint new_pt = pt;
                    new_pt.x = new_loc.x();
                    new_pt.y = new_loc.y();
                    new_pt.head = max_head;
                    tmp_points->push_back(new_pt);
                }
            }
        }
    }
    
    tmp_search_tree->setInputCloud(tmp_points);
    
    // Sample the tmp_points
    new_points->clear();
    vector<bool> is_covered(tmp_points->size(), false);
    for (size_t i = 0; i < tmp_points->size(); ++i) {
        if (is_covered[i]) {
            continue;
        }
        
        PclPoint& pt = tmp_points->at(i);
        float pt_speed = pt.speed / 100.0f;
        if(pt_speed < 1.0f){
            continue;
        }
        
        vector<int> k_indices;
        vector<float> k_dist_sqrs;
        tmp_search_tree->radiusSearch(pt, 5.0f, k_indices, k_dist_sqrs);
        // Mark corresponding nearby points
        float cum_x = 0.0f;
        float cum_y = 0.0f;
        int n_count = 0;
        for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
            PclPoint& nearby_pt = tmp_points->at(*it);
            float delta_heading = abs(deltaHeading1MinusHeading2(nearby_pt.head, pt.head));
//            if(!is_oneway && delta_heading > 90.0f){
//                delta_heading = 180.0f - delta_heading;
//            }
            
            if (delta_heading < 7.5f) {
                ++n_count;
                cum_x += nearby_pt.x;
                cum_y += nearby_pt.y;
                is_covered[*it] = true;
            }
        }
        
        if (n_count > 0) {
            cum_x /= n_count;
            cum_y /= n_count;
            
            pt.x = cum_x;
            pt.y = cum_y;
            new_points->push_back(pt);
        }
    }
    
    if (new_points->size() > 0) {
        new_search_tree->setInputCloud(new_points);
    }
}

void adjustRoadPtHeading(RoadPt& r_pt,
                         PclPointCloud::Ptr& points,
                         PclSearchTree::Ptr& search_tree,
                         float search_radius,
                         float heading_threshold,
                         float delta_bin,
                         bool pt_id_sample_store_weight){
    
    int n_heading_bins = ceil(360.0f / delta_bin);
    
    PclPoint pt;
    pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
    
    vector<float> votes(n_heading_bins, 0.0f);
    vector<int> k_indices;
    vector<float> k_dist_sqrs;
    search_tree->radiusSearch(pt, search_radius, k_indices, k_dist_sqrs);
    for(vector<int>::iterator vit = k_indices.begin(); vit != k_indices.end(); ++vit){
        PclPoint& nb_pt = points->at(*vit);
        float d_heading = deltaHeading1MinusHeading2(nb_pt.head, r_pt.head);
        if (abs(d_heading) < heading_threshold) {
            int bin_idx = floor(nb_pt.head / delta_bin);
            for (int k = bin_idx - 1; k < bin_idx + 1; ++k) {
                int corresponding_k = k;
                if (k < 0) {
                    corresponding_k += n_heading_bins;
                }
                if (k >= n_heading_bins) {
                    corresponding_k %= n_heading_bins;
                }
                
                float bin_center = (corresponding_k + 0.5f) * delta_bin;
                float d = bin_center - r_pt.head;
                if(pt_id_sample_store_weight){
                    votes[corresponding_k] += nb_pt.id_sample * exp(-1.0f * d * d / 2.0f / delta_bin / delta_bin);
                }
                else{
                    votes[corresponding_k] += exp(-1.0f * d * d / 2.0f / delta_bin / delta_bin);
                }
            }
        }
    }
    
    int max_idx = -1;
    findMaxElement(votes, max_idx);
    if (max_idx != -1) {
        r_pt.head = floor((max_idx + 0.5f) * delta_bin);
    }
}

void adjustRoadCenterAt(RoadPt&             r_pt,
                        PclPointCloud::Ptr& points,
                        PclSearchTree::Ptr& search_tree,
                        float               trajecotry_avg_speed,
                        float               search_radius,
                        float               heading_threshold,
                        float               delta_bin,
                        float               sigma_s,
                        bool                pt_id_sample_store_weight){
    /*
     *We will do a parallel bin voting to determine the true location of the road and the width of the road
     */

    PclPoint pt;
    pt.setCoordinate(r_pt.x, r_pt.y, 0.0f);
    pt.head = r_pt.head;
    
    if (search_radius > 250.0f) {
        cout << "Warning: search radius " << search_radius << "m is too big in computeRoadCenterAt()." << endl;
        return;
    }
    
    if(delta_bin < 1e-3){
        cout << "ERROR: delta_bin " << delta_bin << "m cannot be negative or too small in computeRoadCenterAt() ." << endl;
        return;
    }
    
    int N_BINS = ceil(2.0f * search_radius / delta_bin);
    
    if (N_BINS > 1000) {
        cout << "Warning: delta_bin = " << delta_bin << "m might be too small with search radius = " << search_radius << "m in computeRoadCenterAt(). This will lead to " << N_BINS << " bins in histogram" << endl;
        return;
    }
    
    if (N_BINS < 5) {
        cout << "Warning: delta_bin = " << delta_bin << "m might be too big with search radius = " << search_radius << "m in computeRoadCenterAt(). This will lead to " << N_BINS << " bins in histogram" << endl;
        return;
    }
    
    int half_window = ceil(sigma_s / delta_bin);
    
    Eigen::Vector3d pt_dir = headingTo3dVector(pt.head);
    Eigen::Vector3d pt_perp_dir(-pt_dir[1], pt_dir[0], 0.0f);
    
    vector<int> k_indices;
    vector<float> k_dist_sqrs;
    
    search_tree->radiusSearch(pt,
                              search_radius,
                              k_indices,
                              k_dist_sqrs);
    
    vector<float> votes(N_BINS, 0.0f);
    
    for (vector<int>::iterator it = k_indices.begin(); it != k_indices.end(); ++it) {
        PclPoint& nb_pt = points->at(*it);
        float delta_heading = abs(deltaHeading1MinusHeading2(nb_pt.head, pt.head));
        
        if (delta_heading < heading_threshold) {
            Eigen::Vector3d vec(nb_pt.x - pt.x,
                                nb_pt.y - pt.y,
                                0.0f);
            float perp_proj = pt_dir.cross(vec)[2];
            int bin_idx = floor((perp_proj + search_radius) / delta_bin);
            
            if (bin_idx < 0 || bin_idx >= N_BINS) {
                continue;
            }
            
            for (int s = bin_idx - half_window; s <= bin_idx + half_window; ++s) {
                if (s < 0 || s >= N_BINS) {
                    continue;
                }
                
                float delta_bin_center = perp_proj + search_radius - s * delta_bin;
                
                float adjusted_sigma_s = sigma_s;
                if(nb_pt.speed < 1.5f * trajecotry_avg_speed)
                    adjusted_sigma_s = sigma_s * trajecotry_avg_speed / (nb_pt.speed + 0.1f);

                if (pt_id_sample_store_weight) {
                    votes[s] += nb_pt.id_sample * exp(-1.0f * delta_bin_center * delta_bin_center / 2.0f / adjusted_sigma_s / adjusted_sigma_s);
                }
                else{
                    votes[s] += exp(-1.0f * delta_bin_center * delta_bin_center / 2.0f / adjusted_sigma_s / adjusted_sigma_s);
                }
            }
        }
    }
    
    int max_idx = -1;
    vector<int> peak_idxs;
    int window_size = floor(3.7f * 4 / delta_bin);

    peakDetector(votes,
                 window_size,
                 1.5f,
                 peak_idxs,
                 false);

    if(peak_idxs.size() > 0){
        float closest_dist = POSITIVE_INFINITY;
        for (const auto& idx : peak_idxs) { 
            float delta_dist = abs(idx * delta_bin - search_radius);
            if(delta_dist < closest_dist){
                closest_dist = delta_dist;
                max_idx = idx;
            } 
        } 
    } 
    
    float width_ratio = 0.8f;
    if(max_idx != -1){
        float avg_perp_dist = (max_idx + 0.5f) * delta_bin - search_radius;
        Eigen::Vector2d perp_dir_2d(pt_perp_dir.x(), pt_perp_dir.y());
        Eigen::Vector2d loc = avg_perp_dist * perp_dir_2d;
        r_pt.x += loc.x();
        r_pt.y += loc.y();
        
        // Update road width
        int left_border = max_idx;
        while (left_border >= 0) {
            if (votes[left_border] < width_ratio * votes[max_idx]) {
                break;
            }
            left_border--;
        }
        if(left_border < 0){
            left_border = 0;
        }
        
        int right_border = max_idx;
        while (right_border < N_BINS) {
            if (votes[right_border] < width_ratio * votes[max_idx]) {
                break;
            }
            right_border++;
        }
        if(right_border >= N_BINS){
            right_border = N_BINS-1;
        }
        
        float road_width = (right_border - left_border + 1) * LANE_WIDTH;
        if (road_width < LANE_WIDTH) {
            road_width = LANE_WIDTH;
        }
        r_pt.n_lanes = floor(road_width / LANE_WIDTH + 0.5f);
    }
}

void SceneConst::updateAttr(){
    delta_x_ = bound_box_[1] - bound_box_[0];
    delta_y_ = bound_box_[3] - bound_box_[2];
    
    center_x_ = 0.5*bound_box_[0] + 0.5*bound_box_[1];
    center_y_ = 0.5*bound_box_[2] + 0.5*bound_box_[3];
   
    scale_factor_ = (delta_x_ > delta_y_) ? 0.5*delta_x_ : 0.5*delta_y_;
}

Vertex SceneConst::normalize(float x, float y, float z){
    if (delta_x_ < 0 || delta_y_ < 0){
        fprintf(stderr, "Trajectory bounding box error! Min greater than Max!\n");
        exit(1);
    }
   
    Vertex normalized_loc = Vertex(0.0f, 0.0f, z);
    normalized_loc.x = (x - center_x_) / scale_factor_;
    normalized_loc.y = (y - center_y_) / scale_factor_;
    return normalized_loc;
}

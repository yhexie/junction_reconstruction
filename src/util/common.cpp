#include <sstream>
#include <iomanip>

#include "common.h"
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

void peakDetector(vector<float>& hist, int window, float ratio, vector<int>& peak_idxs,bool is_closed){
    // Detecting peaks in a histogram
    //is_closed: true - loop around; false: do not loop around.
    
    peak_idxs.clear();
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
                    peak_idxs.push_back(i);
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
                    peak_idxs.push_back(i);
                }
            }
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
        cout << "Error when converting heading to 2d vector. Invalid input.!" << endl;
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
    
    vector<float> orig_xs;
    vector<float> orig_ys;
    for (int i = 0; i < center_line.size(); ++i) {
        orig_xs.push_back(center_line[i].x);
        orig_ys.push_back(center_line[i].y);
    }
    
    if(fix_front){
        float orig_last_x = center_line.back().x;
        float orig_last_y = center_line.back().y;
        int max_iter = 100;
        int i_iter = 0;
        while(i_iter <= max_iter){
            i_iter++;
            float cum_change = 0.0f;
            for(int i = 1; i < center_line.size() - 1; ++i){
                RoadPt& cur_pt = center_line[i];
                RoadPt& prev_pt = center_line[i-1];
                RoadPt& nxt_pt = center_line[i+1];
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
            int last_idx = center_line.size()-1;
            RoadPt& cur_pt = center_line[last_idx];
            RoadPt& prev_pt = center_line[last_idx-1];
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
            
            if (cum_change < 0.1f) {
                break;
            }
        }
    }
    else{
        float orig_first_x = center_line.front().x;
        float orig_first_y = center_line.front().y;
        int max_iter = 100;
        int i_iter = 0;
        while(i_iter <= max_iter){
            i_iter++;
            float cum_change = 0.0f;
            
            for(int i = center_line.size() - 2; i > 0; --i){
                RoadPt& cur_pt = center_line[i];
                RoadPt& prev_pt = center_line[i-1];
                RoadPt& nxt_pt = center_line[i+1];
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
            RoadPt& cur_pt = center_line[0];
            RoadPt& nxt_pt = center_line[1];
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
            
            if (cum_change < 0.1f) {
                break;
            }
        }
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
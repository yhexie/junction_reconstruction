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
        cout << "Error when converting heading to 3d vector. Invalid input.!" << endl;
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
    
    return (orig_heading + delta_heading) % 360;
}

int decreaseHeadingBy(int delta_heading,
                      const int orig_heading){
    if (orig_heading < 0 || orig_heading > 360) {
        cout << "Error (from decreaseHeadingBy). Invalid input.!" << endl;
        return -1;
    }
    
    return (orig_heading - delta_heading + 360) % 360;
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
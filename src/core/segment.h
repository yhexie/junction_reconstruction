#ifndef SEGMENT_H_
#define SEGMENT_H_

#include <vector>
using namespace std;

struct SegPoint {
    float x;
    float y;
    float t;
    int orig_idx;
    
    SegPoint() {x = -1.0f; y = -1.0f; t = -1.0f;}
    SegPoint(float vx, float vy, float vt, int vidx = -1) {x = vx; y = vy; t = vt; orig_idx = vidx;}
};


class Segment {
public:
    Segment();
    ~Segment();
    
    vector<SegPoint> &points() {return points_;}
    float &segLength() {return segLength_;}
    
    float distanceTo(Segment &);

private:
    vector<SegPoint> points_; // Relative to Trajectory.data
    float segLength_;
};

#endif // SEGMENT_H_
#ifndef SEGMENT_H_
#define SEGMENT_H_

#include <vector>
using namespace std;

struct SegPoint {
    float x;
    float y;
    float t;
    int head;
    int speed;
    int orig_idx;
    
    SegPoint() {x = -1.0f; y = -1.0f; t = -1.0f;}
    SegPoint(float vx, float vy, float vt, int vhead = 0, int vspeed = 0, int vidx = -1) {x = vx; y = vy; t = vt; orig_idx = vidx; head = vhead; speed = vspeed;}
};


class Segment {
public:
    Segment();
    ~Segment();
    
    vector<SegPoint> &points() {return points_;}
    float &segLength() {return segLength_;}
    float distanceTo(Segment &);
    bool &quality() {return quality_;}

private:
    vector<SegPoint> points_; // Relative to Trajectory.data
    float segLength_;
    bool quality_;  // Good: true; Bad: false
};

#endif // SEGMENT_H_
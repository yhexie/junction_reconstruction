#include "segment.h"
#include "cgal_types.h"
#include <CGAL/squared_distance_2.h>

Segment::Segment(){
    points_.clear();
    segLength_ = 0.0f;
    quality_ = true;
}

Segment::~Segment(){
    points_.clear();
}

float Segment::distanceTo(Segment &another_seg){
    // Compute Hausdorff distance between this segment and another segment
    if (another_seg.points().size() == 1 && points_.size() == 1) {
        CgalPoint2 pt1 = CgalPoint2(points_[0].x, points_[0].y);
        CgalPoint2 pt2 = CgalPoint2(another_seg.points()[0].x, another_seg.points()[0].y);
        float squared_dist = CGAL::squared_distance(pt1, pt2);
        return sqrt(squared_dist);
    }
    
    float sqrt_distance1 = 0.0f;
    if (another_seg.points().size() > 1) {
        for (size_t pt_idx = 0; pt_idx < points_.size(); ++pt_idx) {
            CgalPoint2 pt = CgalPoint2(points_[pt_idx].x, points_[pt_idx].y);
            float min_dist = 1e9;
            for (size_t other_pt_idx = 0; other_pt_idx < another_seg.points().size() -1; ++other_pt_idx) {
                CgalPoint2 pt_start = CgalPoint2(another_seg.points()[other_pt_idx].x, another_seg.points()[other_pt_idx].y);
                CgalPoint2 pt_end = CgalPoint2(another_seg.points()[other_pt_idx+1].x, another_seg.points()[other_pt_idx+1].y);
                CgalSegment2 segment = CgalSegment2(pt_start, pt_end);
                float squared_dist = CGAL::squared_distance(pt, segment);
                if (min_dist > squared_dist) {
                    min_dist = squared_dist;
                }
            }
            if (sqrt_distance1 < min_dist) {
                sqrt_distance1 = min_dist;
            }
        }
    }
   
    float sqrt_distance2 = 0.0f;
    if (points_.size() > 1) {
        for (size_t pt_idx = 0; pt_idx < another_seg.points().size(); ++pt_idx) {
            CgalPoint2 pt = CgalPoint2(another_seg.points()[pt_idx].x, another_seg.points()[pt_idx].y);
            float min_dist = 1e9;
            for (size_t other_pt_idx = 0; other_pt_idx < points_.size() -1; ++other_pt_idx) {
                CgalPoint2 pt_start = CgalPoint2(points_[other_pt_idx].x, points_[other_pt_idx].y);
                CgalPoint2 pt_end = CgalPoint2(points_[other_pt_idx+1].x, points_[other_pt_idx+1].y);
                CgalSegment2 segment = CgalSegment2(pt_start, pt_end);
                float squared_dist = CGAL::squared_distance(pt, segment);
                if (min_dist > squared_dist) {
                    min_dist = squared_dist;
                }
            }
            if (sqrt_distance2 < min_dist && points_.size() > 1) {
                sqrt_distance2 = min_dist;
            }
        }
    }
    
    float sqrt_distance = (sqrt_distance1 > sqrt_distance2) ? sqrt_distance1 : sqrt_distance2;
    return sqrt(sqrt_distance);
}

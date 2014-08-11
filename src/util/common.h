#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <vector>

enum TrajectoryColorMode{
    UNIFORM,
    TRAJECTORY,
    SAMPLE_TIME,
    SAMPLE_ORDER
};

enum ToolMode{
    NO_ACTIVE_TOOL,
    ADD_VALVE
};

namespace Common{
    std::string int2String(int i, int width);
    void randomK(std::vector<int>& random_k, int k, int N);
}



#endif //COMMON_H_
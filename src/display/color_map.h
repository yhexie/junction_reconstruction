#ifndef ColorMap_H_
#define ColorMap_H_

#include <vector>
#include <cmath>
#include <iostream>

struct Color{
    float r, g, b, alpha;
    Color(float vr, float vg, float vb, float valpha) : r(vr), g(vg), b(vb), alpha(valpha){}
    Color(){ r = 0.0f; g = 0.0f; b = 0.0f; alpha = 0.0f;}
};

class ColorMap {
public:
    static ColorMap& getInstance() {
        static ColorMap singleton_;
        return singleton_;
    }
    
    typedef enum {LIGHT_BLUE, DARK_GRAY, LIGHT_GRAY, ORANGE} NamedColor;
    
    /* more (non-static) functions here */
    const Color& getNamedColor(NamedColor named_color);
    const Color& getDiscreteColor(int idx);
    Color getContinusColor(float value, float low=0.0f, float high=1.0f, bool interpolate=false);
    
private:
    ColorMap();                            // ctor hidden
    ColorMap(ColorMap const&);           // copy ctor hidden
    ColorMap& operator=(ColorMap const&); // assign op. hidden
    virtual ~ColorMap(){}                          // dtor hidden
    
    std::vector<Color> continuous_;
    Color light_blue_;
    Color dark_gray_;
    Color light_gray_;
    Color orange_;
    std::vector<Color> discrete_;
};

#endif // ColorMap_H_
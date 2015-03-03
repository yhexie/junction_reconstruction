#ifndef RENDERABLE_H_
#define RENDERABLE_H_

#include <QObject>
#include <QtOpenGL>
#include "customized_shader_program.h"

class Renderable : public QObject{
public:
    Renderable(QObject *parent);
    void setShadderProgram(CustomizedShaderProgram *shadder) {shadder_program_ = shadder;}
    virtual ~Renderable();
    virtual void draw();
    
protected:
    QVector4D                           bound_box_; //[min_easting, max_easting, min_northing, max_northing]
    QOpenGLBuffer                       vertexPositionBuffer_;
    QOpenGLBuffer                       vertexColorBuffer_;
    CustomizedShaderProgram*            shadder_program_;
};

#endif //RENDERABLE_H_ 

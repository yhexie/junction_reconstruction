#ifndef RENDERABLE_H_
#define RENDERABLE_H_

#include <QObject>
#include <QtOpenGL>
#include "customized_shader_program.h"

class Renderable : public QObject{
public:
    Renderable(QObject *parent);
    void setShaderProgram(std::shared_ptr<CustomizedShaderProgram> shader) {shader_program_ = shader;}

    virtual ~Renderable();
    virtual void draw();
    
protected:
    QVector4D                           bound_box_; //[min_easting, max_easting, min_northing, max_northing]
    QOpenGLBuffer                       vertexPositionBuffer_;
    QOpenGLBuffer                       vertexColorBuffer_;
    std::shared_ptr<CustomizedShaderProgram>  shader_program_;
};

#endif //RENDERABLE_H_ 

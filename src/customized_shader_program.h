#ifndef CUSTOMIZED_SHADER_PROGRAM_H_
#define CUSTOMIZED_SHADER_PROGRAM_H_

#include <QtOpenGL>
#include <QMatrix4x4>

class CustomizedShaderProgram : public QOpenGLShaderProgram{
public:
    CustomizedShaderProgram(QObject *parent);
    ~CustomizedShaderProgram();
    void setupPositionAttributes();
    void setupColorAttributes();
    void setUniformMatrix(QMatrix4x4 matrix);
private:
    GLuint m_matrixUniform;
};

#endif //CUSTOMIZED_SHADER_PROGRAM_H_
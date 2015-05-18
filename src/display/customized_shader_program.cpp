#include "customized_shader_program.h"

static const char *vertexShaderSource =
"attribute highp vec4 posAttr;\n"
"attribute lowp vec4 colAttr;\n"
"varying lowp vec4 col;\n"
"uniform highp mat4 matrix;\n"
"void main() {\n"
"   col = colAttr;\n"
"   gl_Position = matrix * posAttr;\n"
"}\n";

static const char *fragmentShaderSource =
"varying lowp vec4 col;\n"
"void main() {\n"
"   gl_FragColor = col;\n"
"}\n";

CustomizedShaderProgram::CustomizedShaderProgram(QObject *object){
    addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource);
    addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource);
}

CustomizedShaderProgram::~CustomizedShaderProgram(){
    
}

void CustomizedShaderProgram::setUniformMatrix(QMatrix4x4 matrix){
    m_matrixUniform = uniformLocation("matrix");
    setUniformValue(m_matrixUniform, matrix);
}

void CustomizedShaderProgram::setupPositionAttributes(){
    enableAttributeArray("posAttr");
    setAttributeBuffer("posAttr", GL_FLOAT, 0, 3);
}

void CustomizedShaderProgram::setupColorAttributes(){
    enableAttributeArray("colAttr");
    setAttributeBuffer("colAttr", GL_FLOAT, 0, 4);
}

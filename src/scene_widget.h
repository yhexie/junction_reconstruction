#ifndef SCENE_WIDGET_H_
#define SCENE_WIDGET_H_

#include <QGLWidget>
#include <QtOpenGL>
#include <QVector2D>
#include <QQuaternion>
#include <QMatrix4x4>
#include <QBasicTimer>
#include "customized_shader_program.h"

class QMenu;
class Trajectories;

class SceneWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    explicit SceneWidget(QWidget *parent=0, const QGLWidget *shareWidget=0, Qt::WindowFlags f=0);
    ~SceneWidget();
    
    QSize sizeHint() const {return QSize(256, 256);}
signals:
    void trajFileLoaded(QString &filename);
    public slots:
    void slotOpenTrajectories(void);
    void slotSaveTrajectories(void);
    void slotToggleTrajectories(void);
    void slotColorizeUniform(void);
    void slotColorizeSampleTime(void);
    void slotColorizeSampleOrder(void);
    //void slotOpenOsmFile(QModelIndex index);
    
protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    virtual void prepareContextMenu(QMenu *menu);
    
private:
    Trajectories                        *trajectories_;
    CustomizedShaderProgram                  * m_program_;
    QMatrix4x4                          matrix_;
    
    QPoint                             lastPos;
    float                              scaleFactor_;
    float                              scaleStep_;
    
    QMenu               *right_click_menu_;
    
    QOpenGLBuffer       test_buffer;
    QOpenGLBuffer       test_color_buffer;
    int xRot;
    int yRot;
    int zRot;
    int i_;
    
    void clearMeshModels(void);
    void updateInformation(void);
};


#endif //SCENE_WIDGET_H_
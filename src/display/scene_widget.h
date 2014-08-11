#ifndef SCENE_WIDGET_H_
#define SCENE_WIDGET_H_

#include <QGLWidget>
#include <QtOpenGL>
#include <QVector2D>
#include <QQuaternion>
#include <QMatrix4x4>
#include <QBasicTimer>
#include "customized_shader_program.h"

using namespace std;

class QMenu;
class Trajectories;
class OpenStreetMap;

class SceneWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    explicit SceneWidget(QWidget *parent=0, const QGLWidget *shareWidget=0, Qt::WindowFlags f=0);
    ~SceneWidget();
    
    QSize sizeHint() const {return QSize(256, 256);}
    void toggleSelectionMode();
    bool getSelectionMode() { return selection_mode_;}
    void toggleTrajectories(void);
    
signals:
    void trajFileLoaded(QString &filename);
    void osmFileLoaded(QString &filename);
    
    public slots:
    void slotOpenTrajectories(void);
    void slotSaveTrajectories(void);
    void slotOpenOsmMap(void);
    void slotColorizeUniform(void);
    void slotColorizeSampleTime(void);
    void slotColorizeSampleOrder(void);
    void resetView(void);
    void clearData(void);
    void slotExtractTrajectories(void);
    //void slotOpenOsmFile(QModelIndex index);
    
protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    
    
    void setXRotation(int angle);
    void setYRotation(int angle);
    
    void setXTranslate(int deltaX);
    void setYTranslate(int deltaY);
    
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    virtual void prepareContextMenu(QMenu *menu);
    
    void updateSceneBoundary(QVector4D bound_box_to_insert);
    void setSceneBoundary(QVector4D new_bound);
    void updateMaxScaleFactor();
    
private:
    CustomizedShaderProgram             *m_program_;
    QMatrix4x4                          view_matrix_;
    int xRot;
    int yRot;
    int xTrans;
    int yTrans;
    
    // Scene boundary box, [min_easting, max_easting, min_northing, max_northing]
    QVector4D                           bound_box_;
    
    // Trajectory container
    Trajectories                        *trajectories_;
    
    // OpenStreetMap container
    OpenStreetMap                       *osmMap_;
    
    // Visualization mode
    bool                                selection_mode_;
    
    // Mouse movement
    QPoint                              lastPos;
    float                               scaleFactor_;
    float                               maxScaleFactor_;
    float                               zoomTransX_;
    float                               zoomTransY_;
    QMenu                               *right_click_menu_;
    
    void updateInformation(void);
};


#endif //SCENE_WIDGET_H_
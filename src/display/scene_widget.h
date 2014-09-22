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
class Samples;
class OpenStreetMap;
class Graph;

class SceneWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    explicit SceneWidget(QWidget *parent=0, const QGLWidget *shareWidget=0, Qt::WindowFlags f=0);
    ~SceneWidget();
    
    QSize sizeHint() const {return QSize(256, 256);}
    void toggleSelectionMode();
    void enableSelectionMode();
    bool getSelectionMode() { return selection_mode_;}
    void toggleTrajectories(void);
    void drawSelectedTraj(vector<int> &idx);
    
signals:
    // Trajectories
    void trajFileLoaded(QString &filename, const size_t &numTraj, const size_t &numPoint);
    void nClusterComputed(int &n_cluster);
    
    // Map
    void osmFileLoaded(QString &filename);
    
    // Samples
    void newSamplesDrawn(QString &);
    
    // Segments
    void newSegmentsComputed(QString &);
    
    // Graph
    void newGraphComputed(QString &);
    
    public slots:
    // Trajectories
    void slotOpenTrajectories(void);
    void slotOpenTrajectoriesFromFile(const QString &filename);
    void slotSaveTrajectories(void);
    void slotExtractTrajectories(void);
    void slotComputePointCloudVariance(void);
    void slotSetShowDistanceGraph(int state);
    void slotComputeDistanceGraph();
    
    // Map
    void slotOpenOsmMap(void);
    void slotOpenOsmMapFromFile(const QString &filename);
    void slotSetShowMap(int state);
    
    // Samples
    void slotSamplePointCloud(void);
    void slotEnterSampleSelectionMode(void);
    void slotClearPickedSamples(void);
    void slotSetShowSamples(int);
    void slotClusterSegmentsAtSample(void);
    void slotClusterSegmentsAtAllSamples(void);
    void slotPickClusterAtIdx(int);
    
    // Segments
    void slotGenerateSegments(void);
    void slotSampleCoverDistanceChange(double);
    void slotDrawSegmentAndShortestPathInterpolation(int seg_idx);
    void slotSetShowSegments(int);
    void slotInterpolateSegments(void);
    
    // Graph
    void slotInitializeGraph(void);
    void slotSetShowGraph(int state);
    void slotUpdateGraph();
    
    // Others
    void resetView(void);
    void clearData(void);
    
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
    bool                                sample_selection_mode;
    
    // OpenStreetMap container
    OpenStreetMap                       *osmMap_;
    bool                                show_map_;
    
    // Graph container
    bool                                show_graph_;
    Graph                               *graph_;
    
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
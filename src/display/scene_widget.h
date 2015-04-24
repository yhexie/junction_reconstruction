#ifndef SCENE_WIDGET_H_
#define SCENE_WIDGET_H_

#include <QGLWidget>
#include <QtOpenGL>
#include <QVector2D>
#include <QQuaternion>
#include <QMatrix4x4>
#include <QBasicTimer>
#include "customized_shader_program.h"

#include "features.h"

using namespace std;
class QMenu;
class Trajectories;
class RoadGenerator;
class Samples;
class OpenStreetMap;
class Graph;

class SceneWidget : public QGLWidget
{
    Q_OBJECT
    
public:
    SceneWidget(QWidget *parent=0, const QGLWidget *shareWidget=0, Qt::WindowFlags f=0);
    ~SceneWidget();
    
    void toggleSelectionMode();
    void enableSelectionMode();
    bool getSelectionMode() { return selection_mode_;}
    void toggleTrajectories(void);
    void drawSelectedTraj(vector<int> &idx);
    
signals:
    // Mouse Clicked Signal
    void mouseClickedAt(float &x, float &y); // x, y are converted coordinates using CGAL intersection
    
    // Trajectories
    void trajFileLoaded(QString &filename, const size_t &numTraj, const size_t &numPoint);
    void nClusterComputed(int &n_cluster);
    void nPathletSelected(int &n_pathlet);
    void nDBSCANClusterComputed(int &n_dbscan_cluster);
    
    // Map
    void osmFileLoaded(QString &filename);
    
    // Samples
    void newSamplesDrawn(QString &);
    
    // Graph
    void newGraphComputed(QString &);
    
    public slots:
    // Trajectories
    void slotOpenTrajectories(void);
    void slotOpenTrajectoriesFromFile(const QString &filename);
    void slotSaveTrajectories(void);
    void slotExtractTrajectories(void);
    void slotSetShowDirection(int state);
    
        // Samples
    void slotSamplePointCloud(void);
    void slotEnterSampleSelectionMode(void);
    void slotClearPickedSamples(void);
    void slotSetShowSamples(int);
    
    // Map
    void slotOpenOsmMap(void);
    void slotOpenOsmMapFromFile(const QString &filename);
    void slotSetShowMap(int state);
    void slotExtractMapBranchingPoints();
    
    // Road Generator
    void slotRoadGeneratorLoadQueryInitClassifier();
    void slotRoadGeneratorApplyQueryInitClassifier();
    void slotRoadGeneratorSaveQueryInitResult();
    void slotRoadGeneratorLoadQueryInitResult();
    
    void slotRoadGeneratorLoadQueryQClassifier();
    void slotRoadGeneratorComputeInitialRoadGuess();
    void slotRoadGeneratorAddInitialRoad();
    void slotRoadGeneratorTmp();
    
    void slotRoadGeneratorDevelopRoadNetwork();
    void slotRoadGeneratorLocalAdjustment();
    void slotRoadGeneratorMCMCOptimization();
    
    // Features
    void slotExtractQueryInitTrainingSamplesFromMap();
    void slotLoadQueryInitTrainingSamples();
    void slotSaveQueryInitTrainingSamples();
    void slotTrainQueryInitClassifier();
    void slotSaveQueryInitClassifer();
    
    void slotExtractQueryQTrainingSamplesFromMap();
    void slotLoadQueryQTrainingSamples();
    void slotSaveQueryQTrainingSamples();
    void slotTrainQueryQClassifier();
    void slotSaveQueryQClassifer();
    
    // Parameters
    void slotParameterSearchRadiusChanged(double);
    void slotParameterGpsErrorSigmaChanged(double);
    void slotParameterGpsErrorHeadinbgChanged(double);
    
    void slotParameterDeltaGrowingLengthChanged(double);
    void slotParameterBranchPredictorExtensionRatioChanged(double);
    void slotParameterBranchPredictorMaxTExtension(double);
    
    // Others
    void slotClearAll(void);
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
    
    void updateSceneBoundary();
    void setSceneBoundary(QVector4D new_bound);
    void updateMaxScaleFactor();
    
private:
    CustomizedShaderProgram             *m_program_;
    QMatrix4x4                          view_matrix_;
    int xRot;
    int yRot;
    int xTrans;
    int yTrans;
    
    // Trajectory container
    Trajectories                        *trajectories_;
    bool                                sample_selection_mode;
    
    // OpenStreetMap container
    OpenStreetMap                       *osmMap_;
    bool                                show_map_;
    
    // Road Generator
    RoadGenerator                       *road_generator_;
    
    // Graph container
    bool                                show_graph_;
    // Feature selection
    bool                                feature_selection_mode_;
    
    QueryInitFeatureSelector            *query_init_feature_selector_;
    QueryQFeatureSelector               *query_q_feature_selector_;
    
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
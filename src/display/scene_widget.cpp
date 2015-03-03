#include "scene_widget.h"
#include <ctime>
#include <cmath>
#include <QDebug>
#include <QMenu>
#include <QMouseEvent>
#include <QColor>
#include <array>
#include "cgal_types.h"
#include <CGAL/intersections.h>
#include "trajectories.h"
#include "road_generator.h"
#include "openstreetmap.h"
#include "renderable.h"
#include "color_map.h"
#include "main_window.h"
#include "common.h"

SceneWidget::SceneWidget(QWidget * parent, const QGLWidget * shareWidget, Qt::WindowFlags f)
:QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    view_matrix_.setToIdentity();
    xRot = 0;
    yRot = 0;
    xTrans = 0;
    yTrans = 0;
    
    SceneConst::getInstance().setBoundBox(QVector4D(1e10, -1e10, 1e10, -1e10));
    
    trajectories_               = new Trajectories(this);
    road_generator_             = new RoadGenerator(this);
    feature_selection_mode_     = false;
    query_init_feature_selector_ = new QueryInitFeatureSelector(this);
    query_q_feature_selector_ = new QueryQFeatureSelector(this);
    
    
    sample_selection_mode       = false;
    osmMap_                     = new OpenStreetMap(this);
//    graph_                      = new Graph(this);
    selection_mode_             = false;
    
    scaleFactor_                = 1.0;
    maxScaleFactor_             = 1.0;
    zoomTransX_                 = 0.0f;
    zoomTransY_                 = 0.0f;
    
    right_click_menu_           = new QMenu;
    right_click_menu_->addAction("Open trajectories", this, SLOT(slotOpenTrajectories()));
    right_click_menu_->addAction("Save trajectories", this, SLOT(slotSaveTrajectories()));
    right_click_menu_->addSeparator();
    right_click_menu_->addAction("Open OSM", this, SLOT(slotOpenOsmMap()));
    
    show_map_ = true;
//    show_graph_ = true;
}

SceneWidget::~SceneWidget()
{
    delete m_program_;
    delete right_click_menu_;
    delete trajectories_;
    delete road_generator_;
    delete query_init_feature_selector_;
    delete query_q_feature_selector_;
    delete osmMap_;
    //delete graph_;
}

void SceneWidget::toggleSelectionMode(){
    selection_mode_ = !selection_mode_;
    trajectories_->setSelectionMode(selection_mode_);
    updateGL();
}

void SceneWidget::enableSelectionMode(){
    selection_mode_ = true;
    trajectories_->setSelectionMode(selection_mode_);
    updateGL();
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

void SceneWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
    if (event->button() == Qt::LeftButton) {
        float tx = static_cast<float>(event->x()) / width()*2 - 1;
        float ty = -1*static_cast<float>(event->y()) / height()*2 + 1;
        
        QVector3D near_point = view_matrix_.inverted().map(QVector3D(tx, ty, 0.0));
        QVector3D far_point = view_matrix_.inverted().map(QVector3D(tx, ty, 1.0));
        
        CgalLine line = CgalLine(Vec3Caster<QVector3D, CgalPoint>(near_point), Vec3Caster<QVector3D, CgalPoint>(far_point));
        
        CgalPlane plane = CgalPlane(CgalPoint(0,0,0), CgalVector(0,0,1));
        
        CGAL::Object intersection = CGAL::intersection(line, plane);
        if (const CgalPoint *intersection_point = CGAL::object_cast<CgalPoint>(&intersection)){
            if(fabs(intersection_point->x()) > 1.0 || fabs(intersection_point->y()) > 1.0){
                return;
            }
           
            QVector4D bound_box = SceneConst::getInstance().getBoundBox();
            
            float center_x = 0.5*bound_box[0] + 0.5*bound_box[1];
            float center_y = 0.5*bound_box[2] + 0.5*bound_box[3];
            float delta_x = bound_box[1] - bound_box[0];
            float delta_y = bound_box[3] - bound_box[2];
            float scale_factor_ = (delta_x > delta_y) ? 0.5*delta_x : 0.5*delta_y;
            
            float intersect_x = center_x + intersection_point->x()*scale_factor_;
            float intersect_y = center_y + intersection_point->y()*scale_factor_;
            if(sample_selection_mode){
                trajectories_->pickAnotherSampleNear(intersect_x, intersect_y);
                sample_selection_mode = false;
            }
            else{
                if (selection_mode_){
                    trajectories_->selectNearLocation(intersect_x, intersect_y);
                }
            }
            emit mouseClickedAt(intersect_x, intersect_y);
            updateGL();
        }
    }
}

void SceneWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (event->buttons() & Qt::RightButton) {
        int dx = event->x() - lastPos.x();
        int dy = event->y() - lastPos.y();
        //setXRotation(xRot + 8*dx);
        //setYRotation(yRot + 8*dy);
        setXTranslate(xTrans + dx);
        setYTranslate(yTrans - dy);
    }
//    else if(event->buttons() & Qt::LeftButton){
//        int dx = event->x() - lastPos.x();
//        int dy = event->y() - lastPos.y();
//        setXTranslate(xTrans + dx);
//        setYTranslate(yTrans - dy);
//    }
    lastPos = event->pos();
}

void SceneWidget::mouseDoubleClickEvent(QMouseEvent *event){
    right_click_menu_->exec(event->globalPos());
    return;
}

void SceneWidget::wheelEvent(QWheelEvent *event){
    QPoint numPixels = event->pixelDelta();
    int dy = numPixels.y();
    
    dy = (dy > 10) ? 10 : dy;
    dy = (dy < -10) ? -10 : dy;
    if (dy == 0) {
        return;
    }
    
    float ratio = static_cast<float>(dy) / 10.0f;
    float tx = static_cast<float>(event->x()) / width()*2 - 1;
    float ty = -1*static_cast<float>(event->y()) / height()*2 + 1;
    
    QMatrix4x4 matrix;
    view_matrix_.rotate(-yRot/16.0, 1.0, 0.0, 0.0);
    view_matrix_.rotate(-xRot/16.0, 0.0, 0.0, 1.0);
    QVector3D tp = matrix.map(QVector3D(tx, ty, 0.0));
    
    float previous_scale = scaleFactor_;
    scaleFactor_ *= pow(2.0, ratio);
    scaleFactor_ = (scaleFactor_>0.5) ? scaleFactor_:0.5;
    scaleFactor_ = (scaleFactor_< maxScaleFactor_) ? scaleFactor_:maxScaleFactor_;
    zoomTransX_ += (tp[0] - 0.005*xTrans) * (previous_scale - scaleFactor_) / previous_scale / scaleFactor_;
    zoomTransY_ += (tp[1] - 0.005*yTrans) * (previous_scale - scaleFactor_) / previous_scale / scaleFactor_ ;
    updateGL();
}

void SceneWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        updateGL();
    }
}

void SceneWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        updateGL();
    }
}

void SceneWidget::setXTranslate(int deltaX){
    if (deltaX != xTrans){
        xTrans = deltaX;
        updateGL();
    }
}

void SceneWidget::setYTranslate(int deltaY){
    if (deltaY != yTrans){
        yTrans = deltaY;
        updateGL();
    }
}

void SceneWidget::initializeGL(){
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendEquation(GL_SRC_ALPHA | GL_ONE_MINUS_SRC_ALPHA);
    glBlendColor(1.0, 0.0, 0.0, 0.5);
    m_program_ = new CustomizedShaderProgram(this);
    trajectories_->setShadderProgram(m_program_);
    osmMap_->setShadderProgram(m_program_);
    //graph_->setShadderProgram(m_program_);
    road_generator_->setShadderProgram(m_program_);
    query_init_feature_selector_->setShadderProgram(m_program_);
    query_q_feature_selector_->setShadderProgram(m_program_);
    m_program_->link();
}

void SceneWidget::paintGL(){
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);
    
    float aspect_ratio = static_cast<float>(width()) / height();
    
    qglClearColor(QColor(220, 220, 220));
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    m_program_->bind();
    view_matrix_.setToIdentity();
    view_matrix_.ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    view_matrix_.translate(0, 0, -2);
    view_matrix_.translate(0.005*xTrans, 0.005*yTrans);
    view_matrix_.scale(scaleFactor_, scaleFactor_*aspect_ratio, 1.0f);
    view_matrix_.translate(zoomTransX_, zoomTransY_);
    //view_matrix_.rotate(-xRot/16.0, 0.0, 0.0, 1.0);
    //view_matrix_.rotate(-yRot/16.0, 1.0, 0.0, 0.0);
    m_program_->setUniformMatrix(view_matrix_);
    if (show_map_){
        osmMap_->draw();
    }
    
//    if (show_graph_){
//        graph_->draw();
//    }
    trajectories_->draw();
    
    // Draw Axis at the bottom left corner
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    
    if (delta_x < 0 || delta_y < 0) {
        return;
    }
    float delta = (delta_x > delta_y) ? delta_x : delta_y;
    float axis_length = 100.0f / delta * 2;
    
    QVector3D near_point = view_matrix_.inverted().map(QVector3D(-0.8, -0.8, 0.0));
    QVector3D far_point = view_matrix_.inverted().map(QVector3D(-0.8, -0.8, 1.0));
    
    CgalLine line = CgalLine(Vec3Caster<QVector3D, CgalPoint>(near_point), Vec3Caster<QVector3D, CgalPoint>(far_point));
    
    CgalPlane plane = CgalPlane(CgalPoint(0,0,0), CgalVector(0,0,1));
    
    CGAL::Object intersection = CGAL::intersection(line, plane);
    if (const CgalPoint *axis_orig = CGAL::object_cast<CgalPoint>(&intersection)){
        float axis_origin_x = axis_orig->x();
        float axis_origin_y = axis_orig->y();
        vector<Vertex> axis_vertices;
        axis_vertices.push_back(Vertex(axis_origin_x, axis_origin_y+axis_length, 0.0f));
        axis_vertices.push_back(Vertex(axis_origin_x, axis_origin_y, 0.0f));
        axis_vertices.push_back(Vertex(axis_origin_x, axis_origin_y, 0.0f));
        axis_vertices.push_back(Vertex(axis_origin_x+axis_length, axis_origin_y, 0.0f));
        vector<Color> axis_colors;
        axis_colors.push_back(Color(1, 0, 0, 1.0));
        axis_colors.push_back(Color(1, 0, 0, 1.0));
        axis_colors.push_back(Color(0, 1, 0, 1.0));
        axis_colors.push_back(Color(0, 1, 0, 1.0));
        
        QOpenGLBuffer axis_position_buffer;
        axis_position_buffer.create();
        axis_position_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        axis_position_buffer.bind();
        axis_position_buffer.allocate(&axis_vertices[0], 3*axis_vertices.size()*sizeof(float));
        m_program_->setupPositionAttributes();
        
        QOpenGLBuffer axis_color_buffer;
        axis_color_buffer.create();
        axis_color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
        axis_color_buffer.bind();
        axis_color_buffer.allocate(&axis_colors[0], 4*axis_colors.size()*sizeof(float));
        m_program_->setupColorAttributes();
        glLineWidth(5);
        glDrawArrays(GL_LINE_STRIP, 0, axis_vertices.size());
    }
    
    // Draw Road Seed Selection
    road_generator_->draw();
    
    // Features
    query_init_feature_selector_->draw();
    query_q_feature_selector_->draw();
    
    m_program_->release();
}

void SceneWidget::resizeGL(int w, int h){
    if (h == 0) {
        h = 1;
    }
   
    glViewport(0, 0, (GLint)w, (GLint)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void SceneWidget::updateInformation(void)
{
	return;
}

void SceneWidget::prepareContextMenu(QMenu* menu)
{
	//menu->addAction("Open Trajectories", this, SLOT(slotOpenTrajectories()));
    
	return;
}

void SceneWidget::updateSceneBoundary(){
    updateMaxScaleFactor();
    trajectories_->prepareForVisualization();
    osmMap_->prepareForVisualization();
//    graph_->prepareForVisualization();
}

void SceneWidget::setSceneBoundary(QVector4D new_bound){
    SceneConst::getInstance().setBoundBox(new_bound);
    updateMaxScaleFactor();
}

void SceneWidget::updateMaxScaleFactor(){
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    float delta_x = bound_box[1] - bound_box[0];
    float delta_y = bound_box[3] - bound_box[2];
    float delta_boundary = (delta_x > delta_y) ? delta_x : delta_y;
    maxScaleFactor_ = delta_boundary / 20.0;
}

void SceneWidget::slotOpenTrajectories(void)
{
    MainWindow* main_window = MainWindow::getInstance();
    
	QString filename = QFileDialog::getOpenFileName(main_window,
                                                    "Open Trajectories",
                                                    default_trajectory_dir.c_str(),
                                                    "Trajectories (*.txt *.pbf)");
	if (filename.isEmpty())
		return;
    
    if (trajectories_->load(filename.toStdString()))
	{
        updateSceneBoundary();
        updateGL();
        emit trajFileLoaded(filename, trajectories_->getNumTraj(), trajectories_->getNumPoint());
        
        road_generator_->setTrajectories(trajectories_);
        query_init_feature_selector_->setTrajectories(trajectories_);
        query_q_feature_selector_->setTrajectories(trajectories_);
	}
    
	return;
}

void SceneWidget::slotOpenTrajectoriesFromFile(const QString &filename){
	if (filename.isEmpty())
		return;
    
    if (trajectories_->load(filename.toStdString()))
	{
        updateGL();
	}
    
	return;
}

void SceneWidget::slotSaveTrajectories(void)
{
	MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window, "Save Trajectories",
                                                    main_window->getWorkspace().c_str(), "Trajectories (*.pbf)");
    if (filename.isEmpty())
        return;
    
	trajectories_->save(filename.toStdString());
    
	return;
}

void SceneWidget::drawSelectedTraj(vector<int> &idx){
    trajectories_->setSelectedTrajectories(idx);
    enableSelectionMode();
}


void SceneWidget::slotOpenOsmMap(void)
{
    MainWindow* main_window = MainWindow::getInstance();
    
	QString filename = QFileDialog::getOpenFileName(main_window,
                                                    "Open OpenStreetMap",
                                                    default_map_dir.c_str(),
                                                    "Trajectories (*.osm)");
    
	if (filename.isEmpty())
		return;
    
    if (osmMap_->loadOSM(filename.toStdString()))
	{
        updateSceneBoundary();
        updateGL();
        emit osmFileLoaded(filename);
	}
	
	return;
}

void SceneWidget::slotOpenOsmMapFromFile(const QString &filename){
    if (filename.isEmpty())
		return;
    
    clearData();
    if (osmMap_->loadOSM(filename.toStdString()))
	{
        updateGL();
	}
}

void SceneWidget::slotSetShowMap(int state){
    if (state == Qt::Unchecked)
        show_map_ = false;
    else
        show_map_ = true;
    updateGL();
}

void SceneWidget::slotExtractMapBranchingPoints(){
        MainWindow* main_window = MainWindow::getInstance();
        
        QString filename = QFileDialog::getSaveFileName(main_window, "Save to file",
                                                        main_window->getWorkspace().c_str(), "Trajectories (*.txt)");
        if (filename.isEmpty())
            return;
        
        osmMap_->extractMapBranchingPoints(filename.toStdString());
        
        return;
}

void SceneWidget::slotRGeneratorExportQueryInitFeatures(){
    if (trajectories_->samples()->size() == 0){
        QMessageBox msgBox;
        msgBox.setText("Please sample point cloud first.");
        msgBox.exec();
        return;
    }
    
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window,
                                                    "Road Generator: Export ?I Features",
                                                    default_python_test_dir.c_str(),
                                                    "*.txt");
    if (filename.isEmpty())
        return;
   
    float sigma = main_window->getUi()->queryInitSearchRadius->value();
    road_generator_->sparseVoting(sigma);
    
    road_generator_->exportQueryInitFeatures(sigma, filename.toStdString());
    
    updateGL();
}

void SceneWidget::slotRGeneratorLoadQueryInitPredictions(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getOpenFileName(main_window,
                                                    "Road Generator: load ?I Prediction",
                                                    default_python_test_dir.c_str(),
                                                    "*.txt");
    if (filename.isEmpty())
        return;
    
    if (road_generator_->loadQueryInitFeatures(filename.toStdString()))
    {
        updateGL();
    }
    
    return;
    
}

void SceneWidget::slotRGeneratorAddQueryInitToString(){
    if(road_generator_->addQueryInitToString()){
        updateGL();
    }
}

void SceneWidget::slotRGeneratorExportQueryQFeatures(){
    if (trajectories_->samples()->size() == 0){
        QMessageBox msgBox;
        msgBox.setText("Please sample point cloud first.");
        msgBox.exec();
        return;
    }
    
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window,
                                                    "Road Generator: Export ?Q Features",
                                                    default_python_test_dir.c_str(),
                                                    "*.txt");
    if (filename.isEmpty())
        return;
    
    road_generator_->exportQueryQFeatures(filename.toStdString());
    
    updateGL();
}

void SceneWidget::slotRGeneratorLoadQueryQPredictions(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getOpenFileName(main_window,
                                                    "Road Generator: load ?I Prediction",
                                                    default_python_test_dir.c_str(),
                                                    "*.txt");
    if (filename.isEmpty())
        return;
    
    if (road_generator_->loadQueryQPredictions(filename.toStdString()))
    {
        updateGL();
    }
    
    return;
}

void SceneWidget::slotRGeneratorLocalAdjust(){
    road_generator_->localAdjustment();
}

void SceneWidget::slotRGeneratorApplyRules(){
    road_generator_->applyRules();
    updateGL();
}

void SceneWidget::slotRGeneratorCleanupSymbols(){
}

void SceneWidget::toggleTrajectories(void)
{
    trajectories_->toggleRenderMode();
    updateGL();
	return;
}

void SceneWidget::slotSamplePointCloud(void){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load trajectory file first.");
        msgBox.exec();
        return;
    }
    bool ok;
    MainWindow* main_window = MainWindow::getInstance();
    double d = QInputDialog::getDouble(main_window, tr("Set Sampling Parameter"), tr("Neighborhood Size (in meters):"), 20.0, 1.0, 250.0, 1, &ok);
    if (ok) {
        trajectories_->singleDirectionSamplePointCloud(static_cast<float>(d));
        trajectories_->prepareForVisualization();
       
        // Output information
        QString str;
        QTextStream(&str) <<trajectories_->samples()->size() << " samples. Grid size: "<<d << " m.";
        emit newSamplesDrawn(str);
        updateGL();
    }
}

void SceneWidget::slotSetShowSamples(int state){
    if (state == Qt::Unchecked)
        trajectories_->setShowSamples(false);
    else
        trajectories_->setShowSamples(true);
    updateGL();
}

void SceneWidget::slotClearAll(void){
    clearData();
}

void SceneWidget::resetView(void){
    xRot = 0;
    yRot = 0;
    xTrans = 0;
    yTrans = 0;
    scaleFactor_ = 1.0f;
    zoomTransX_ = 0.0f;
    zoomTransY_ = 0.0f;
    updateGL();
}

void SceneWidget::clearData(void){
    view_matrix_.setToIdentity();
    xRot = 0;
    yRot = 0;
    xTrans = 0;
    yTrans = 0;
   
    SceneConst::getInstance().setBoundBox(QVector4D(1e10, -1e10, 1e10, -1e10));
    
    trajectories_->clearData();
    osmMap_->clearData();
    feature_selection_mode_ = false;
    
    query_init_feature_selector_->clearVisibleFeatures();
    query_q_feature_selector_->clearVisibleFeatures();
    
    road_generator_->clear();
    selection_mode_ = false;
    
    scaleFactor_ = 1.0;
    maxScaleFactor_ = 1.0;
    zoomTransX_ = 0.0f;
    zoomTransY_ = 0.0f;
    updateGL();
}

void SceneWidget::slotExtractTrajectories(void){
    if (osmMap_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load an openstreetmap first.");
        msgBox.exec();
        return;
    }
    MainWindow* main_window = MainWindow::getInstance();
    
	QStringList filenames = QFileDialog::getOpenFileNames(main_window, "Open Trajectories",
                                                          main_window->getWorkspace().c_str(), "Trajectories (*.pbf)");
	if (filenames.isEmpty())
		return;
  
    osmMap_->updateMapSearchTree(10.0f); // update the map search tree with 10.0meter grid
    
    QVector4D bound_box = SceneConst::getInstance().getBoundBox();
    trajectories_->extractFromFiles(filenames, bound_box, osmMap_->map_search_tree());
    
    SceneConst::getInstance().setBoundBox(bound_box);
   
    trajectories_->prepareForVisualization();
    osmMap_->prepareForVisualization();
    
    updateGL();
}

void SceneWidget::slotSetShowDirection(int state){
    if (state == Qt::Unchecked)
        trajectories_->setShowDirection(false);
    else
        trajectories_->setShowDirection(true);
    updateGL();
}

//void SceneWidget::slotOpenOsmFile(QModelIndex index){
//}

//void SceneWidget::mouseMoveEvent(QMouseEvent* event)
//{
//	//MainWindow* main_window = MainWindow::getInstance();
//
//	return;
//}

void SceneWidget::slotEnterSampleSelectionMode(void){
    sample_selection_mode = true;
}

void SceneWidget::slotClearPickedSamples(void){
    trajectories_->clearPickedSamples();
    updateGL();
}

void SceneWidget::slotInitializeGraph(void){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please loat trajectories first.");
        msgBox.exec();
        return;
    }
    
    if (trajectories_->samples()->size() == 0){
        QMessageBox msgBox;
        msgBox.setText("Please do sampling first.");
        msgBox.exec();
        return;
    }
    
    //graph_->updateGraphUsingSamplesAndSegments(trajectories_->samples(), trajectories_->sample_tree(), trajectories_->segments(), trajectories_->sample_segment_clusters(), trajectories_->sample_cluster_sizes(), trajectories_->data(), trajectories_->tree());
    //graph_->updateGraphUsingDBSCANClustersAndSamples(trajectories_->samples(), trajectories_->sample_tree(),  trajectories_->data(), trajectories_->tree(), trajectories_->dbscanClusterSamples());
    
//    graph_->updateGraphUsingSamplesAndGpsPointCloud(trajectories_->samples(), trajectories_->sample_tree(),  trajectories_->data(), trajectories_->tree());
//    graph_->updateGraphUsingDescriptor(trajectories_->cluster_centers(), trajectories_->cluster_center_search_tree(), trajectories_->descriptors(), trajectories_->cluster_popularity(), trajectories_->data(), trajectories_->tree());
    
    trajectories_->prepareForVisualization();
//    graph_->prepareForVisualization();
    
    // Generate information
    QString str;
//    QTextStream(&str) << graph_->nVertices() << " vertices. " << graph_->nEdges()<< " edges.";
//    emit newGraphComputed(str);
    updateGL();
}

void SceneWidget::slotUpdateGraph(void){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please loat trajectories first.");
        msgBox.exec();
        return;
    }
    
    if (trajectories_->samples()->size() == 0){
        QMessageBox msgBox;
        msgBox.setText("Please do sampling first.");
        msgBox.exec();
        return;
    }
    
    //graph_->updateGraphUsingSamplesAndSegments(trajectories_->samples(), trajectories_->sample_tree(), trajectories_->segments(), trajectories_->sample_segment_clusters(), trajectories_->sample_cluster_sizes(),trajectories_->data(), trajectories_->tree());
//    graph_->prepareForVisualization();
    
    // Generate information
    QString str;
//    QTextStream(&str) << graph_->nVertices() << " vertices. " << graph_->nEdges()<< " edges.";
    emit newGraphComputed(str);
    updateGL();
}

void SceneWidget::slotSaveQueryQFeatures(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window, "Save ?Q Features",
                                                    main_window->getWorkspace().c_str(), "*.txt");
    if (filename.isEmpty())
        return;
    
    query_q_feature_selector_->save(filename.toStdString());
}

void SceneWidget::slotExportQueryQFeatures(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window, "Export ?Q Features",
                                                    main_window->getWorkspace().c_str(), "*.txt");
    if (filename.isEmpty())
        return;
    query_q_feature_selector_->exportFeatures(filename.toStdString());
    
    updateGL();
}

void SceneWidget::slotLoadQueryQPredictions(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getOpenFileName(main_window, "Load Query Q Prediction",
                                                    main_window->getWorkspace().c_str(), "*.txt");
    if (filename.isEmpty())
        return;
    
    if (query_q_feature_selector_->loadPrediction(filename.toStdString()))
    {
        updateGL();
    }
    
    return;
}

void SceneWidget::slotComputeQueryInitFeaturesFromMap(){
    if (osmMap_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load an openstreetmap first.");
        msgBox.exec();
        return;
    }
    
    MainWindow* main_window = MainWindow::getInstance();
    float sigma = main_window->getUi()->queryInitSearchRadius->value();
    road_generator_->sparseVoting(sigma);
    
    query_init_feature_selector_->computeQueryInitFeaturesFromMap(sigma, osmMap_);
    updateGL();
}

void SceneWidget::slotSaveQueryInitFeatures(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window, "Save Features",
                                                    default_python_test_dir.c_str(),
                                                    "*.txt");
    if (filename.isEmpty())
        return;
    query_init_feature_selector_->save(filename.toStdString());
}

void SceneWidget::slotExportQueryInitFeatures(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window,
                                                    "Export Features",
                                                    default_python_test_dir.c_str(),
                                                    "*.txt");
    if (filename.isEmpty())
        return;
    
    float sigma = main_window->getUi()->queryInitSearchRadius->value();
    road_generator_->sparseVoting(sigma);
    
    query_init_feature_selector_->exportFeatures(sigma, filename.toStdString());
    
    updateGL();
}

void SceneWidget::slotLoadQueryInitPredictions(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getOpenFileName(main_window, "Load Query Init Prediction",
                                                    main_window->getWorkspace().c_str(), "*.txt");
    if (filename.isEmpty())
        return;
    
    if (query_init_feature_selector_->loadPrediction(filename.toStdString()))
    {
        updateGL();
    }
    
    return;
    
}

void SceneWidget::slotComputeQueryQFeaturesFromMap(){
    if (osmMap_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load an openstreetmap first.");
        msgBox.exec();
        return;
    }
    
    query_q_feature_selector_->computeQueryQFeaturesFromMap(osmMap_);
    updateGL();
}

void SceneWidget::slotSetShowGraph(int state){
//    if (state == Qt::Unchecked)
//        show_graph_ = false;
//    else
//        show_graph_ = true;
    updateGL();
}
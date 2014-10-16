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
#include "openstreetmap.h"
#include "graph.h"
#include "renderable.h"
#include "color_map.h"
#include "main_window.h"

SceneWidget::SceneWidget(QWidget * parent, const QGLWidget * shareWidget, Qt::WindowFlags f)
:QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    view_matrix_.setToIdentity();
    xRot = 0;
    yRot = 0;
    xTrans = 0;
    yTrans = 0;
    
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    
    trajectories_ = new Trajectories(this);
    sample_selection_mode = false;
    osmMap_ = new OpenStreetMap(this);
    graph_ = new Graph(this);
    selection_mode_ = false;
    
    scaleFactor_ = 1.0;
    maxScaleFactor_ = 1.0;
    zoomTransX_ = 0.0f;
    zoomTransY_ = 0.0f;
    
    right_click_menu_ = new QMenu;
    right_click_menu_->addAction("Open trajectories", this, SLOT(slotOpenTrajectories()));
    right_click_menu_->addAction("Save trajectories", this, SLOT(slotSaveTrajectories()));
    right_click_menu_->addSeparator();
    right_click_menu_->addAction("Open OSM", this, SLOT(slotOpenOsmMap()));
    
    show_map_ = true;
    show_graph_ = true;
}

SceneWidget::~SceneWidget()
{
    delete m_program_;
    delete right_click_menu_;
    delete trajectories_;
    delete osmMap_;
    delete graph_;
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
            
            float center_x = 0.5*bound_box_[0] + 0.5*bound_box_[1];
            float center_y = 0.5*bound_box_[2] + 0.5*bound_box_[3];
            float delta_x = bound_box_[1] - bound_box_[0];
            float delta_y = bound_box_[3] - bound_box_[2];
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
    graph_->setShadderProgram(m_program_);
    m_program_->link();
}

void SceneWidget::paintGL(){
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);
    
    qglClearColor(QColor(220, 220, 220));
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    m_program_->bind();
    view_matrix_.setToIdentity();
    view_matrix_.ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 100.0f);
    view_matrix_.translate(0, 0, -2);
    view_matrix_.translate(0.005*xTrans, 0.005*yTrans);
    view_matrix_.scale(scaleFactor_, scaleFactor_, 1.0f);
    view_matrix_.translate(zoomTransX_, zoomTransY_);
    //view_matrix_.rotate(-xRot/16.0, 0.0, 0.0, 1.0);
    //view_matrix_.rotate(-yRot/16.0, 1.0, 0.0, 0.0);
    m_program_->setUniformMatrix(view_matrix_);
    if (show_map_){
        osmMap_->draw();
    }
    
    if (show_graph_){
        graph_->draw();
    }
    trajectories_->draw();
    
    // Draw Axis at the bottom left corner
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
    
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

void SceneWidget::updateSceneBoundary(QVector4D bound_box_to_insert){
    if (bound_box_[0] > bound_box_to_insert[0]) {
        bound_box_[0] = bound_box_to_insert[0];
    }
    if (bound_box_[1] < bound_box_to_insert[1]) {
        bound_box_[1] = bound_box_to_insert[1];
    }
    if (bound_box_[2] > bound_box_to_insert[2]) {
        bound_box_[2] = bound_box_to_insert[2];
    }
    if (bound_box_[3] < bound_box_to_insert[3]) {
        bound_box_[3] = bound_box_to_insert[3];
    }
    updateMaxScaleFactor();
    trajectories_->prepareForVisualization(bound_box_);
    osmMap_->prepareForVisualization(bound_box_);
    graph_->prepareForVisualization(bound_box_);
}

void SceneWidget::setSceneBoundary(QVector4D new_bound){
    bound_box_ = new_bound;
    updateMaxScaleFactor();
}

void SceneWidget::updateMaxScaleFactor(){
    float delta_x = bound_box_[1] - bound_box_[0];
    float delta_y = bound_box_[3] - bound_box_[2];
    float delta_boundary = (delta_x > delta_y) ? delta_x : delta_y;
    maxScaleFactor_ = delta_boundary / 20.0;
}

void SceneWidget::slotOpenTrajectories(void)
{
    MainWindow* main_window = MainWindow::getInstance();
    
	QString filename = QFileDialog::getOpenFileName(main_window, "Open Trajectories",
                                                    main_window->getWorkspace().c_str(), "Trajectories (*.txt *.pbf)");
	if (filename.isEmpty())
		return;
    
    if (trajectories_->load(filename.toStdString()))
	{
        updateSceneBoundary(trajectories_->BoundBox());
        updateGL();
        emit trajFileLoaded(filename, trajectories_->getNumTraj(), trajectories_->getNumPoint());
	}
    
	return;
}

void SceneWidget::slotOpenTrajectoriesFromFile(const QString &filename){
	if (filename.isEmpty())
		return;
    
    if (trajectories_->load(filename.toStdString()))
	{
        updateSceneBoundary(trajectories_->BoundBox());
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
    
	QString filename = QFileDialog::getOpenFileName(main_window, "Open OpenStreetMap",
                                                    main_window->getWorkspace().c_str(), "Trajectories (*.osm)");
    
	if (filename.isEmpty())
		return;
    
    if (osmMap_->loadOSM(filename.toStdString()))
	{
        updateSceneBoundary(osmMap_->BoundBox());
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
        updateSceneBoundary(osmMap_->BoundBox());
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

void SceneWidget::slotClusterSegmentsAtSample(void){
    set<int> &picked_samples = trajectories_->picked_sample_idxs();
    if (picked_samples.size() == 0) {
        return;
    }
    
    vector<int> selected_samples;
    for(set<int>::iterator it = picked_samples.begin(); it != picked_samples.end(); ++it){
        selected_samples.push_back(*it);
    }
    
    MainWindow* main_window = MainWindow::getInstance();
    double sigmaValue = main_window->getSigmaValue();
    int minClusterSize = main_window->getMinClusterSize();
    //int n_cluster = trajectories_->clusterSegmentsUsingDistanceAtSample(sample_id, sigmaValue, thresholdValue, minClusterSize);
    vector<vector<int>> clusters;
    int n_cluster = trajectories_->fpsMultiSiteSegmentClustering(selected_samples, sigmaValue, minClusterSize, clusters);
    QString str;
    QTextStream(&str) << n_cluster << " cluster extracted.";
    main_window->showInStatusBar(str);
    //trajectories_->clusterSegmentsWithGraphAtSample(sample_id, graph_);
    emit nClusterComputed(n_cluster);
    
    updateGL();
    
}

void SceneWidget::slotClusterSegmentsAtAllSamples(void){
    MainWindow* main_window = MainWindow::getInstance();
    double sigmaValue = main_window->getSigmaValue();
    int minClusterSize = main_window->getMinClusterSize();
    clock_t begin = clock();
    trajectories_->clusterSegmentsAtAllSamples(sigmaValue, minClusterSize);
    clock_t end = clock();
    double time_elapsed = double(end - begin) / CLOCKS_PER_SEC;
    QString str;
    QTextStream(&str) << "Clustering completed. Time elapsed: "<<time_elapsed <<" sec.";
    main_window->showInStatusBar(str);
    updateGL();
}

void SceneWidget::slotPickClusterAtIdx(int cluster_idx){
    trajectories_->showClusterAtIdx(cluster_idx);
    updateGL();
}

void SceneWidget::slotExportSampleSegments(){
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window, "Save Sample Clusters",
                                                    main_window->getWorkspace().c_str(), "(*.txt)");
    if (filename.isEmpty())
        return;
    
    trajectories_->exportSampleSegments(filename.toStdString());
}

void SceneWidget::slotExtractSampleFeatures(){
    if (trajectories_->samples()->size() == 0){
        QMessageBox msgBox;
        msgBox.setText("Please do sampling first.");
        msgBox.exec();
        return;
    }
    
    MainWindow* main_window = MainWindow::getInstance();
    
    QString filename = QFileDialog::getSaveFileName(main_window, "Save Sample Descriptors to",
                                                    main_window->getWorkspace().c_str(), "(*.txt)");
    if (filename.isEmpty())
        return;
    
    trajectories_->extractSampleFeatures(filename.toStdString());
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
    double d = QInputDialog::getDouble(this, tr("Set Sampling Parameter"), tr("Neighborhood Size (in meters):"), 50.0, 1.0, 250.0, 1, &ok);
    if (ok) {
        trajectories_->samplePointCloud(static_cast<float>(d));
        trajectories_->prepareForVisualization(bound_box_);
       
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

void SceneWidget::slotGenerateSegments(void){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load trajectory file first.");
        msgBox.exec();
        return;
    }
    bool ok;
    double d = QInputDialog::getDouble(this, tr("Set Segment Parameter"), tr("Segment extension (in meters):"), 100.0, 1.0, 250.0, 1, &ok);
    if (ok) {
        trajectories_->computeSegments(static_cast<float>(d));
        
        // Generate information
        QString str;
        QTextStream(&str) << trajectories_->nSegments() << " segments. Extension: " << d << "m.";
        emit newSegmentsComputed(str);
    }
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
    
    bound_box_ = QVector4D(1e10, -1e10, 1e10, -1e10);
    
    trajectories_->clearData();
    osmMap_->clearData();
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
    
    trajectories_->extractFromFiles(filenames, bound_box_, osmMap_->map_search_tree());
    updateSceneBoundary(trajectories_->BoundBox());
    updateGL();
}

void SceneWidget::slotComputePointCloudVariance(){
    bool ok;
    int n = QInputDialog::getInt(this, tr("Set Neighborhood"), tr("Neighborhood size):"), 10, 5, 100, 5, &ok);
    if (ok) {
        trajectories_->computeWeightAtScale(n);
    }
    updateGL();
}

void SceneWidget::slotSetShowDistanceGraph(int state){
    if (state == Qt::Unchecked)
        trajectories_->setShowDistanceGraph(false);
    else
        trajectories_->setShowDistanceGraph(true);
    updateGL();
}

void SceneWidget::slotComputeDistanceGraph(){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please loat trajectories first.");
        msgBox.exec();
        return;
    }
    trajectories_->computeDistanceGraph();
    updateGL();
}

void SceneWidget::slotSetShowDirection(int state){
    if (state == Qt::Unchecked)
        trajectories_->setShowDirection(false);
    else
        trajectories_->setShowDirection(true);
    updateGL();
}

void SceneWidget::slotDBSCAN(){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load trajectories first.");
        msgBox.exec();
        return;
    }
    
    MainWindow* main_window = MainWindow::getInstance();
    double eps = main_window->getDBSCANEpsValue();
    int minPts = main_window->getDBSCANMinPtsValue();
   
    trajectories_->DBSCAN(eps, minPts);
    int n_dbscan_clusters = trajectories_->DBSCANNumClusters();
    emit nDBSCANClusterComputed(n_dbscan_clusters);
    updateGL();
}

void SceneWidget::slotSampleDBSCANClusters(){
    if (trajectories_->isEmpty()){
        QMessageBox msgBox;
        msgBox.setText("Please load trajectory file first.");
        msgBox.exec();
        return;
    }
    if (trajectories_->DBSCANNumClusters() == 0){
        QMessageBox msgBox;
        msgBox.setText("Please do DBSCAN clustering first.");
        msgBox.exec();
        return;
    }
    bool ok;
    double d = QInputDialog::getDouble(this, tr("Set Sampling Parameter"), tr("Neighborhood Size (in meters):"), 50.0, 1.0, 250.0, 1, &ok);
    if (ok) {
        trajectories_->sampleDBSCANClusters(static_cast<float>(d));
        trajectories_->prepareForVisualization(bound_box_);
        
        // Output information
        QString str;
        QTextStream(&str) <<trajectories_->samples()->size() << " samples. Grid size: "<<d << " m.";
        emit newSamplesDrawn(str);
        updateGL();
    }
}

void SceneWidget::slotSelectClusterAt(int clusterId){
    if (trajectories_->DBSCANNumClusters() == 0){
        return;
    }
    trajectories_->selectDBSCANClusterAt(clusterId);
    updateGL();
}

void SceneWidget::slotShowAllClusters(){
    if (trajectories_->DBSCANNumClusters() == 0){
        return;
    }
    trajectories_->showAllDBSCANClusters();
    updateGL();
}

void SceneWidget::slotCutTraj(){
    trajectories_->cutTraj();
    updateGL();
}

void SceneWidget::slotMergePathlet(){
    trajectories_->mergePathlet();
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

void SceneWidget::slotSampleCoverDistanceChange(double d){
    trajectories_->selectSegmentsWithSearchRange(static_cast<float>(d));
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
    graph_->updateGraphUsingDBSCANClustersAndSamples(trajectories_->samples(), trajectories_->sample_tree(),  trajectories_->data(), trajectories_->tree(), trajectories_->dbscanClusterSamples());
    
//    graph_->updateGraphUsingSamplesAndGpsPointCloud(trajectories_->samples(), trajectories_->sample_tree(),  trajectories_->data(), trajectories_->tree());
//    graph_->updateGraphUsingDescriptor(trajectories_->cluster_centers(), trajectories_->cluster_center_search_tree(), trajectories_->descriptors(), trajectories_->cluster_popularity(), trajectories_->data(), trajectories_->tree());
    
    trajectories_->setGraph(graph_);
    trajectories_->prepareForVisualization(bound_box_);
    graph_->prepareForVisualization(bound_box_);
    
    // Generate information
    QString str;
    QTextStream(&str) << graph_->nVertices() << " vertices. " << graph_->nEdges()<< " edges.";
    emit newGraphComputed(str);
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
    graph_->prepareForVisualization(bound_box_);
    
    // Generate information
    QString str;
    QTextStream(&str) << graph_->nVertices() << " vertices. " << graph_->nEdges()<< " edges.";
    emit newGraphComputed(str);
    updateGL();
}

void SceneWidget::slotSetShowGraph(int state){
    if (state == Qt::Unchecked)
        show_graph_ = false;
    else
        show_graph_ = true;
    updateGL();
}

void SceneWidget::slotDrawSegmentAndShortestPathInterpolation(int seg_idx){
    if (trajectories_->nSegments() == 0){
        return;
    }
    printf("Segment %d selected.\n", seg_idx);
    
    trajectories_->clearDrawnSegments();
    trajectories_->drawSegmentAt(seg_idx);
    Segment &seg = trajectories_->segmentAt(seg_idx);
    vector<int> seg_pt_idxs;
    for(size_t i = 0; i < seg.points().size(); ++i){
        SegPoint &pt = seg.points()[i];
        seg_pt_idxs.push_back(pt.orig_idx);
    }
    
    graph_->clearPathsToDraw();
    graph_->drawShortestPathInterpolationFor(seg_pt_idxs);
    
    trajectories_->drawSegmentAt(seg_idx+1);
    Segment &seg1 = trajectories_->segmentAt(seg_idx+1);
    vector<int> seg_pt_idxs1;
    for(size_t i = 0; i < seg1.points().size(); ++i){
        SegPoint &pt = seg1.points()[i];
        seg_pt_idxs1.push_back(pt.orig_idx);
    }
    graph_->drawShortestPathInterpolationFor(seg_pt_idxs1);
    
    printf("dist = %.2f\n", graph_->SegDistance(seg_pt_idxs, seg_pt_idxs1));
    
    updateGL();
}

void SceneWidget::slotSetShowSegments(int state){
    if (state == Qt::Unchecked)
        trajectories_->setShowSegments(false);
    else
        trajectories_->setShowSegments(true);
    updateGL();
}

void SceneWidget::slotInterpolateSegments(){
    if (graph_->nVertices() == 0) {
        QMessageBox msgBox;
        msgBox.setText("Please initialize graph first.");
        msgBox.exec();
        return;
    }
    
    if (trajectories_->nSegments() == 0) {
        QMessageBox msgBox;
        msgBox.setText("Please compute segments first.");
        msgBox.exec();
        return;
    }
    clock_t begin = clock();
    trajectories_->interpolateSegmentWithGraph(graph_);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("%.1f sec elapsed.\n", elapsed_secs);
}
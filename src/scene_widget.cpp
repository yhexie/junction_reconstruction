#include "scene_widget.h"
#include <cmath>
#include <QDebug>
#include <QMenu>
#include <QMouseEvent>
#include <QColor>
#include <array>
#include "cgal_types.h"
#include <CGAL/intersections.h>
#include "trajectories.h"
#include "main_window.h"

SceneWidget::SceneWidget(QWidget * parent, const QGLWidget * shareWidget, Qt::WindowFlags f)
:QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    trajectories_ = new Trajectories(this);
    right_click_menu_ = new QMenu;
    right_click_menu_->addAction("Open trajectory", this, SLOT(slotOpenTrajectories()));
    right_click_menu_->addAction("Toggle trajectory display", this, SLOT(slotToggleTrajectories()));
    
    matrix_.setToIdentity();
    scaleFactor_ = 1.0;
    scaleStep_ = 0.02;
    xRot = 0.0;
    yRot = 0.0;
    i_ = 0;
}

SceneWidget::~SceneWidget()
{
    delete m_program_;
    delete right_click_menu_;
    delete trajectories_;
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
    
    float tx = static_cast<float>(event->x()) / width()*2 - 1;
    float ty = -1*static_cast<float>(event->y()) / height()*2 + 1;
    
    QVector3D near_point = matrix_.inverted().map(QVector3D(tx, ty, 0.0));
    QVector3D far_point = matrix_.inverted().map(QVector3D(tx, ty, 1.0));
    
    CgalLine line = CgalLine(Vec3Caster<QVector3D, CgalPoint>(near_point), Vec3Caster<QVector3D, CgalPoint>(far_point));
    
    CgalPlane plane = CgalPlane(CgalPoint(0,0,0), CgalVector(0,0,1));
    
    CGAL::Object intersection = CGAL::intersection(line, plane);
    if (const CgalPoint *intersection_point = CGAL::object_cast<CgalPoint>(&intersection)){
        printf("Intersection point at: \n\t x=%f, y=%f, z=%f\n", intersection_point->x(), intersection_point->y(), intersection_point->z());
    }
}

void SceneWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();
    
    if (event->buttons() & Qt::LeftButton) {
        setXRotation(xRot + 8 * dy);
        setYRotation(yRot + 8 * dx);
    }
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
    
    scaleFactor_ += dy * scaleStep_;
    scaleFactor_ = (scaleFactor_>0.1) ? scaleFactor_:0.1;
    scaleFactor_ = (scaleFactor_<10) ? scaleFactor_:10;
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

void SceneWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        updateGL();
    }
}

void SceneWidget::initializeGL(){
    m_program_ = new CustomizedShaderProgram(this);
    trajectories_->setShadderProgram(m_program_);
    m_program_->link();
}

void SceneWidget::paintGL(){
    qglClearColor(QColor(230, 230, 230));
    glClear(GL_COLOR_BUFFER_BIT);
    m_program_->bind();
    matrix_.setToIdentity();
    float aspectRatio = float(width()) / height();
    
    matrix_.perspective(60.0f, aspectRatio, 0.1f, 100.0f);
    matrix_.translate(0, 0, -2);
    matrix_.scale(scaleFactor_);
    matrix_.rotate(xRot/16.0, 1.0, 0.0, 0.0);
    matrix_.rotate(yRot/16.0, 0.0, 1.0, 0.0);
    
    m_program_->setUniformMatrix(matrix_);
    trajectories_->draw();
    
    //float vertices[] = {
    //    -0.8f, -0.8f, 0.0f,
    //    0.8f, -0.8f, 0.0f,
    //    0.0f,  0.8f, 0.0f
    //};
    //
    //float colors[] = {
    //    1.0f, 0.0f, 0.0f, 1.0f,
    //    0.0f, 1.0f, 0.0f, 1.0f,
    //    0.0f, 0.0f, 1.0f, 1.0f
    //};
    //
    //unsigned elements[] = {
    //   0, 1, 2
    //};
    
    //QOpenGLBuffer test_buffer;
//    test_buffer.create();
//    test_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
//    test_buffer.bind();
//    test_buffer.allocate(vertices, 3*3*sizeof(float));
//    m_program_->enableAttributeArray("posAttr");
//    m_program_->setAttributeBuffer("posAttr", GL_FLOAT, 0, 3);
//    
//    test_color_buffer.create();
//    test_color_buffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
//    test_color_buffer.bind();
//    test_color_buffer.allocate(colors, 3*4*sizeof(float));
//    
//    m_program_->enableAttributeArray("colAttr");
//    m_program_->setAttributeBuffer("colAttr", GL_FLOAT, 0, 4);
//    
//    QOpenGLBuffer elementBuffer(QOpenGLBuffer::IndexBuffer);
//    elementBuffer.create();
//    elementBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
//    elementBuffer.bind();
//    elementBuffer.allocate(elements, 3*sizeof(unsigned));
//    
//    glPointSize(20);
//    glDrawElements(GL_POINTS, 3, GL_UNSIGNED_INT, 0);

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

void SceneWidget::slotOpenTrajectories(void)
{
    MainWindow* main_window = MainWindow::getInstance();
    
	QString filename = QFileDialog::getOpenFileName(main_window, "Open Trajectories",
                                                    main_window->getWorkspace().c_str(), "Trajectories (*.txt)");
	if (filename.isEmpty())
		return;
    
	if (trajectories_->load(filename.toStdString()))
	{
        updateGL();
        emit trajFileLoaded(filename);
	}
    
	return;
}

void SceneWidget::slotSaveTrajectories(void)
{
	//MainWindow* main_window = MainWindow::getInstance();
    
    //	QString filename = QFileDialog::getSaveFileName(main_window, "Save Trajectories",
    //                                                    main_window->getWorkspace().c_str(), "Trajectories (*.pcd)");
    //	if (filename.isEmpty())
    //		return;
    //
	//trajectories_->save(filename.toStdString());
    
	return;
}


void SceneWidget::slotToggleTrajectories(void)
{
    trajectories_->toggleRenderMode();
    updateGL();
	return;
}

void SceneWidget::slotColorizeUniform(void)
{
	//trajectories_->setColorMode(UNIFORM);
    
	return;
}

void SceneWidget::slotColorizeSampleTime(void)
{
	//trajectories_->setColorMode(SAMPLE_TIME);
    
	return;
}

void SceneWidget::slotColorizeSampleOrder(void)
{
	//trajectories_->setColorMode(SAMPLE_ORDER);
    
	return;
}

//void SceneWidget::slotOpenOsmFile(QModelIndex index){
//}

//void SceneWidget::mouseMoveEvent(QMouseEvent* event)
//{
//	//MainWindow* main_window = MainWindow::getInstance();
//
//	return;
//}
#include <QToolTip>
#include <QKeyEvent>
#include <QSettings>
#include <QGridLayout>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QApplication>
#include <QFileSystemModel>
#include <QTreeView>
#include "scene_widget.h"
#include "main_window.h"
#include <QDebug>

MainWindow::MainWindow(QWidget *parent):
QMainWindow(parent), ui_(new Ui::MainWindowClass), workspace_(".")
{
    MainWindowInstancer::getInstance().main_window_ = this;
    
    ui_->setupUi(this);
    
    init();
}

MainWindow::~MainWindow()
{
    delete ui_;
    delete osmFileModel;
    delete trajFileModel;
    saveSettings();
    return;
}

void MainWindow::slotShowInformation(const QString& information)
{
	QToolTip::showText(QCursor::pos(), information);
}

void MainWindow::showInformation(const std::string& information)
{
	emit showInformationRequested(information.c_str());
}

void MainWindow::slotShowStatus(const QString& status, int timeout)
{
    
}

void MainWindow::slotOsmDirSelected(QModelIndex index){
    if (osmFileModel->isDir(index)) {
        ui_->osmDirView->setRootIndex(osmFileModel->setRootPath(osmFileModel->fileInfo(index).absoluteFilePath()));
        return;
    }
    
    // Check file extension
    if (osmFileModel->fileInfo(index).fileName().endsWith(".txt")) {
        
    }
}

void MainWindow::slotTrajDirSelected(QModelIndex index){
    if (trajFileModel->isDir(index)) {
        ui_->trajDirView->setRootIndex(trajFileModel->setRootPath(trajFileModel->fileInfo(index).absoluteFilePath()));
        return;
    }
    
    // Check file extension
}

void MainWindow::showStatus(const std::string& status, int timeout)
{
	emit showStatusRequested(status.c_str(), timeout);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);
    
    return;
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Down)
    {
        emit keyDownPressed();
    }
    
    QMainWindow::keyPressEvent(event);
    
    return;
}

MainWindow* MainWindow::getInstance()
{
    assert(MainWindowInstancer::getInstance().main_window_ != NULL);
    return MainWindowInstancer::getInstance().main_window_;
}

void MainWindow::init(void)
{
    setMouseTracking(true);
    //SceneWidget *scene_widget = new SceneWidget(this);
    
    //setCentralWidget(ui_->scene_widget);
    //    scene_widget->startRendering();
    
    connect(this, SIGNAL(showInformationRequested(const QString&)), this, SLOT(slotShowInformation(const QString&)));
    //	connect(this, SIGNAL(showStatusRequested(const QString&, int)), this, SLOT(slotShowStatus(const QString&, int)));
    connect(ui_->actionOpenTrajectories, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotOpenTrajectories()));
    //	connect(ui_->actionSaveTrajectories, SIGNAL(triggered()), scene_widget, SLOT(slotSaveTrajectories()));
    
    loadSettings();
    
    // File
	connect(ui_->actionSetWorkspace, SIGNAL(triggered()), this, SLOT(slotSetWorkspace()));
    
    // Algorithms
    
    // Visualization
    
	// Tools
    
    // Directory View
    osmFileModel = new QFileSystemModel(this);
    trajFileModel = new QFileSystemModel(this);
    osmFileModel->setFilter(QDir::Files | QDir::AllDirs | QDir::NoDot);
    trajFileModel->setFilter(QDir::Files | QDir::AllDirs | QDir::NoDot);
    
    ui_->osmDirView->setModel(osmFileModel);
    ui_->osmDirView->setColumnWidth(0, 200);
    ui_->osmDirView->hideColumn(2);
    ui_->osmDirView->hideColumn(3);
    ui_->osmDirView->setRootIndex(osmFileModel->setRootPath(QDir::currentPath()));
    
    ui_->trajDirView->setModel(trajFileModel);
    ui_->trajDirView->setColumnWidth(0, 200);
    ui_->trajDirView->hideColumn(2);
    ui_->trajDirView->hideColumn(3);
    ui_->trajDirView->setRootIndex(trajFileModel->setRootPath(QDir::currentPath()));
    
    connect(ui_->osmDirView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(slotOsmDirSelected(QModelIndex)));
    connect(ui_->trajDirView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(slotTrajDirSelected(QModelIndex)));
    connect(ui_->scene_widget, SIGNAL(trajFileLoaded(QString &)), this, SLOT(slotTrajFileLoaded(QString &)));
    
    return;
}

bool MainWindow::slotSetWorkspace(void)
{
    QString directory = QFileDialog::getExistingDirectory(this, tr("Set Workspace"), workspace_.c_str(), QFileDialog::ShowDirsOnly);
    
    if (directory.isEmpty())
        return false;
    
    workspace_ = directory.toStdString();
    
    return true;
}

void MainWindow::loadSettings()
{
    QSettings settings("GPS_Map_Construction", "GPS_Map_Construction");
    
    workspace_ = settings.value("workspace").toString().toStdString();
    
    return;
}

void MainWindow::saveSettings()
{
    QSettings settings("GPS_Map_Construction", "GPS_Map_Construction");
    
    QString workspace(workspace_.c_str());
	settings.setValue("workspace", workspace);
    
    return;
}

bool MainWindow::slotShowYesNoMessageBox(const std::string& text, const std::string& informative_text)
{
	QMessageBox msg_box;
	msg_box.setText(text.c_str());
	msg_box.setInformativeText(informative_text.c_str());
	msg_box.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
	msg_box.setDefaultButton(QMessageBox::Yes);
	int ret = msg_box.exec();
    
	return (ret == QMessageBox::Yes);
}

void MainWindow::slotTrajFileLoaded(QString &filename){
    ui_->trajDirView->setRootIndex(trajFileModel->setRootPath(QFileInfo(filename).absoluteDir().absolutePath()));
    ui_->trajDirView->setCurrentIndex(trajFileModel->index(filename));
}

void MainWindow::slotOsmFileLoaded(const QString &filename){
    
}
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
#include "latlon_converter.h"
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

void MainWindow::showStatus(const std::string& status, int timeout)
{
	emit showStatusRequested(status.c_str(), timeout);
}

void MainWindow::showInStatusBar(const QString &str){
    emit showStatusInformation(str);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    QMainWindow::closeEvent(event);
    return;
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    switch (event->key()) {
        case Qt::Key_Down:
            emit keyDownPressed();
            break;
        case Qt::Key_S:
            ui_->scene_widget->toggleSelectionMode();
            if (ui_->scene_widget->getSelectionMode()) {
                ui_->statusBar->showMessage("Selection Mode : ON");
            }else{
                ui_->statusBar->showMessage("Selection Mode : OFF");
            }
            break;
        case Qt::Key_T:
            ui_->scene_widget->toggleTrajectories();
            break;
        case Qt::Key_R:
            ui_->scene_widget->resetView();
            break;
        case Qt::Key_C:
            ui_->scene_widget->slotClearPickedSamples();
            break;
        case Qt::Key_P:
            ui_->scene_widget->slotEnterSampleSelectionMode();
            break;
        default:
            break;
    }
    
    QMainWindow::keyPressEvent(event);
    
    return;
}

void MainWindow::keyReleaseEvent(QKeyEvent *event){
    
}

MainWindow* MainWindow::getInstance()
{
    assert(MainWindowInstancer::getInstance().main_window_ != NULL);
    return MainWindowInstancer::getInstance().main_window_;
}

void MainWindow::init(void)
{
    setMouseTracking(true);
    
    connect(this, SIGNAL(showInformationRequested(const QString&)), this, SLOT(slotShowInformation(const QString&)));
    connect(this, SIGNAL(showStatusInformation(const QString&)), this, SLOT(slotShowStatusInformation(const QString&)));
    connect(ui_->actionOpenTrajectories, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotOpenTrajectories()));
    connect(ui_->actionOpenOsmMap, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotOpenOsmMap()));

    loadSettings();
    
    // File
	connect(ui_->actionSetWorkspace, SIGNAL(triggered()), this, SLOT(slotSetWorkspace()));
    
    // Visualization
	// Tools
    connect(ui_->actionExtractTrajectory, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotExtractTrajectories()));
    connect(ui_->actionSamplePointCloud, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotSamplePointCloud()));
    connect(ui_->actionPickSample, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotEnterSampleSelectionMode()));
    connect(ui_->actionClearPickedSamples, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotClearPickedSamples()));
    
    // Trajectory View
    connect(ui_->scene_widget, SIGNAL(trajFileLoaded(QString &, const size_t &, const size_t &)), this, SLOT(slotTrajFileLoaded(QString &, const size_t &, const size_t &)));
    connect(ui_->showDirection, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowDirection(int)));
    connect(ui_->utmZoneSelector, SIGNAL(currentIndexChanged(int)), this, SLOT(slotSetUTMZone(int)));
        // Sample View
    connect(ui_->showSampleCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowSamples(int)));
    connect(ui_->scene_widget, SIGNAL(newSamplesDrawn(QString &)), this, SLOT(slotNewSamplesDrawn(QString &)));
    
    // Map View
    connect(ui_->showMapCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowMap(int)));
    connect(this, SIGNAL(newOsmFileSelected(const QString &)), ui_->scene_widget, SLOT(slotOpenOsmMapFromFile(const QString &)));
    connect(ui_->scene_widget, SIGNAL(osmFileLoaded(QString &)), this, SLOT(slotOsmFileLoaded(QString &)));
    
    // Road Generator View
    connect(ui_->roadGeneratorLoadQueryInitClassifier, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorLoadQueryInitClassifier()));
    connect(ui_->roadGeneratorApplyQueryInitClassifier, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorApplyQueryInitClassifier()));
    connect(ui_->roadGeneratorSaveQueryInitResult, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorSaveQueryInitResult()));
    connect(ui_->roadGeneratorLoadQueryInitResult, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorLoadQueryInitResult()));
    connect(ui_->roadGeneratorComputeInitialRoadGuess, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorComputeInitialRoadGuess()));
    connect(ui_->roadGeneratorAddInitialRoad, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorAddInitialRoad()));
    connect(ui_->roadGeneratorTmp, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorTmp()));
    
    connect(ui_->roadGeneratorDevelopRoadNetwork, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorDevelopRoadNetwork()));
    connect(ui_->roadGeneratorLocalAdjustment, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorLocalAdjustment()));
    connect(ui_->roadGeneratorMCMCOptimization, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotRoadGeneratorMCMCOptimization()));
    
    // Feature View
    connect(ui_->extractQueryInitSamplesFromMap, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotExtractQueryInitTrainingSamplesFromMap()));
    connect(ui_->loadQueryInitTrainingSamples, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotLoadQueryInitTrainingSamples()));
    connect(ui_->saveQueryInitSamples, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotSaveQueryInitTrainingSamples()));
    connect(ui_->trainQueryInitClassifier, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotTrainQueryInitClassifier()));
    connect(ui_->saveQueryInitClassifier, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotSaveQueryInitClassifer()));
    
    connect(ui_->extractQueryQSamplesFromMap, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotExtractQueryQTrainingSamplesFromMap()));
    connect(ui_->loadQueryQTrainingSamples, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotLoadQueryQTrainingSamples()));
    connect(ui_->saveQueryQSamples, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotSaveQueryQTrainingSamples()));
    connect(ui_->trainQueryQClassifier, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotTrainQueryQClassifier()));
    connect(ui_->saveQueryQClassifier, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotSaveQueryQClassifer()));
    
    // Clear All
    connect(ui_->actionClearAll, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotClearAll()));
    
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
    return;
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

void MainWindow::slotTrajFileLoaded(QString &filename, const size_t &numTraj, const size_t &numPoint){
    QString str;
    QFileInfo info(filename);
    QTextStream(&str) << "Filename: " << info.baseName() << "." << info.completeSuffix() <<"\n" << numTraj << " trajectories, " << numPoint <<" points.";
    ui_->trajInfo->setText(str);
}

void MainWindow::slotOsmFileLoaded(QString &filename){
    QString str;
    QFileInfo info(filename);
    QTextStream(&str) << info.baseName() << "." << info.completeSuffix() <<" is loaded.";
    ui_->mapInfo->setText(str);
}

void MainWindow::slotNewSamplesDrawn(QString &info){
    ui_->sampleInfoLabel->setText(info);
}

void MainWindow::slotSetUTMZone(int index){
    Projector &projector = Projector::getInstance();
    if (index == 0) {
        projector.setUTMZone(Projector::BEIJING);
    }
    else{
        projector.setUTMZone(Projector::SAN_FRANCISCO);
    }
}

void MainWindow::slotTrajSelectionChanged(){
    ui_->statusBar->showMessage("Selection Mode : ON");
}

void MainWindow::slotShowStatusInformation(const QString &str){
    ui_->statusBar->showMessage(str);
}
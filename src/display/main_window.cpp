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

TrajListModel::TrajListModel(QObject *parent) : QAbstractListModel(parent){
    trajCount = 0;
}

int TrajListModel::rowCount(const QModelIndex & /* parent */) const{
    return trajCount;
}

void TrajListModel::setNumTraj(int number){
    beginResetModel();
    trajCount = number;
    endResetModel();
}

QVariant TrajListModel::data(const QModelIndex &index, int role) const{
    if ( role == Qt::DisplayRole ) {
        return index.row();
    }

    return QVariant();
}

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
    delete trajListModel;
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
    
    // Algorithms
    
    // Visualization
	// Tools
    connect(ui_->actionExtractTrajectory, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotExtractTrajectories()));
    connect(ui_->actionSamplePointCloud, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotSamplePointCloud()));
    connect(ui_->actionGenerateSegments, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotGenerateSegments()));
    connect(ui_->actionPickSample, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotEnterSampleSelectionMode()));
    connect(ui_->actionClearPickedSamples, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotClearPickedSamples()));
    connect(ui_->actionInitializeGraph, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotInitializeGraph()));
    
    // Trajectory View
    trajListModel = new TrajListModel(this);
    ui_->trajListView->setSelectionMode( QAbstractItemView::ExtendedSelection);
    ui_->trajListView->setModel(trajListModel);
    connect(ui_->scene_widget, SIGNAL(trajFileLoaded(QString &, const size_t &, const size_t &)), this, SLOT(slotTrajFileLoaded(QString &, const size_t &, const size_t &)));
    connect(this, SIGNAL(trajNumberChanged(int)), trajListModel, SLOT(setNumTraj(int)));
    connect(ui_->trajListView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), this, SLOT(slotTrajSelectionChanged()));
    connect(ui_->actionComputeDistanceGraph, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotComputeDistanceGraph()));
    connect(ui_->showDirection, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowDirection(int)));
    connect(ui_->actionDBSCAN, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotDBSCAN()));
    connect(ui_->actionSampleDBSCANClusters, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotSampleDBSCANClusters()));
    connect(ui_->scene_widget, SIGNAL(nDBSCANClusterComputed(int &)), this, SLOT(slotSetNDBSCANClusters(int &)));
 
    connect(ui_->utmZoneSelector, SIGNAL(currentIndexChanged(int)), this, SLOT(slotSetUTMZone(int)));
    connect(ui_->cutTraj, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotCutTraj()));
    connect(ui_->mergePathlet, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotMergePathlet()));
    connect(ui_->selectPathlet, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotSelectPathlet()));
    connect(ui_->scene_widget, SIGNAL(nPathletSelected(int &)), this, SLOT(slotSetNPathlets(int &)));
    connect(ui_->pathletThreshold, SIGNAL(valueChanged(int)), ui_->scene_widget, SLOT(slotSetPathletThreshold(int)));
    connect(ui_->showAllPathlets, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotShowAllPathlets()));
    connect(ui_->selectedPathletId, SIGNAL(valueChanged(int)), ui_->scene_widget, SLOT(slotShowPathletAt(int)));
    
    // Map View
    connect(ui_->showMapCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowMap(int)));
    connect(this, SIGNAL(newOsmFileSelected(const QString &)), ui_->scene_widget, SLOT(slotOpenOsmMapFromFile(const QString &)));
    connect(ui_->scene_widget, SIGNAL(osmFileLoaded(QString &)), this, SLOT(slotOsmFileLoaded(QString &)));
    connect(ui_->exportBranchingPoints, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotExtractMapBranchingPoints()));
    
    // Sample View
    connect(ui_->showSampleCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowSamples(int)));
    connect(ui_->scene_widget, SIGNAL(newSamplesDrawn(QString &)), this, SLOT(slotNewSamplesDrawn(QString &)));
    connect(ui_->sampleCoverDistanceSpinBox, SIGNAL(valueChanged(double)), ui_->scene_widget, SLOT(slotSampleCoverDistanceChange(double)));
    connect(ui_->exportSamplePathlets, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotExportSamplePathlets()));
    connect(ui_->scene_widget, SIGNAL(ithPathletToShowChanged(int &)), this, SLOT(slotSetIthPathletToShow(int &)));
    
    // Pathlet View
    connect(ui_->showPathletsCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowPathlets(int)));
    connect(ui_->showPathletDirection, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowPathletDirection(int)));
    connect(ui_->scene_widget, SIGNAL(updatePathletInfo(QString &)), this, SLOT(slotNewPathletsComputed(QString &)));
    connect(ui_->showIthPathlet, SIGNAL(valueChanged(int)), ui_->scene_widget, SLOT(slotShowIthPathlet(int)));
    connect(ui_->generateRoads, SIGNAL(clicked()), ui_->scene_widget, SLOT(slotGenerateRoads()));
    
    // Graph View
    connect(ui_->showGraphCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowGraph(int)));
    connect(ui_->scene_widget, SIGNAL(newGraphComputed(QString &)), this, SLOT(slotNewGraphComputed(QString &)));
    connect(ui_->actionUpdateGraph, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotUpdateGraph()));
    connect(ui_->actionInterpolateSegments, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotInterpolateSegments()));
    
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

void MainWindow::slotTrajFileLoaded(QString &filename, const size_t &numTraj, const size_t &numPoint){
    QString str;
    QFileInfo info(filename);
    QTextStream(&str) << "Filename: " << info.baseName() << "." << info.completeSuffix() <<"\n" << numTraj << " trajectories, " << numPoint <<" points.";
    ui_->trajInfo->setText(str);
    emit trajNumberChanged(numTraj);
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

void MainWindow::slotNewPathletsComputed(QString &info){
    ui_->pathletInfoLabel->setText(info);
}

void MainWindow::slotNewGraphComputed(QString &info){
    ui_->graphInfoLabel->setText(info);
}

void MainWindow::slotSetNPathlets(int &n_pathlets){
    if (n_pathlets > 0) {
        ui_->selectedPathletId->setMaximum(n_pathlets-1);
    }
    else{
        ui_->selectedPathletId->setValue(0);
    }
}

void MainWindow::slotSetIthPathletToShow(int &value){
    if (value >= 0) {
        ui_->showIthPathlet->setValue(value);
    }
    else{
        ui_->showIthPathlet->setValue(-1);
    }
}

void MainWindow::slotSetNDBSCANClusters(int &n_dbscan_cluster){
    if (n_dbscan_cluster > 0) {
        ui_->selectedPathletId->setMaximum(n_dbscan_cluster);
    }
    else{
        ui_->selectedPathletId->setValue(0);
    }
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
    QModelIndexList selectedRows = ui_->trajListView->selectionModel()->selectedRows();
    vector<int> selectedTrajIdx(selectedRows.size());
   
    for (int i = 0; i < selectedRows.size(); ++i) {
        QModelIndex idx = selectedRows.at(i);
        selectedTrajIdx[i] = idx.row();
    }
    ui_->scene_widget->drawSelectedTraj(selectedTrajIdx);
    ui_->statusBar->showMessage("Selection Mode : ON");
}

void MainWindow::slotShowStatusInformation(const QString &str){
    ui_->statusBar->showMessage(str);
}
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
    delete osmFileModel;
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

void MainWindow::slotOsmDirSelected(QModelIndex index){
    if (osmFileModel->isDir(index)) {
        ui_->osmDirView->setRootIndex(osmFileModel->setRootPath(osmFileModel->fileInfo(index).absoluteFilePath()));
        return;
    }
    
    // Check file extension
    if (osmFileModel->fileInfo(index).fileName().endsWith(".osm")) {
        emit newOsmFileSelected(osmFileModel->filePath(index));
    }
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
            ui_->scene_widget->clearData();
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
    connect(ui_->actionToggleSegmentsAtSample, SIGNAL(triggered()), ui_->scene_widget, SLOT(slotToggleSegmentsAtPickedSamples()));
    
    // Directory View
    osmFileModel = new QFileSystemModel(this);
    osmFileModel->setFilter(QDir::Files | QDir::AllDirs | QDir::NoDot);
    
    ui_->osmDirView->setModel(osmFileModel);
    ui_->osmDirView->setColumnWidth(0, 200);
    ui_->osmDirView->hideColumn(2);
    ui_->osmDirView->hideColumn(3);
    ui_->osmDirView->setRootIndex(osmFileModel->setRootPath(QDir::currentPath()));
    
    connect(ui_->showMapCheckBox, SIGNAL(stateChanged(int)), ui_->scene_widget, SLOT(slotSetShowMap(int)));
    connect(this, SIGNAL(newOsmFileSelected(const QString &)), ui_->scene_widget, SLOT(slotOpenOsmMapFromFile(const QString &)));
    connect(ui_->scene_widget, SIGNAL(osmFileLoaded(QString &)), this, SLOT(slotOsmFileLoaded(QString &)));
    
    connect(ui_->osmDirView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(slotOsmDirSelected(QModelIndex)));
    
    // Trajectory View
    trajListModel = new TrajListModel(this);
    ui_->trajListView->setSelectionMode( QAbstractItemView::ExtendedSelection);
    ui_->trajListView->setModel(trajListModel);
    connect(ui_->scene_widget, SIGNAL(trajFileLoaded(QString &, const size_t &, const size_t &)), this, SLOT(slotTrajFileLoaded(QString &, const size_t &, const size_t &)));
    connect(this, SIGNAL(trajNumberChanged(int)), trajListModel, SLOT(setNumTraj(int)));
    connect(ui_->trajListView->selectionModel(), SIGNAL(selectionChanged(QItemSelection, QItemSelection)), this, SLOT(slotTrajSelectionChanged()));
    
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
    QTextStream(&str) << numTraj << " trajectories, " << numPoint <<" points.";
    ui_->trajInfo->setText(str);
    emit trajNumberChanged(numTraj);
}

void MainWindow::slotOsmFileLoaded(QString &filename){
    ui_->osmDirView->setRootIndex(osmFileModel->setRootPath(QFileInfo(filename).absoluteDir().absolutePath()));
    ui_->osmDirView->setCurrentIndex(osmFileModel->index(filename));
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
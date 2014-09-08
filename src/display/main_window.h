#ifndef MainWindow_H
#define MainWindow_H

#include <string>
#include <cassert>

#include <QMutex>
#include <QMainWindow>
#include <QAbstractListModel>

#include "ui_main_window.h"

class SceneWidget;
class QFileSystemModel;

class TrajListModel : public QAbstractListModel{
    Q_OBJECT
public:
    TrajListModel(QObject *parent = 0);
    
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

    public slots:
    void setNumTraj(int number);
    
private:
    int trajCount;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent=0);
    void init(void);
    virtual ~MainWindow();
    static MainWindow* getInstance();
    
    SceneWidget* getSceneWidget(void);
    
	void showInformation(const std::string& information);
	void showStatus(const std::string& status, int timeout=0);
    void showInStatusBar(const QString &str);
    
    double getSigmaValue() {return ui_->sigmaValue->value(); }
    double getThresholdValue() { return ui_->thresholdValue->value(); }
    int getMinClusterSize() {return ui_->minClusterSize->value(); }
   
	const std::string& getWorkspace(void) const {return workspace_;}
    
    public slots:
	bool slotShowYesNoMessageBox(const std::string& text, const std::string& informative_text);
    void slotTrajSelectionChanged();
    
signals:
    void keyDownPressed(void);
	void showInformationRequested(const QString& information);
    void showStatusInformation(const QString &str);
	void showStatusRequested(const QString& status, int timeout);
    void newOsmFileSelected(const QString &filename);
    void newTrajFileSelected(const QString &filename);
    void trajNumberChanged(int number);
    
protected:
    virtual void closeEvent(QCloseEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
    
    
private slots:
    bool slotSetWorkspace(void);
	void slotShowInformation(const QString& information);
    void slotShowStatusInformation(const QString &str);
	void slotShowStatus(const QString& status, int timeout);
    void slotTrajFileLoaded(QString &filename, const size_t &numTraj, const size_t &numPoint);
    void slotOsmFileLoaded(QString &filename);
    void slotNewSamplesDrawn(QString &);
    void slotNewSegmentsComputed(QString &);
    void slotNewGraphComputed(QString &);
    
private:
    void loadSettings();
    void saveSettings();
    void saveStatusLog();
    
    TrajListModel                   *trajListModel;
    Ui::MainWindowClass             *ui_;
    std::string                     workspace_;
};

class MainWindowInstancer
{
public:
    static MainWindowInstancer& getInstance() {
        static MainWindowInstancer singleton_;
        return singleton_;
    }
    
private:
    MainWindowInstancer():main_window_(NULL){}
    MainWindowInstancer(const MainWindowInstancer &) {}            // copy ctor hidden
    MainWindowInstancer& operator=(const MainWindowInstancer &) {return (*this);}   // assign op. hidden
    virtual ~MainWindowInstancer(){}
    
    friend class MainWindow;
    MainWindow*   main_window_;
};

#endif // MainWindow_H
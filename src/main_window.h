#ifndef MainWindow_H
#define MainWindow_H

#include <string>
#include <cassert>

#include <QMutex>
#include <QMainWindow>

#include "ui_main_window.h"

class SceneWidget;
class QFileSystemModel;

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
    
	const std::string& getWorkspace(void) const {return workspace_;}
    
    public slots:
	bool slotShowYesNoMessageBox(const std::string& text, const std::string& informative_text);
    
signals:
    void keyDownPressed(void);
	void showInformationRequested(const QString& information);
	void showStatusRequested(const QString& status, int timeout);
    void selectOsmFile(QModelIndex &index);
    
protected:
    virtual void closeEvent(QCloseEvent *event);
    virtual void keyPressEvent(QKeyEvent* event);
    
    
private slots:
    bool slotSetWorkspace(void);
	void slotShowInformation(const QString& information);
	void slotShowStatus(const QString& status, int timeout);
    void slotOsmDirSelected(QModelIndex index);
    void slotTrajDirSelected(QModelIndex index);
    void slotOsmFileLoaded(const QString &filename);
    void slotTrajFileLoaded(QString &filename);
    
private:
    void loadSettings();
    void saveSettings();
    void saveStatusLog();
    
    QFileSystemModel                *osmFileModel;
    QFileSystemModel                *trajFileModel;
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
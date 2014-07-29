#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_

#include <QObject>

class Trajectories;

class Trajectory : public QObject
{
public:
	Trajectory(QObject *parent, int id_trajectory, Trajectories* trajectories);
    ~Trajectory(void);
    
    void draw() const;
	//virtual void pickEvent(int pick_mode, osg::Vec3 position);
    
protected:
	//virtual void updateImpl(void);
    
private:
	int                                     id_trajectory_;
	Trajectories                            *trajectories_;
};

#endif // TRAJECTORY_H_
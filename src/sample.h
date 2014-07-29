#ifndef SAMPLE_H_
#define SAMPLE_H_

#include <osg/observer_ptr>

#include "renderable.h"

class Trajectories;

class Sample : public Renderable
{
public:
	Sample(int id_sample, Trajectories* trajectories);
	virtual ~Sample(void);
    
	META_Renderable(Sample);
    
	virtual void pickEvent(int pick_mode, osg::Vec3 position);
    
protected:
	virtual void updateImpl(void);
    
private:
	int                                     id_sample_;
	osg::observer_ptr<Trajectories>			trajectories_;
};

#endif // SAMPLE_H_
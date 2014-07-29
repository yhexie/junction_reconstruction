#include "osg_utility.h"
#include "trajectories.h"

#include "sample.h"

Sample::Sample(int id_sample, Trajectories* trajectories)
:id_sample_(id_sample), trajectories_(trajectories)
{
    
}

Sample::~Sample(void)
{
    
}

void Sample::pickEvent(int pick_mode, osg::Vec3 position)
{
	if (pick_mode == osgGA::GUIEventAdapter::MODKEY_CTRL)
		trajectories_->toggleSampleHighlight(id_sample_);
	else if (pick_mode == osgGA::GUIEventAdapter::MODKEY_ALT)
		trajectories_->showSampleInfo(id_sample_);
	else if (pick_mode == osgGA::GUIEventAdapter::MODKEY_SHIFT)
		trajectories_->toggleTrajectory(trajectories_->data()->at(id_sample_).id_trajectory);
    
	return;
}

void Sample::updateImpl(void)
{
	if (!trajectories_.valid())
		return;
    
	osg::Vec3 offset(0.0, 0.0, 1.0);
	const PclPoint& point = trajectories_->data()->at(id_sample_);
	content_root_->addChild(OSGUtility::drawSphere(point.castCoordinate<osg::Vec3>()-trajectories_->center() + offset, 3, osg::Vec4(1.0, 0.0, 0.0, 1.0)));
    
	return;
}
#include "camera_manipulator.h"

CameraManipulator::CameraManipulator(int flags)
:osgGA::TrackballManipulator(flags)
{
    
}

CameraManipulator::CameraManipulator(const TrackballManipulator& tm, const osg::CopyOp& copyOp)
:osgGA::TrackballManipulator(tm, copyOp)
{
    
}

CameraManipulator::~CameraManipulator(void)
{
    
}

bool CameraManipulator::performMovementLeftMouseButton(const double eventTimeDelta, const double dx, const double dy)
{
	// just bypass it...
	return true;
}
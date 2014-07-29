#ifndef CAMERA_MANIPULATOR_H_
#define CAMERA_MANIPULATOR_H_

#include <osgGA/TrackballManipulator>

class CameraManipulator : public osgGA::TrackballManipulator
{
public:
	CameraManipulator( int flags = DEFAULT_SETTINGS );
	CameraManipulator( const TrackballManipulator& tm, const osg::CopyOp& copyOp = osg::CopyOp::SHALLOW_COPY );
    
	virtual ~CameraManipulator(void);
    
protected:
	virtual bool performMovementLeftMouseButton( const double eventTimeDelta, const double dx, const double dy );
    
};


#endif // CAMERA_MANIPULATOR_H_
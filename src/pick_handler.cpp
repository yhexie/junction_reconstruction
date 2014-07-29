#include "common.h"
#include "renderable.h"
#include "osg_utility.h"

#include "pick_handler.h"

PickHandler::PickHandler(void)
{
}

PickHandler::~PickHandler(void)
{
}

void PickHandler::getUsage(osg::ApplicationUsage &usage) const
{
    usage.addKeyboardMouseBinding("L-Click + ModKey", "Pick Renderable Object");
	usage.addKeyboardMouseBinding("R-Click + ModKey", "Pick Individual Point");
    return;
}

bool PickHandler::handle(const osgGA::GUIEventAdapter& ea,osgGA::GUIActionAdapter& aa)
{
    switch(ea.getEventType())
    {
        case(osgGA::GUIEventAdapter::PUSH):
        {
			if (ea.getModKeyMask() == 0)
				return false;
            
            osgViewer::View* view = dynamic_cast<osgViewer::View*>(&aa);
            if (view == NULL)
                return false;
            
            osgUtil::LineSegmentIntersector::Intersection intersection;
            osg::NodePath node_path;
            Renderable* renderable = NULL;
			if (ea.getButtonMask() == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON)
				renderable = OSGUtility::computeIntersection<Renderable>(view, ea, intersection, node_path);
			else if (ea.getButtonMask() == osgGA::GUIEventAdapter::RIGHT_MOUSE_BUTTON)
				renderable = OSGUtility::computePointIntersection<Renderable>(view, ea, intersection, node_path);
            if (renderable == NULL)
                return false;
            
            renderable->pickEvent(ea.getModKeyMask(), intersection.getWorldIntersectPoint());
            return true;
        }
            break;
        default:
            return false;
    }
    
    return false;
}
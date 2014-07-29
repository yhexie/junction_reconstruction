#ifndef UpdateCallback_H_
#define UpdateCallback_H_

#include <osg/NodeCallback>

class UpdateCallback : public osg::NodeCallback
{
public:
    UpdateCallback(void);
    virtual ~UpdateCallback(void);
    
    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);
};

#endif // UpdateCallback_H_
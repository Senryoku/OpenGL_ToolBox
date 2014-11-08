#include <OpenGLObject.hpp>

OpenGLObject::OpenGLObject() :
	_handle(0)
{
}

OpenGLObject::OpenGLObject(GLuint handle) :
	_handle(handle)
{
}

OpenGLObject::~OpenGLObject()
{
	_handle = 0;
}

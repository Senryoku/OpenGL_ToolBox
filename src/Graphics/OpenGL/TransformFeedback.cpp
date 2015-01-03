#include <TransformFeedback.hpp>

void TransformFeedback::init()
{
	if(_handle != 0)
		cleanup();
	glGenTransformFeedbacks(1, &_handle);
}

void TransformFeedback::cleanup()
{
	glDeleteTransformFeedbacks(1, &_handle);
}


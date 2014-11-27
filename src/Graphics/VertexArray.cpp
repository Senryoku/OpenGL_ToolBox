#include <VertexArray.hpp>

VertexArray::~VertexArray()
{
	glDeleteVertexArrays(1, &_handle);
}

void VertexArray::init()
{
	glGenVertexArrays(1, &_handle);
}

void VertexArray::bind() const
{
	glBindVertexArray(_handle);
}

void VertexArray::unbind()
{
	glBindVertexArray(0);
}

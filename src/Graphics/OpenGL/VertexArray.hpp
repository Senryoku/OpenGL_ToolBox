#pragma once

#include <OpenGLObject.hpp>
#include <Buffer.hpp>

/**
 * Vertex Array Object (VAO)
 *
**/
class VertexArray : public OpenGLObject
{
public:
	virtual ~VertexArray();

	void init(); 
	
	void bind() const;

	static void unbind();
	
	/**
	 * glVertexAttribPointer
	**/
	void attribute(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid* pointer) const;
	
	void cleanup();
private:
};

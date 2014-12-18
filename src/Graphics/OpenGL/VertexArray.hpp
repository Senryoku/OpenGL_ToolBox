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
	/**
	 * Destructor
	**/
	virtual ~VertexArray();
	
	/**
	 * Initialization of the OpenGL object
	**/
	void init(); 
	
	/**
	 * Binds the VOA.
	**/
	void bind() const;

	/**
	 * Unbind any VAO currently bound.
	**/
	static void unbind();
	
	/**
	 * glVertexAttribPointer
	**/
	void attribute(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid* pointer) const;
	
	/**
	 * Destroys the OpenGLObject.
	**/ 
	void cleanup();
private:
};

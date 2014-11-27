#pragma once

#include <OpenGLObject.hpp>
#include <Buffer.hpp>

/**
 * TODO
 *
**/
class VertexArray : public OpenGLObject
{
public:
	virtual ~VertexArray();

	void init(); 
	
	void bind() const;

	static void unbind();
	
	/* TODO ?
	void link(const Buffer& buf)
	{
		add buf;
	
		buf.bind();
		glEnableVertexAttribArray(x);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
		buf.unbind();
	}
	*/
private:
};

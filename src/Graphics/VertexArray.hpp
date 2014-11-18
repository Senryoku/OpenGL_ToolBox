#pragma once

#include <OpenGLObject.hpp>

/**
 * TODO
 *
**/
class VertexArray : public OpenGLObject
{
	glGenVertexArrays(1, &vaoID[0]); // Créer le VAO
	glBindVertexArray(vaoID[0]); // Lier le VAO pour l'utiliser
	glBindVertexArray(0); // Délier le VAO  
	
	void link(const Buffer& buf)
	{
		add buf;
	
		buf.bind();
		glEnableVertexAttribArray(x);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
		buf.unbind();
	}
};

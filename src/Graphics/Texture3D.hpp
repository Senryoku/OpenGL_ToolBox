#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include <string>

#include <Texture.hpp>

/**
 * Texture 2D
 * @todo Add a LOT of options !
 * @todo Change default parameters ?
**/
class Texture3D : public Texture
{
public:
	Texture3D() =default;
	
	Texture3D(GLenum type) :
		Texture(),
		_type(type)
	{
	}
	
	void create(const void* data, size_t width, size_t height, size_t depth, int compCount);
	
	virtual void bind(unsigned int unit = 0) const override;
	
	virtual void unbind(unsigned int unit = 0) const override;
	
	virtual GLuint getBound(unsigned int unit = 0) const override
	{
		activeUnit(unit);
		GLint r;
		glGetIntegerv(GL_TEXTURE_BINDING_3D, &r);
		return static_cast<GLuint>(r);
	}
private:
	GLenum	_type = GL_UNSIGNED_BYTE;
	
	virtual GLenum getType() const override { return GL_TEXTURE_3D; }
};

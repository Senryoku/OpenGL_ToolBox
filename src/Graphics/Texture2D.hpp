#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include <string>

#include <Texture.hpp>

/**
 * Texture 2D
 * @todo Add a LOT of options !
**/
class Texture2D : public Texture
{
public:
	Texture2D() =default;
	
	Texture2D(GLenum type) :
		Texture(),
		_type(type)
	{
	}

	void load(const std::string& Path);
	void create(const void* data, size_t width, size_t height, int compCount);
	
	virtual void bind(int UnitTexture = 0) const;
	
	static void unbind();
private:
	GLenum	_type = GL_UNSIGNED_BYTE;
};

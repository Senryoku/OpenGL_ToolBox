#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include <string>

#include <Texture.hpp>

class Texture2D : public Texture
{
public:
	void load(const std::string& Path);
	void create(const void* data, size_t width, size_t height, int compCount);
	
	virtual void bind(int UnitTexture = 0) const;
	
	static void unbind();
private:
};

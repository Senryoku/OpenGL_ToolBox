#include <Texture2D.hpp>

#include <iostream>
#include "stb_image.hpp"

void Texture2D::load(const std::string& Path)
{
	int x, y, n;
	unsigned char *data = stbi_load(Path.c_str(), &x, &y, &n, 0);
	if(data == nullptr)
		std::cerr << "Error Loading Texture " << Path << std::endl;
	create(data, x, y, n);
	stbi_image_free(data);
}

void Texture2D::create(const void* data, size_t width, size_t height, int compCount)
{	
	GLenum format = getFormat(compCount);
	create(data, width, height, format, format);
}

void Texture2D::create(const void* data, size_t width, size_t height, GLint internalFormat, GLenum format, bool generateMipmaps)
{
	cleanup();
	glGenTextures(1, &_handle);
	
	bind();
	
	glTexImage2D(GL_TEXTURE_2D, 
				 0,
				 internalFormat,
	 			 static_cast<GLsizei>(width),
				 static_cast<GLsizei>(height),
				 0,
				 format,
				 _pixelType,
				 data
			); 

	// Default Parameters
	if(generateMipmaps)
		set(MinFilter, GL_LINEAR_MIPMAP_LINEAR);
	else
		set(MinFilter, GL_LINEAR);
	set(MagFilter, GL_LINEAR);
	set(WrapS, GL_REPEAT);
	set(WrapT, GL_REPEAT);
	
	// Mmh ?
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	glSamplerParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
	
	if(generateMipmaps)
		glGenerateMipmap(GL_TEXTURE_2D);
	
	unbind();
}

void Texture2D::bind(unsigned int unit) const
{
	activeUnit(unit);
	glBindTexture(GL_TEXTURE_2D, _handle);
}

void Texture2D::unbind(unsigned int unit) const
{
	activeUnit(unit);
	glBindTexture(GL_TEXTURE_2D, 0);
}

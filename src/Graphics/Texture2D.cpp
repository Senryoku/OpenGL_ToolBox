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
	cleanup();
	
	glEnable(GL_TEXTURE_2D);
	
	glGenTextures(1, &_handle);
	bind();
	
	GLenum format;
	switch(compCount)
	{
		case 1 :
			format = GL_RED;
			break;
		case 2 :
			format = GL_RG;
			break;
		case 3 :
			format = GL_RGB;
			break;
		case 4 :
			format = GL_RGBA;
			break;
		default:
			format = GL_RGBA;
			break;
	}
	
	glTexImage2D(GL_TEXTURE_2D, 
				 0,
				 format,
	 			 static_cast<GLsizei>(width),
				 static_cast<GLsizei>(height),
				 0,
				 format,
				 _type,
				 data); 	
	glGenerateMipmap(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	glSamplerParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
	
	unbind();
}

void Texture2D::bind(int UnitTexture) const
{
	glActiveTexture(GL_TEXTURE0 + UnitTexture);
	glBindTexture(GL_TEXTURE_2D, _handle);
}

void Texture2D::unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

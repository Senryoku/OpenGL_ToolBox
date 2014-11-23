#include <Texture3D.hpp>

#include <iostream>

void Texture3D::create(const void* data, size_t width, size_t height, size_t depth, int compCount)
{
	cleanup();
	
	glEnable(GL_TEXTURE_3D);
	
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
	
	glTexImage3D(GL_TEXTURE_3D, 
				 0,
				 format,
	 			 static_cast<GLsizei>(width),
				 static_cast<GLsizei>(height),
				 static_cast<GLsizei>(depth),
				 0,
				 format,
				 _type,
				 data
				); 

	// Default Parameters
	set(MinFilter, GL_LINEAR_MIPMAP_LINEAR);
	set(MagFilter, GL_LINEAR);
	set(WrapS, GL_REPEAT);
	set(WrapT, GL_REPEAT);
	
	// Mmh ?
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	glSamplerParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
	
	glGenerateMipmap(GL_TEXTURE_3D);
	
	unbind();
}

void Texture3D::bind(unsigned int unit) const
{
	activeUnit(unit);
	glBindTexture(GL_TEXTURE_3D, _handle);
}

void Texture3D::unbind(unsigned int unit) const
{
	activeUnit(unit);
	glBindTexture(GL_TEXTURE_3D, 0);
}

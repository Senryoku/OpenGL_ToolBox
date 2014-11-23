#include <CubeMap.hpp>

#include <iostream>
#include "stb_image.hpp"

void CubeMap::load(const std::array<std::string, 6>& paths)
{
	int x, y, n;
	std::array<void*, 6> data;
	for(size_t i = 0; i < 6; ++i)
	{
		data[i] = stbi_load(paths[i].c_str(), &x, &y, &n, 0);
		if(data[i] == nullptr)
			std::cerr << "Error Loading Texture " << paths[i] << std::endl;
	}
	create(data, x, y, n);
	for(size_t i = 0; i < 6; ++i)
		stbi_image_free(data[i]);
}

void CubeMap::create(const std::array<void*, 6>& data, size_t width, size_t height, int compCount)
{
	cleanup();
	
	glEnable(GL_TEXTURE_CUBE_MAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	
	glGenTextures(1, &_handle);
	bind();
	
	GLenum format = GL_RGBA;
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
	
	for(size_t i = 0; i < 6; ++i)
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
					 0,
					 compCount,
					 static_cast<GLsizei>(width),
					 static_cast<GLsizei>(height),
					 0,
					 format,
					 GL_UNSIGNED_BYTE,
					 data[i]);
	/*
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	*/
	
	//set(MinFilter, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	set(MagFilter, GL_LINEAR);
	set(WrapS, GL_CLAMP_TO_EDGE);
	set(WrapT, GL_CLAMP_TO_EDGE);
	set(WrapR, GL_CLAMP_TO_EDGE);
	
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	glSamplerParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
	
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
	
	unbind();
}

void CubeMap::bind(unsigned int unit) const
{
	activeUnit(unit);
	glBindTexture(GL_TEXTURE_CUBE_MAP, _handle);
}

void CubeMap::unbind(unsigned int unit) const
{
	activeUnit(unit);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

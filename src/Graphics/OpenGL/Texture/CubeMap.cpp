#include <CubeMap.hpp>

#include <iostream>
#include "stb_image.hpp"

CubeMap::CubeMap(GLenum pixelType) :
	Texture(pixelType)
{
}

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
	GLenum format = getFormat(compCount);
	create(data, width, height, format, format);
}

void CubeMap::create(const std::array<void*, 6>& data, size_t width, size_t height, GLint internalFormat, GLenum format, bool generateMipmaps)
{
	cleanup();
	
	glEnable(GL_TEXTURE_CUBE_MAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	
	glGenTextures(1, &_handle);
	bind();
	
	for(size_t i = 0; i < 6; ++i)
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
					 0,
					 internalFormat,
					 static_cast<GLsizei>(width),
					 static_cast<GLsizei>(height),
					 0,
					 format,
					 _pixelType,
					 data[i]);
	
	if(generateMipmaps)
		set(MinFilter, GL_LINEAR_MIPMAP_LINEAR);
	else
		set(MinFilter, GL_LINEAR);
	set(MagFilter, GL_LINEAR);
	set(MagFilter, GL_LINEAR);
	set(WrapS, GL_CLAMP_TO_EDGE);
	set(WrapT, GL_CLAMP_TO_EDGE);
	set(WrapR, GL_CLAMP_TO_EDGE);
	
	GLfloat maxAniso = 0.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
	glSamplerParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
	
	if(generateMipmaps)
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

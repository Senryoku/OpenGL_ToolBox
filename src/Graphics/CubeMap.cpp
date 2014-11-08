#include <CubeMap.hpp>

#include <iostream>
#include "stb_image.hpp"

void CubeMap::load(const std::array<std::string, 6>& Paths)
{
	int x, y, n;
	std::array<void*, 6> Data;
	for(size_t i = 0; i < 6; ++i)
	{
		Data[i] = stbi_load(Paths[i].c_str(), &x, &y, &n, 0);
		if(Data[i] == nullptr)
			std::cerr << "Error Loading Texture " << Paths[i] << std::endl;
	}
	create(Data, x, y, n);
	for(size_t i = 0; i < 6; ++i)
		stbi_image_free(Data[i]);
}

void CubeMap::create(const std::array<void*, 6>& Data, size_t width, size_t height, int compCount)
{
	cleanup();
	
	glEnable(GL_TEXTURE_CUBE_MAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	
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
	
	for(size_t i = 0; i < 6; ++i)
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
					 0,
					 compCount,
					 static_cast<GLsizei>(width),
					 static_cast<GLsizei>(height),
					 0,
					 format,
					 GL_UNSIGNED_BYTE,
					 Data[i]);
	
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	
	unbind();
}

void CubeMap::bind(int UnitTexture) const
{
	glActiveTexture(GL_TEXTURE0 + UnitTexture);
	glBindTexture(GL_TEXTURE_CUBE_MAP, _handle);
}

void CubeMap::unbind()
{
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

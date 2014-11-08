#include <Texture.hpp>

Texture::~Texture()
{
	cleanup();
}

void Texture::cleanup()
{
	glDeleteTextures(1, &_handle);
}

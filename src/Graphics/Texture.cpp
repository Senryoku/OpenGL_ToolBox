#include <Texture.hpp>

Texture::~Texture()
{
	cleanup();
}

void Texture::cleanup()
{
	glDeleteTextures(1, &_handle);
}

void Texture::bindImage(GLuint unit, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format) const
{
	glBindImageTexture(unit, _handle, level, layered, layer, access, format);
}
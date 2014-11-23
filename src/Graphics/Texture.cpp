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

void Texture::generateMipmaps() const
{
	bind();
	glGenerateMipmap(getType());
	unbind();
}

////////////////////////////////////////////////////////////////////////////////
// Binder

Texture::Binder::Binder(const Texture& t, unsigned int unit) :
	_unit(unit),
	_tex(t)
{
	_prevBound = _tex.getBound(unit);
	if(_prevBound != _tex.getName())
		_tex.bind(_unit);
}

Texture::Binder::~Binder()
{
	if(_prevBound != _tex.getName())
	{
		activeUnit(_unit);
		if(_prevBound != 0)
			glBindTexture(_tex.getType(), _prevBound);
		else
			glBindTexture(_tex.getType(), 0);
	}
}

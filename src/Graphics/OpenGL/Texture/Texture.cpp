#include <Texture.hpp>

Texture::Texture(GLenum pixelType) :
	OpenGLObject(),
	_pixelType(pixelType)
{
}

Texture::Texture(GLenum pixelType, GLuint handle) : 
	OpenGLObject(handle)
{
	if(!isTexture(handle))
		std::cerr << "Error constructing texture: Provided OpenGL name isn't a texture name." << std::endl;
}

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
	Binder B(*this);
	glGenerateMipmap(getType());
}

GLenum Texture::getFormat(GLuint compCount)
{	
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
	return format;
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

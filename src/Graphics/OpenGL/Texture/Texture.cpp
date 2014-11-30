#include <Texture.hpp>

#include <stb_image_write.hpp>

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

GLuint Texture::getCompCount(GLenum format)
{	
	switch(format)
	{
		case GL_RED :
			return 1;
			break;
		case  GL_RG :
			return 2;
			break;
		case GL_RGB :
			return 3;
			break;
		case GL_RGBA :
			return 4;
			break;
		default:
			return 0;
			break;
	}
}

void Texture::dump(const std::string& path) const
{
	GLint width, height, format;
	glGetTexLevelParameteriv(getType(), 0, GL_TEXTURE_WIDTH, &width);
	glGetTexLevelParameteriv(getType(), 0, GL_TEXTURE_HEIGHT, &height);
	glGetTexLevelParameteriv(getType(), 0, GL_TEXTURE_INTERNAL_FORMAT, &format);
	if(_pixelType == GL_UNSIGNED_BYTE) {
		GLubyte* data = nullptr;
		data = new GLubyte[getCompCount(format) * width * height];
	
		glGetTexImage(getType(), 0, format, _pixelType, data);
		stbi_write_png(path.c_str(), width, height, getCompCount(format), data, 0);
		
		delete[] data;
	} else {
		return;
	}
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

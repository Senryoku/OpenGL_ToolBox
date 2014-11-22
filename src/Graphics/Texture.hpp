#pragma once

#include <OpenGLObject.hpp>

/**
 * Describes a texture living on the GPU's memory.
 * @todo Store more information about the managed texture (type, format...)
 * 		 in order to provide easier use (image binding for example).
**/
class Texture : public OpenGLObject
{
public:
	Texture() =default;
	
	Texture(GLuint handle) : 
		OpenGLObject(handle)
	{
	}
	
	Texture(const Texture&) =default;
	Texture(Texture&&) =default;
	Texture& operator=(const Texture&) =default;
	Texture& operator=(Texture&&) =default;
		
	virtual ~Texture();

	virtual void bind(unsigned int unit = 0) const {}
	
	virtual void unbind(unsigned int unit = 0) const {}
	
	/**
	 * TODO
	**/
	virtual void bindImage(GLuint unit, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format) const;
	
	inline static void activeUnit(unsigned int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
	}
protected:
	void cleanup();
};

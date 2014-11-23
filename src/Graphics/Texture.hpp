#pragma once

#include <iostream>

#include <OpenGLObject.hpp>
#include <ParameterBinding.hpp>

/**
 * Describes a texture living on the GPU's memory.
 * @todo Store more information about the managed texture (type, format...)
 * 		 in order to provide easier use (image binding for example).
 * @todo Use Sampler class ?
 * @todo Parameter value enums
**/
class Texture : public OpenGLObject
{
public:
	/**
	 * Scoped texture Binder.
	 * Texture is unbound at destruction and previously
	 * bound texture is restored.
	 *
	 * Should do nothing if the texture is already bound to
	 * the specified texture unit. However, this will always
	 * change the current active texture unit !
	**/
	class Binder
	{
	public:
		Binder(const Texture& t, unsigned int unit = 0);
		
		~Binder();
	private:
		GLuint			_unit = 0;
		const Texture&	_tex;
		GLuint 			_prevBound = 0;
	};

	enum Parameter
	{
		DepthStencilMode = GL_DEPTH_STENCIL_TEXTURE_MODE,
		BaseLevel = GL_TEXTURE_BASE_LEVEL,
		SwizzleR = GL_TEXTURE_SWIZZLE_R, 
		SwizzleG = GL_TEXTURE_SWIZZLE_G, 
		SwizzleB = GL_TEXTURE_SWIZZLE_B,
		SwizzleA = GL_TEXTURE_SWIZZLE_A,
		// Common with Sampler
		MinFilter = GL_TEXTURE_MIN_FILTER,
		MagFilter = GL_TEXTURE_MAG_FILTER,
		MinLoD = GL_TEXTURE_MIN_LOD,
		MaxLoD = GL_TEXTURE_MAX_LOD,
		LoDBias = GL_TEXTURE_LOD_BIAS,
		WrapS = GL_TEXTURE_WRAP_S,
		WrapT = GL_TEXTURE_WRAP_T,
		WrapR = GL_TEXTURE_WRAP_R,
		BorderColor = GL_TEXTURE_BORDER_COLOR,
		CompareMode = GL_TEXTURE_COMPARE_MODE,
		CompareFunc = GL_TEXTURE_COMPARE_FUNC
	};
	
	Texture() =default;
	
	Texture(GLuint handle) : 
		OpenGLObject(handle)
	{
		if(!isTexture(handle))
			std::cerr << "Error constructing texture: Provided OpenGL name isn't a texture name." << std::endl;
	}
	
	Texture(const Texture&) =default;
	Texture(Texture&&) =default;
	Texture& operator=(const Texture&) =default;
	Texture& operator=(Texture&&) =default;
		
	virtual ~Texture();

	virtual void bind(unsigned int unit = 0) const {}
	
	virtual void unbind(unsigned int unit = 0) const {}
	
	inline virtual bool isValid() const override { return OpenGLObject::isValid() && isTexture(_handle); }
	
	template<typename T>
	void set(Parameter opt, T value) const
	{
		Binder(*this);
		texParameter(getType(), static_cast<GLenum>(opt), value);
	}
	
	template<typename T>
	void setI(Parameter opt, T value) const
	{
		Binder(*this);
		texParameterI(getType(), static_cast<GLenum>(opt), value);
	}
	
	void generateMipmaps() const;
	
	/**
	 * TODO
	**/
	virtual void bindImage(GLuint unit, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format) const;
	
	inline static void activeUnit(unsigned int unit)
	{
		glActiveTexture(GL_TEXTURE0 + unit);
	}
	
	virtual GLuint getBound(unsigned int unit = 0) const =0;
protected:
	void cleanup();
	
	virtual GLenum getType() const =0;
	
	inline static bool isTexture(GLuint name) { return glIsTexture(name) == GL_TRUE; }
};

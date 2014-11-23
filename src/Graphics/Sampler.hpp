#pragma once

#include <iostream>

#include <OpenGLObject.hpp>
#include <ParameterBinding.hpp>

/**
 * Describes a Sampler object.
 * @todo Parameter value enums.
**/
class Sampler : public OpenGLObject
{
public:
	enum Parameter
	{
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

	Sampler() =default;
	
	Sampler(GLuint handle) : 
		OpenGLObject(handle)
	{
		if(!isSampler(handle))
			std::cerr << "Error constructing Sampler: Provided OpenGL name isn't a Sampler name." << std::endl;
	}
	
	Sampler(const Sampler&) =default;
	Sampler(Sampler&&) =default;
	Sampler& operator=(const Sampler&) =default;
	Sampler& operator=(Sampler&&) =default;
		
	virtual ~Sampler();

	void bind(unsigned int unit = 0) const;
	
	void unbind(unsigned int unit = 0) const;
	
	template<typename T>
	void set(Parameter opt, T value) const
	{
		samplerParameter(_handle, static_cast<GLenum>(opt), value);
	}
	
	template<typename T>
	void setI(Parameter opt, T value) const
	{
		samplerParameterI(_handle, static_cast<GLenum>(opt), value);
	}
	
	inline virtual bool isValid() const override { return OpenGLObject::isValid() && isSampler(_handle); }
	
protected:
	void cleanup();
	
	inline static bool isSampler(GLuint name) { return glIsSampler(name) == GL_TRUE; }
};

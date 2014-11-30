#pragma once
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>
#include <ComputeShader.hpp>
#include <UniformBinding.hpp>

/**
 * Program
 *
 * OpenGL Shader Program
**/
class Program : public OpenGLObject
{
public:
	Program();
	virtual ~Program();
	
	/**
	 * Attach a shader to this Program by its OpenGL name.
	 *
	 * Be careful using this (especially for a ComputeShader)
	 * @param shaderName Shader's OpenGL name.
	 * @see attachShader(Shader&)
	 * @see attachShader(ComputeShader&)
	**/
	void attachShader(GLuint shaderName);
	
	/**
	 * Attach a shader to this Program.
	 * @see attachShader(GLuint)
	 * @see attachShader(ComputeShader&)
	**/
	void attachShader(Shader& shader);
	
	/**
	 * Attach a compute shader to this Program.
	 * (Overload of attachShader(Shader&), also links the compute shader
	 * to this particular program).
	 * @see attachShader(GLuint)
	 * @see attachShader(Shader&)
	**/
	void attachShader(ComputeShader& cshader);
	
	/**
	 * Linking the program (all shaders must me compiled).
	**/
	void link();
	
	/**
	 * Use (~bind) this Shader Program.
	**/
	void use() const;
	
	/**
	 * @return true if the program is set and linked, false otherwise.
	**/
	inline bool isValid() const { return _handle != 0 && _linked; }
	
	/**
	 * Query for the location of the specified uniform.
	 * (Simple wrapper around glGetUniformLocation)
	 *
	 * @param name Uniform's name
	 * @return -1 if the uniform isn't defined in any of the attached
	 *	shaders, its location otherwise.
	**/
	GLint getUniformLocation(const std::string& name) const;
	
	/**
	 * Call the wrapper around glUniform* declared for T in UniformBinding.hpp
	 * @see UniformBinding.hpp
	 * @see setUniform(GLint program, GLuint location, const T& value);
	**/
	template<typename T>
	inline void setUniform(const std::string& name, const T& value) const
	{
		::setUniform(getName(), getUniformLocation(name), value);
	}
	
	static void useNone();
	
private:
	bool	_linked = false;
};


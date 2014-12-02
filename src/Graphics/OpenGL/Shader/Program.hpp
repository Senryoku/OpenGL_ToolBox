#pragma once
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>
#include <ComputeShader.hpp>
#include <UniformBinding.hpp>
#include <Buffer.hpp>

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
	void attachShader(const Shader& shader);
	
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
	inline bool isValid() const override { return _handle != 0 && _linked; }
	
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
	
	/**
	 * glGetProgramResourceIndex
	 * @param interface Type of searched resource (ex: GL_SHADER_STORAGE_BLOCK)
	 * @param name Name of the resource
	 * @return Index of name in the program, or GL_INVALID_INDEX if not found
	**/
	GLuint getResourceIndex(GLenum interface, const std::string& name) const;
	
	/**
	 * glGetuniformBlockIndex
	 * @param name Name of the searched Uniform Block
	 * @return Index of name in the program, or GL_INVALID_INDEX if not found
	 * @see bindUniformBlock(GLuint uniformBlockIndex, GLuint uniformBlockBindingPoint)
	**/
	GLuint getUniformBlockIndex(const std::string& name) const;
	
	/**
	 * Assign a binding point to an active uniform block.
	 * @param uniformBlockIndex Index of a uniform block in one of the program's shaders
	 * @param uniformBlockBindingPoint Binding point of an active Uniform Buffer Object (UBO) to assign to this uniform block
	**/
	void bindUniformBlock(GLuint uniformBlockIndex, GLuint uniformBlockBindingPoint) const;
	
	/**
	 * Assign a Uniform Buffer Object (UBO) to an active uniform block.
	 * @param name Name of a uniform block in one of the program's shaders
	 * @param uniformBlockBindingPoint Binding point of an active Uniform Buffer Object (UBO) to assign to this uniform block
	**/
	void bindUniformBlock(const std::string& name, GLuint uniformBlockBindingPoint) const;
	
	/**
	 * Assign a Uniform Buffer Object (UBO) to an active uniform block.
	 * @param name Name of a uniform block in one of the program's shaders
	 * @param uniformBuffer Uniform Buffer Object (UBO) to assign to this uniform block
	**/
	void bindUniformBlock(const std::string& name, const UniformBuffer& uniformBuffer) const;
	
	/**
	 * Unbind any currently bound shader program.
	**/
	static void useNone();
	
private:
	bool	_linked = false;
};


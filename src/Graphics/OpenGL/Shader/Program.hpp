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
	 * Initializes the OpenGL object.
	**/
	void init();
	
	/**
	 * Destroys the OpenGL object.
	**/
	void cleanup();
	
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
	 * Specifies values to record in transform feedback buffers. This have to be called before linking.
	 * @param count Number of values to record.
	 * @param varyings Array of null terminated char strings specifying the names of the varyings to record.
	 * @param intervealed Specifies the mode of capture, if true, values will be intervealed (packed in one buffer). They will be separated otherwise (one buffer per value).
	**/
	inline void setTransformFeedbackVaryings(GLsizei count, const char** varyings, bool intervealed) const;
	
	/**
	 * Specifies values to record in transform feedback buffers. Syntactic sugar around setTransformFeedbackVaryings(GLsizei, const char**, bool).
	 * This have to be called before linking.
	 * @param varyings Array of null terminated char strings specifying the names of the varyings to record.
	 * @param intervealed Specifies the mode of capture, if true, values will be intervealed (packed in one buffer). They will be separated otherwise (one buffer per value).
	 * @see setTransformFeedbackVaryings(GLsizei, const char**, bool).
	**/
	template<size_t Count>
	inline void setTransformFeedbackVaryings(const std::array<const char*, Count>& varyings, bool intervealed) const;
	
	/**
	 * Linking the program (all shaders must be compiled).
	**/
	void link();
	
	/**
	 * Use (~bind) this Shader Program.
	**/
	void use() const;
	
	/**
	 * @return true if the program is set and linked, false otherwise.
	**/
	inline bool isValid() const override;

	/**
	 * @param name Name of the vertex shader attribute
	 * @return Location of the queried attribute
	 * @see glGetAttribLocation
	**/
	inline GLint getAttribLocation(const std::string& name) const;
	
	/**
	 * Associate a vertex attribute index with named attribute variable
	 * @param index Index of the vertex attribute to be bound
	 * @param name Name of the attribute to which index is to be bound
	 * @see glBindAttribLocation
	**/
	inline void bindAttribLocation(GLuint index, const std::string& name) const;
	
	/**
	 * Query for the location of the specified uniform.
	 * (Simple wrapper around glGetUniformLocation)
	 *
	 * @param name Uniform's name
	 * @return -1 if the uniform isn't defined in any of the attached
	 *		   shaders, its location otherwise.
	**/
	GLint getUniformLocation(const std::string& name) const;
	
	/**
	 * Call the wrapper around glUniform* declared for T in UniformBinding.hpp
	 * @see UniformBinding.hpp
	 * @see setUniform(GLint program, GLuint location, const T& value);
	**/
	template<typename T>
	inline void setUniform(const std::string& name, const T& value) const;
	
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
	
	/**
	 * @return Program currently bound.
	**/
	inline static GLint getCurrent();
	
	/**
	 * @return Location of uniform uniformName in specified program.
	**/
	inline static GLint getUniformLocation(GLuint programName, const std::string& uniformName);
	
private:
	bool	_linked = false;	///< Link status
};

/**
 * Utility function for quick uniform setting on currently bound program.
 * @param name Name of the uniform.
 * @param value New value of the uniform.
**/
template<typename T>
void setUniform(const std::string& name, const T& value)
{
	::setUniform(Program::getUniformLocation(Program::getCurrent(), name), value);
}

// Inline functions definitions

inline bool Program::isValid() const
{ 
	return _handle != 0 && _linked;
}

inline GLint Program::getAttribLocation(const std::string& name) const
{
	return glGetAttribLocation(_handle, name.c_str());
}

inline void Program::bindAttribLocation(GLuint index, const std::string& name) const
{
	return glBindAttribLocation(_handle, index, name.c_str());
}
	
template<typename T>
inline void Program::setUniform(const std::string& name, const T& value) const
{
	auto loc = getUniformLocation(name);
	if(loc < 0)
		std::cerr << "Warning: Uniform '" << name << "' not found (" << __PRETTY_FUNCTION__ << ")." << std::endl;
	else
		::setUniform(getName(), loc, value);
}

inline void Program::setTransformFeedbackVaryings(GLsizei count, const char** varyings, bool intervealed) const
{
	glTransformFeedbackVaryings(_handle, count, varyings, (intervealed) ? GL_INTERLEAVED_ATTRIBS : GL_SEPARATE_ATTRIBS);
}
	
template<size_t Count>
inline void Program::setTransformFeedbackVaryings(const std::array<const char*, Count>& varyings, bool intervealed) const
{
	glTransformFeedbackVaryings(_handle, static_cast<GLsizei>(Count), varyings.data(),  (intervealed) ? GL_INTERLEAVED_ATTRIBS : GL_SEPARATE_ATTRIBS);
}

inline GLint Program::getCurrent()
{
	GLint name;
	glGetIntegerv(GL_CURRENT_PROGRAM, &name);
	return name;
}

inline GLint Program::getUniformLocation(GLuint programName, const std::string& uniformName)
{
	assert(glIsProgram(programName));
	return glGetUniformLocation(programName, uniformName.c_str());
}

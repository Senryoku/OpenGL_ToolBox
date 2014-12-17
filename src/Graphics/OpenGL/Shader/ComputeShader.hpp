#pragma once

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>

class Program;

/**
 * @see Shader
 * @see Program
**/
class ComputeShader : public Shader
{
public:
	/// Structure containing size information about Compute shader workgroups
	struct WorkgroupSize
	{
		int x = 0, ///< X Size
			y = 0, ///< Y Size
			z = 0; ///< Z Size
	};

	/**
	 * Default constructor
	 * @param standalone If true, attach the shader to a new dedicated program.
	**/
	ComputeShader(bool standalone = true);
	
	/**
	 * Destructor
	**/
	~ComputeShader();
	
	/**
	 * Compiles and, if set as standalone, links the associated Program.
	 * @see Shader::compile()
	**/
	void compile();
	
	/**
	 * Use the associated Program.
	 * @see Program::use()
	**/
	void use() const;
	
	/**
	 * Starts the execution of the compute shader
	 * @param x Number of workgroups to start on the x 'axis'
	 * @param y Number of workgroups to start on the y 'axis'
	 * @param z Number of workgroups to start on the z 'axis'
	**/
	inline void compute(GLint x, GLint y = 1, GLint z = 1) const;
	
	/**
	 * @return Associated Program
	 * @see Program
	**/
	inline Program& getProgram();
	
	/**
	 * @return Associated Program
	 * @see Program
	**/
	inline const Program& getProgram() const;
	
	/**
	 * Note: Shader must be compiled and associated to a linked Program.
	 * @return Workgroup sizes as specified in the compute shader.
	 * @see compile()
	 * @see Program
	**/
	const WorkgroupSize& getWorkgroupSize();
	
	// Static functions
	
	/**
	 * @see glMemoryBarrier
	**/
	inline static void memoryBarrier(GLbitfield BF = GL_ALL_BARRIER_BITS); 
	
	/**
	 * Starts the execution of the curently 'bound' compute shader
	 * @param x Number of workgroups to start on the x 'axis'
	 * @param y Number of workgroups to start on the y 'axis'
	 * @param z Number of workgroups to start on the z 'axis'
	**/
	inline static void dispatchCompute(GLint x, GLint y = 1, GLint z = 1);
	
private:
	bool 			_standalone = true;
	Program*		_program = nullptr;
	WorkgroupSize	_workgroupSize;
	
	void initOGL();
	void initProgram();
	
	friend class Program;
};

// Inline functions declarations

inline Program& ComputeShader::getProgram()
{
	assert(_program != nullptr);
	return *_program;
}

inline const Program& ComputeShader::getProgram() const
{
	assert(_program != nullptr);
	return *_program;
}
	
inline void ComputeShader::compute(GLint x, GLint y, GLint z) const
{
	if(_standalone) use();
	dispatchCompute(x, y, z);
}

// Static functions

inline void ComputeShader::memoryBarrier(GLbitfield BF)
{
	glMemoryBarrier(BF);
} 

inline void ComputeShader::dispatchCompute(GLint x, GLint y, GLint z)
{
	glDispatchCompute(x, y, z);
}
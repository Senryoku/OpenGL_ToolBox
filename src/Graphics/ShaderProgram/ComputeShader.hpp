#pragma once

#include <iostream>
#include <fstream>
#include <string>
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>

class Program;

class ComputeShader : public Shader
{
	public:
	ComputeShader(bool standalone = true);
	~ComputeShader();
	
	void compile();
	void use();
	void compute(GLint x, GLint y = 1, GLint z = 1);
	
	GLint getProgramID() const;
	
	// STATIC
	inline static void memoryBarrier(GLbitfield BF = GL_ALL_BARRIER_BITS) { glMemoryBarrier(BF); } 
	inline static void dispatchCompute(GLint x, GLint y = 1, GLint z = 1) { glDispatchCompute(x, y, z); }
	
	private:
	bool 		_standalone;
	Program*	_program;
	
	void initOGL();
	void initProgram();
	
	friend class Program;
};


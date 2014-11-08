#include "ComputeShader.hpp"

#include <Program.hpp>

ComputeShader::ComputeShader(bool standalone) :
	Shader(),
	_standalone(standalone)
{
	if(_standalone)
		initProgram();
}

ComputeShader::~ComputeShader()
{
	if(_standalone)
		delete _program;
}

void ComputeShader::initOGL()
{
	if(_handle != 0)
		return;
		
	_handle = glCreateShader(GL_COMPUTE_SHADER);
	if(_handle == 0)
	{
		std::cerr << __FUNCTION__ << " : Error glCreateShader(GL_COMPUTE_SHADER)" << std::endl;
	}
}

void ComputeShader::initProgram()
{
	_program = new Program();
}

GLint ComputeShader::getProgramID() const
{
	return (_standalone) ? _program->getName() : 0;
}
	
void ComputeShader::compile()
{
	Shader::compile();
	if(_standalone)
	{
		_program->attachShader(_handle);
		_program->link();
	}
}

void ComputeShader::use()
{
	_program->use();
}

void ComputeShader::compute(GLint x, GLint y, GLint z)
{
	if(_standalone) use();
	ComputeShader::dispatchCompute(x, y, z);
}

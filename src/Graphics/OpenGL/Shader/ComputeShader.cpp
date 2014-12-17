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
	
void ComputeShader::compile()
{
	Shader::compile();
	if(_standalone)
	{
		_program->attachShader(_handle);
		_program->link();
		glGetProgramiv(_program->getName(), GL_COMPUTE_WORK_GROUP_SIZE, &_workgroupSize.x);
	}
}

void ComputeShader::use() const
{
	_program->use();
}

const ComputeShader::WorkgroupSize& ComputeShader::getWorkgroupSize()
{
	if(_workgroupSize.x < 1 && _program != nullptr)
		glGetProgramiv(_program->getName(), GL_COMPUTE_WORK_GROUP_SIZE, &_workgroupSize.x);
	return _workgroupSize;
}

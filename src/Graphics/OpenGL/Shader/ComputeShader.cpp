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

void ComputeShader::initProgram()
{
	_program = new Program();
}
	
void ComputeShader::compile()
{
	Shader::compile();
	if(_standalone)
	{
		assert(_program != nullptr);
		_program->attachShader(_handle);
		_program->link();
		glGetProgramiv(_program->getName(), GL_COMPUTE_WORK_GROUP_SIZE, &_workgroupSize.x);
	}
}

void ComputeShader::use() const
{
	assert(_program != nullptr);
	_program->use();
}

const ComputeShader::WorkgroupSize& ComputeShader::getWorkgroupSize()
{
	if(_workgroupSize.x < 1 && _program != nullptr)
		glGetProgramiv(_program->getName(), GL_COMPUTE_WORK_GROUP_SIZE, &_workgroupSize.x);
	return _workgroupSize;
}

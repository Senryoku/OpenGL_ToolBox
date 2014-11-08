#include "Program.hpp"

Program::Program() :
	_program(glCreateProgram())
{
	if(_program == 0)
	{
		std::cerr << __FUNCTION__ << " : Error glCreateProgram()" << std::endl;
	}
}

Program::~Program()
{
	glDeleteProgram(_program);
}

void Program::attachShader(Shader& shader)
{
    glAttachShader(_program, shader.getName());
}

void Program::attachShader(ComputeShader& cshader)
{
    glAttachShader(_program, cshader.getName());
	cshader._program = this;
}

void Program::attachShader(GLint shader)
{
    glAttachShader(_program, shader);
}

void Program::link()
{
	int rvalue;
    glLinkProgram(_program);
    glGetProgramiv(_program, GL_LINK_STATUS, &rvalue);
    if (!rvalue)
	{
        std::cerr << __FUNCTION__ << " : Error in linking shader program" << std::endl;
        GLchar log[10240];
        GLsizei length;
        glGetProgramInfoLog(_program, 10239, &length, log);
        std::cerr << "Linker log: " << log << std::endl;
		
		_linked = false;
    } else {
		_linked = true;
	}
}

void Program::use()
{
    glUseProgram(_program);
}

GLint Program::getUniformLocation(const std::string& name) const
{
	return glGetUniformLocation(getID(), name.c_str());
}

void Program::useNone()
{
    glUseProgram(0);
}

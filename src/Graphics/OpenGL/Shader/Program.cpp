#include "Program.hpp"

Program::Program() :
	OpenGLObject(glCreateProgram())
{
	if(_handle == 0)
	{
		std::cerr << __FUNCTION__ << " : Error glCreateProgram()" << std::endl;
	}
}

Program::~Program()
{
	glDeleteProgram(_handle);
}

void Program::attachShader(GLuint shaderName)
{
    glAttachShader(_handle, shaderName);
}

void Program::attachShader(Shader& shader)
{
    glAttachShader(_handle, shader.getName());
}

void Program::attachShader(ComputeShader& cshader)
{
    glAttachShader(_handle, cshader.getName());
	cshader._program = this;
}

void Program::link()
{
	int rvalue;
    glLinkProgram(_handle);
    glGetProgramiv(_handle, GL_LINK_STATUS, &rvalue);
    if (!rvalue)
	{
        std::cerr << __FUNCTION__ << " : Error in linking shader program" << std::endl;
        GLchar log[10240];
        GLsizei length;
        glGetProgramInfoLog(_handle, 10239, &length, log);
        std::cerr << "Linker log: " << log << std::endl;
		
		_linked = false;
    } else {
		_linked = true;
	}
}

void Program::use() const
{
    glUseProgram(_handle);
}

GLint Program::getUniformLocation(const std::string& name) const
{
	return glGetUniformLocation(getName(), name.c_str());
}

void Program::useNone()
{
    glUseProgram(0);
}

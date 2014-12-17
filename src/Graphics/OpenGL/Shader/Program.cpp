#include "Program.hpp"

Program::Program()
{
}

Program::~Program()
{
	cleanup();
}

void Program::init()
{
	if(_handle != 0)
		cleanup();
		
	_handle = glCreateProgram();
	if(_handle == 0)
		std::cerr << __PRETTY_FUNCTION__ << " : Error glCreateProgram()" << std::endl;
}

void Program::cleanup()
{
	glDeleteProgram(_handle);
	_handle = 0;
}

void Program::attachShader(GLuint shaderName)
{
	if(_handle == 0)
		init();
    glAttachShader(_handle, shaderName);
}

void Program::attachShader(const Shader& shader)
{
	if(_handle == 0)
		init();
    glAttachShader(_handle, shader.getName());
}

void Program::attachShader(ComputeShader& cshader)
{
	if(_handle == 0)
		init();
    glAttachShader(_handle, cshader.getName());
	cshader._program = this;
}

void Program::link()
{
	assert(_handle != 0);
	
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
	assert(isValid());
    glUseProgram(_handle);
}

GLint Program::getUniformLocation(const std::string& name) const
{
	assert(isValid());
	return glGetUniformLocation(getName(), name.c_str());
}

GLuint Program::getResourceIndex(GLenum interface, const std::string& name) const
{
	assert(isValid());
	return glGetProgramResourceIndex(getName(), interface, name.c_str());
}

GLuint Program::getUniformBlockIndex(const std::string& name) const
{
	assert(isValid());
	GLuint r = glGetUniformBlockIndex(getName(), name.c_str());
	if(r == GL_INVALID_INDEX) std::cerr << "Warning: Invalid Uniform Block name " << name << std::endl;
	assert(r != GL_INVALID_INDEX);
	return r;
}
	
void Program::bindUniformBlock(GLuint uniformBlockIndex, GLuint uniformBlockBindingPoint) const
{
	assert(isValid());
	assert(uniformBlockBindingPoint < GL_MAX_UNIFORM_BUFFER_BINDINGS);
	glUniformBlockBinding(getName(), uniformBlockIndex, uniformBlockBindingPoint);
}
	
void Program::bindUniformBlock(const std::string& name, GLuint uniformBlockBindingPoint) const
{
	assert(isValid());
	bindUniformBlock(getUniformBlockIndex(name), uniformBlockBindingPoint);
}
	
void Program::bindUniformBlock(const std::string& name, const UniformBuffer& uniformBuffer) const
{
	assert(isValid());
	assert(uniformBuffer.isBound());
	bindUniformBlock(getUniformBlockIndex(name), uniformBuffer.getBindingPoint());
}
	
void Program::useNone()
{
    glUseProgram(0);
}

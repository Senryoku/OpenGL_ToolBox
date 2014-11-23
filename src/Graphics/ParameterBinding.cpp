#include <ParameterBinding.hpp>

// Texture Parameter
	
void texParameter(GLenum target, GLenum pname, float param)
{
	glTexParameterf(target, pname, param);
}

void texParameter(GLenum target, GLenum pname, int param)
{
	glTexParameteri(target, pname, param);
}

void texParameter(GLenum target, GLenum pname, const float* param)
{
	glTexParameterfv(target, pname, param);
}

void texParameter(GLenum target, GLenum pname, const int* param)
{
	glTexParameteriv(target, pname, param);
}

void texParameterI(GLenum target, GLenum pname, const int* param)
{
	glTexParameterIiv(target, pname, param);
}

void texParameterI(GLenum target, GLenum pname, unsigned const int* param)
{
	glTexParameterIuiv(target, pname, param);
}

// Sampler Parameter

void samplerParameter(GLuint sampler, GLenum pname, float param)
{
	glSamplerParameterf(sampler, pname, param);
}

void samplerParameter(GLuint sampler, GLenum pname, int param)
{
	glSamplerParameteri(sampler, pname, param);
}

void samplerParameter(GLuint sampler, GLenum pname, const float* param)
{
	glSamplerParameterfv(sampler, pname, param);
}

void samplerParameter(GLuint sampler, GLenum pname, const int* param)
{
	glSamplerParameteriv(sampler, pname, param);
}

void samplerParameterI(GLuint sampler, GLenum pname, const int* param)
{
	glSamplerParameterIiv(sampler, pname, param);
}

void samplerParameterI(GLuint sampler, GLenum pname, const unsigned int* param)
{
	glSamplerParameterIuiv(sampler, pname, param);
}

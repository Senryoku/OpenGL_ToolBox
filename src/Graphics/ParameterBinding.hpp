#pragma once

#include <array>

#define GLEW_STATIC
#include <GL/glew.h>

// Texture Parameter
	
void texParameter(GLenum target, GLenum pname, float param);
void texParameter(GLenum target, GLenum pname, int param);
void texParameter(GLenum target, GLenum pname, const float* param);
void texParameter(GLenum target, GLenum pname, const int* param);

template<int I>
void texParameter(GLenum target, GLenum pname, std::array<float, I> param)
{
	glTexParameterfv(target, pname, &param);
}

template<int I>
void texParameter(GLenum target, GLenum pname, std::array<int, I> param)
{
	glTexParameteriv(target, pname, &param);
}

void texParameterI(GLenum target, GLenum pname, const int* param);
void texParameterI(GLenum target, GLenum pname, unsigned const int* param);

template<int I>
void texParameterI(GLenum target, GLenum pname, std::array<int, I> param)
{
	glTexParameterIiv(target, pname, &param);
}

template<int I>
void texParameterI(GLenum target, GLenum pname, std::array<const unsigned int, I> param)
{
	glTexParameterIuiv(target, pname, &param);
}

// Sampler Parameter

void samplerParameter(GLuint sampler, GLenum pname, float param);
void samplerParameter(GLuint sampler, GLenum pname, int param);
void samplerParameter(GLuint sampler, GLenum pname, const float* param);
void samplerParameter(GLuint sampler, GLenum pname, const int* param);

template<int I>
void samplerParameter(GLuint sampler, GLenum pname, std::array<float, I> param)
{
	glSamplerParameterfv(sampler, pname, &param);
}

template<int I>
void samplerParameter(GLuint sampler, GLenum pname, std::array<int, I> param)
{
	glSamplerParameteriv(sampler, pname, &param);
}

void samplerParameterI(GLuint sampler, GLenum pname, const int* param);
void samplerParameterI(GLuint sampler, GLenum pname, const unsigned int* param);

template<int I>
void samplerParameterI(GLuint sampler, GLenum pname, std::array<int, I> param)
{
	glSamplerParameterIiv(sampler, pname, &param);
}

template<int I>
void samplerParameterI(GLuint sampler, GLenum pname, std::array<const unsigned int, I> param)
{
	glSamplerParameterIuiv(sampler, pname, &param);
}

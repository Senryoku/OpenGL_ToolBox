#pragma once

#include <iostream>
#include <fstream>
#include <string>
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>

class FragmentShader : public Shader
{
	public:
	FragmentShader() : Shader() {}
	private:
	void init();
};


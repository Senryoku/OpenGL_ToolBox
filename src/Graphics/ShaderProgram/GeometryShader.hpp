#pragma once

#include <iostream>
#include <fstream>
#include <string>
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>


class GeometryShader : public Shader
{
	public:
	GeometryShader() : Shader() {}
	private:
	void initOGL();
};


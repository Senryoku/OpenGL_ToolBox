#pragma once

#include <iostream>
#include <fstream>
#include <string>
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>


class VertexShader : public Shader
{
public:
	VertexShader() : Shader() {}
	
private:
	virtual void init() override;
};


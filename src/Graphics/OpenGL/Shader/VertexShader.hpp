#pragma once

#include <Shader.hpp>

class VertexShader : public Shader
{
public:
	VertexShader() =default;
	
private:
	inline virtual GLenum getType() const override { return GL_VERTEX_SHADER; }
};


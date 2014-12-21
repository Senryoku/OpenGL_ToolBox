#pragma once

#include <Shader.hpp>


class GeometryShader : public Shader
{
public:
	GeometryShader() : Shader() {}

private:
	inline virtual GLenum getType() const override { return GL_GEOMETRY_SHADER; }
};


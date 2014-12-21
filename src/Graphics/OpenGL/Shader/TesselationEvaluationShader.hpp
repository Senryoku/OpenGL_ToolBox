#pragma once

#include <Shader.hpp>

class TesselationEvaluationShader : public Shader
{
public:
	TesselationEvaluationShader() =default;
	
private:	
	inline virtual GLenum getType() const override { return GL_TESS_EVALUATION_SHADER; }
};


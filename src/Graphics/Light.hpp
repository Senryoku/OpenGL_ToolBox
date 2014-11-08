#pragma once

#include <glm/glm.hpp>

#include <Frustum.hpp>
#include <Texture.hpp>
#include <ResourcesManager.hpp>
#include <AllShader.hpp>

class Light
{
public:
	void init();
	
protected:
	glm::vec3			_position = glm::vec3(0.f);
	glm::vec3			_direction = glm::vec3(0.f, 0.f, 1.f);
	float					_range;
	
	glm::vec4			_color = glm::vec4(1.f);
	
	Frustum				_frustum;

	bool					_castShadows = true;
	unsigned int		_shadowMapResolution = 4096;
	GLuint 				_shadowMapFramebuffer = 0;
	Texture				_shadowMapDepthTexture;
	glm::mat4			_depthProjectionMatrix;
	glm::mat4			_depthBiasMVP;
	
	static Program& 				_depthProgram;
    static VertexShader&		_depthVS;
    static FragmentShader&	_depthFS;
};

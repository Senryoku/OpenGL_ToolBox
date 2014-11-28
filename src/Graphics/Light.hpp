#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <Frustum.hpp>
#include <Texture2D.hpp>
#include <Framebuffer.hpp>
#include <ResourcesManager.hpp>
#include <AllShader.hpp>

class Light
{
public:
	Light();
	~Light() =default;

	void init();
	
	inline const glm::vec3& getPosition() const { return _position; }
	inline const glm::vec3& getDirection() const { return _direction; }
	inline float getRange() const { return _range; }
	
	inline void setPosition(const glm::vec3& pos) { _position = pos; }
	inline void setDirection(const glm::vec3& dir) { _direction = dir; }
	inline void lookAt(const glm::vec3& at) { _direction = glm::normalize(at - _position); }
	inline void setRange(float r) { _range = r; }
	
	inline const glm::mat4& getMatrix() const { return _VPMatrix; }
	inline const glm::mat4& getBiasedMatrix() const { return _biasedVPMatrix; }
	inline const Framebuffer<Texture2D, 0>& getShadowBuffer() const { return _shadowMapFramebuffer; }
	inline const Texture2D& getShadowMap() const { return _shadowMapFramebuffer.getDepth(); }
	
	void updateMatrices();
	
	inline static const glm::mat4& getBiasMatrix() { return s_depthBiasMVP; }
	inline static const Program& getShadowMapProgram() { return *s_depthProgram; }
protected:
	glm::vec3			_position = glm::vec3(0.f);
	glm::vec3			_direction = glm::vec3(0.f, 0.f, 1.f);
	float					_range = 1000.0;
	
	glm::vec4			_color = glm::vec4(1.f);
	
	Frustum				_frustum;

	unsigned int						_shadowMapResolution = 4096;
	Framebuffer<Texture2D, 0>	_shadowMapFramebuffer;
	glm::mat4							_VPMatrix;
	glm::mat4							_biasedVPMatrix;
	
	// Static Attributes
	
	static const glm::mat4	s_depthBiasMVP;
	
	static Program* 				s_depthProgram;
    static VertexShader*		s_depthVS;
    static FragmentShader*	s_depthFS;
};

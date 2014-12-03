#pragma once

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <Frustum.hpp>
#include <Texture2D.hpp>
#include <Framebuffer.hpp>
#include <ResourcesManager.hpp>
#include <AllShader.hpp>

/**
 * Light
 *
 * Describes a spotlight which can cast shadows (Shadow Mapping).
 * @todo Internalize shadow map drawing
 * @todo Re-use frustum
**/
class Light
{
public:
	/**
	 * Constructor
	 *
	 * @param shadowMapResolution Resolution of the shadow map depth texture.
	**/
	Light(unsigned int shadowMapResolution = 4096);
	~Light() =default;

	/**
	 * Initialize the shadow mapping attributes (Shaders, Framebuffer...)
	 * for this light.
	**/
	void init();
	
	inline const glm::vec4& getColor() const { return _color; }
	inline const glm::vec3& getPosition() const { return _position; }
	inline const glm::vec3& getDirection() const { return _direction; }
	inline float getRange() const { return _range; }
	
	inline void setColor(const glm::vec4& col) { _color = col; }
	inline void setPosition(const glm::vec3& pos) { _position = pos; }
	inline void setDirection(const glm::vec3& dir) { _direction = dir; }
	inline void lookAt(const glm::vec3& at) { _direction = glm::normalize(at - _position); }
	inline void setRange(float r) { _range = r; }
	
	/**
	 * Returns the transformation matrix (Projection * View) for this Light. 
	 *
	 * @return World to Light's view space matrix.
	 * @see getBiasedMatrix()
	**/
	inline const glm::mat4& getMatrix() const { return _VPMatrix; }
	
	/**
	 * Returns the biased transformation matrix (Projection * View) for this Light.
	 * (Biased meaning "in [0, 1] range", i.e. texture friendly :])
	 *
	 * @return biased World to Light's view space matrix.
	 * @see getMatrix()
	**/
	inline const glm::mat4& getBiasedMatrix() const { return _biasedVPMatrix; }
	
	/**
	 * @return Light's shadow map framme buffer.
	**/
	inline const Framebuffer<Texture2D, 0>& getShadowBuffer() const { return _shadowMapFramebuffer; }
	
	/**
	 * @return Light's shadow map depth texture.
	**/
	inline const Texture2D& getShadowMap() const { return _shadowMapFramebuffer.getDepth(); }
	
	/**
	 * Updates Light's internal transformation matrices according to
	 * its current position/direction/range.
	**/
	void updateMatrices();
	
	/**
	 * Setup the context to draw to this light' shadow map.
	 * @todo Find a better name...
	**/
	void bind() const;
	
	/**
	 * Restores the default framebuffer.
	 * @todo Find a better name...
	**/
	void unbind() const;
	
	inline static const glm::mat4& getBiasMatrix() { return s_depthBiasMVP; }
	inline static const Program& getShadowMapProgram() { return *s_depthProgram; }
protected:
	glm::vec4			_color = glm::vec4(1.f);
	
	glm::vec3			_position = glm::vec3(0.f); ///< Light's position in World Space
	glm::vec3			_direction = glm::vec3(0.f, 0.f, 1.f); ///< Light's direction in World Space
	float				_range = 1000.0; ///< Light's range, mainly used for the Shadow Mapping settings

	unsigned int				_shadowMapResolution = 4096;
	Framebuffer<Texture2D, 0>	_shadowMapFramebuffer;
	glm::mat4					_VPMatrix;
	glm::mat4					_biasedVPMatrix;
	
	// Static Attributes
	
	static const glm::mat4	s_depthBiasMVP;
	
	static Program* 		s_depthProgram;
    static VertexShader*	s_depthVS;
    static FragmentShader*	s_depthFS;
};

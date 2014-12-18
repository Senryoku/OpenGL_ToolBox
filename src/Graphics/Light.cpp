#include <Light.hpp>

#include <glm/gtc/matrix_transform.hpp> // glm::lookAt, glm::perspective
#include <glm/gtx/transform.hpp> // glm::translate

#include <MathTools.hpp>
#include <ResourcesManager.hpp>

///////////////////////////////////////////////////////////////////
// Static attributes

const glm::mat4 Light::s_depthBiasMVP
(
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 0.5, 0.0,
	0.5, 0.5, 0.5, 1.0
);

Program* 			Light::s_depthProgram = nullptr;
VertexShader*		Light::s_depthVS = nullptr;
FragmentShader*		Light::s_depthFS = nullptr;

///////////////////////////////////////////////////////////////////

Light::Light(unsigned int shadowMapResolution) :
	_shadowMapResolution(shadowMapResolution),
	_shadowMapFramebuffer(_shadowMapResolution, true),
	_projection(glm::perspective(static_cast<float>(pi())/4.f, 1.0f, 2.0f, _range))
{
}
		
void Light::init()
{
	if(s_depthProgram == nullptr)
	{
		s_depthProgram = &ResourcesManager::getInstance().getProgram("Light_Depth");
		s_depthVS = &ResourcesManager::getInstance().getShader<VertexShader>("Light_DepthVS");
		s_depthFS = &ResourcesManager::getInstance().getShader<FragmentShader>("Light_DepthFS");
	}
	
	if(s_depthProgram != nullptr && !s_depthProgram->isValid())
	{
		s_depthVS->loadFromFile("src/GLSL/depth_vs.glsl");
		s_depthVS->compile();
		s_depthFS->loadFromFile("src/GLSL/depth_fs.glsl");
		s_depthFS->compile();
		s_depthProgram->attachShader(*s_depthVS);
		s_depthProgram->attachShader(*s_depthFS);
		s_depthProgram->link();
	}
	
	_shadowMapFramebuffer.init();
}

void Light::updateMatrices()
{
	_view = glm::lookAt(_position, _position + _direction, glm::vec3(0,1,0));
	_VPMatrix = _projection * _view;
	_biasedVPMatrix = s_depthBiasMVP * _VPMatrix;
}

void Light::bind() const
{
	getShadowBuffer().bind();
	getShadowMapProgram().setUniform("DepthVP", getMatrix());
	getShadowMapProgram().use();
	glCullFace(GL_FRONT);
}

void Light::unbind() const
{
	glCullFace(GL_BACK);
	Program::useNone();
	getShadowBuffer().unbind();
}

#include <Light.hpp>

void Light::init()
{
	Program& 		_depthProgram = ResourcesManager::getInstance().getProgram("Depth");
    VertexShader& 	_depthVS = ResourcesManager::getInstance().getShader<VertexShader>("DepthVS");
    FragmentShader&	_depthFS = ResourcesManager::getInstance().getShader<FragmentShader>("DepthFS");
	
	if(!_depthProgram.isValid())
	{
		_depthVS.loadFromFile("src/GLSL/depth_vs.glsl");
		_depthVS.compile();
		_depthFS.loadFromFile("src/GLSL/depth_fs.glsl");
		_depthFS.compile();
		_depthProgram.attachShader(_depthVS);
		_depthProgram.attachShader(_depthFS);
		_depthProgram.link();
	}
	
	GLuint Tex;
	glGenTextures(1, &Tex);
	_shadowMapDepthTexture.setName(Tex);

	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, _shadowMapResolution, _shadowMapResolution, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	glGenFramebuffers(1, &_shadowMapFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _shadowMapFramebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, Tex, 0);

	 // CleanUp Context
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cerr << "Error initializing Shadow Mapping." << std::endl;
		glDeleteFramebuffers(1, &_shadowMapFramebuffer);
		return;
	}
}

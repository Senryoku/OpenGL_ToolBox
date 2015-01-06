#include <cstdlib>
#include <ctime>
#include <sstream>
#include <map>
#include <random>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp> // glm::lookAt, glm::perspective
#include <glm/gtx/transform.hpp> // glm::translate
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include <TimeManager.hpp>
#include <ResourcesManager.hpp>
#include <StringConversion.hpp>
#include <Material.hpp>
#include <Texture2D.hpp>
#include <Framebuffer.hpp>
#include <Buffer.hpp>
#include <TransformFeedback.hpp>
#include <MathTools.hpp>
#include <Camera.hpp>
#include <Skybox.hpp>
#include <Light.hpp>
#include <stb_image_write.hpp>

int			_width = 1366;
int			_height = 720;

float		_fov = 60.0;
glm::vec3 	_resolution(_width, _height, 0.0);
glm::mat4 	_projection;
glm::vec4 	_mouse(0.0);

glm::vec4 	_ambiant = glm::vec4(0.05f, 0.05f, 0.05f, 1.f);

bool 		_updateVFC = true;

int 		_poissonSamples = 4;
float 		_poissonDiskRadius = 2500.f;

bool 		_fullscreen = false;
bool 		_msaa = false;

bool 		_controlCamera = true;
double 		_mouse_x, 
			_mouse_y;

float		_timescale = 1.0;
float 		_time = 0.f;
float		_frameTime;
float		_frameRate;
bool		_paused = false;

int			_colorToRender = 3;

Framebuffer<Texture2D, 3>	_offscreenRender;
	
void screen(const std::string& path)
{
	GLubyte* pixels = new GLubyte[4 * _width * _height];

	glReadPixels(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	
	stbi_write_png(path.c_str(), _width, _height, 4, pixels, 0);
	
	delete[] pixels;
}

void error_callback(int error, const char* description)
{
	std::cerr << "GLFW Error (" << error << "): " << description << std::endl;
}

void resize_callback(GLFWwindow* window, int width, int height)
{
	_width = width;
	_height = height;

	glViewport(0, 0, _width, _height);
	_resolution = glm::vec3(_width, _height, 0.0);
	
	float inRad = _fov * pi()/180.f;
	_projection = glm::perspective(inRad, (float) _width/_height, 0.1f, 1000.0f);
	
	_offscreenRender = Framebuffer<Texture2D, 3>(_width, _height, true);
	// Special format for world positions and normals.
	_offscreenRender.getColor(0).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.getColor(1).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.getColor(2).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.init();
	
	std::cout << "Reshaped to " << width << "*" << height  << " (" << ((GLfloat) _width)/_height << ")" << std::endl;
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if(action == GLFW_PRESS)
	{
		switch(key)
		{
			case GLFW_KEY_ESCAPE:
			{
				glfwSetWindowShouldClose(window, GL_TRUE);
				break;
			}
			case GLFW_KEY_R:
			{
				std::cout << "Reloading shaders..." << std::endl;
				ResourcesManager::getInstance().reloadShaders();
				std::cout << "Reloading shaders... Done !" << std::endl;
				break;
			}
			case GLFW_KEY_SPACE:
			{
				if(!_controlCamera)
				{
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
				} else {
					glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
				}
				_controlCamera = !_controlCamera;
				break;
			}
			case GLFW_KEY_X:
			{
				_msaa = ! _msaa;
				if(_msaa)
				{
					glEnable(GL_MULTISAMPLE);
					glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);
					
					GLint  iMultiSample = 0;
					GLint  iNumSamples = 0;
					glGetIntegerv(GL_SAMPLE_BUFFERS, &iMultiSample);
					glGetIntegerv(GL_SAMPLES, &iNumSamples);
					
					std::cout << "Enabled MSAA (GL_SAMPLES : " << iNumSamples << ", GL_SAMPLE_BUFFERS : " << iMultiSample << ")" << std::endl;
				} else {
					glDisable(GL_MULTISAMPLE);
					
					GLint  iMultiSample = 0;
					GLint  iNumSamples = 0;
					glGetIntegerv(GL_SAMPLE_BUFFERS, &iMultiSample);
					glGetIntegerv(GL_SAMPLES, &iNumSamples);
					std::cout << "Disabled MSAA (GL_SAMPLES : " << iNumSamples << ", GL_SAMPLE_BUFFERS : " << iMultiSample << ")" << std::endl;
				}
				break;
			}
			case GLFW_KEY_V:
			{
				_fullscreen = !_fullscreen;
				if(_fullscreen)
				{
					std::cout << "TODO: Add fullscreen :p (Sorry...)" << std::endl;
				} else {
					std::cout << "TODO: Add fullscreen :p (Sorry...)" << std::endl;
				}
				break;
			}
			case GLFW_KEY_P:
			{
				_paused = !_paused;
				break;
			}
			case GLFW_KEY_C:
			{
				_colorToRender = (_colorToRender + 1) % 7;
				std::cout << "Render setting: " << _colorToRender << std::endl;
				break;
			}
			case GLFW_KEY_L:
			{
				const std::string ScreenPath("out/screenshot.png");
				std::cout << "Saving a screenshot to " << ScreenPath << "..." << std::endl;
				screen(ScreenPath);
			}
		}
	}
}

inline void EventMouseButtonGLFW3(GLFWwindow* window, int button, int action, int mods)
{	
	float z = _mouse.z;
	float w = _mouse.w;
	if(button == GLFW_MOUSE_BUTTON_1)
	{
		if(action == GLFW_PRESS)
		{
			z = 1.0;
		} else {
			z = 0.0;
		}
	} else if(button == GLFW_MOUSE_BUTTON_2) {
		if(action == GLFW_PRESS)
		{
			w = 1.0;
		} else {
			w = 0.0;
		}
	}
	
	_mouse = glm::vec4(_mouse.x, _mouse.y, z, w);
}

inline void EventMousePosGLFW3(GLFWwindow* window, double xpos, double ypos)
{
	_mouse = glm::vec4(xpos, ypos, _mouse.z, _mouse.w);
}

// Temp
struct LightStruct
{
	glm::vec4	position;
	glm::vec4	color;
};

struct CameraStruct
{
	glm::mat4	view;
	glm::mat4	projection;
};

struct Particle
{
	glm::vec4	position_type;
	glm::vec4	speed_lifetime;
};

int main(int argc, char* argv[])
{
	if (glfwInit() == false)
	{
		std::cerr << "Error: couldn't initialize GLFW." << std::endl;
		exit(EXIT_FAILURE);
	}
	glfwSetErrorCallback(error_callback);
	glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow(_width, _height, "OpenGL ToolBox Test", nullptr, nullptr);
	
	if (!window)
	{
		std::cerr << "Error: couldn't create window." << std::endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	
	if(glewInit() != GLEW_OK)
	{
		std::cerr << "Error: couldn't initialize GLEW." << std::endl;
		exit(EXIT_FAILURE);
	}
	
	// Callback Setting
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, (GLFWmousebuttonfun) EventMouseButtonGLFW3);
	glfwSetCursorPosCallback(window, (GLFWcursorposfun) EventMousePosGLFW3);
	
	glfwSetWindowSizeCallback(window, resize_callback);
	
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			
	float LightRadius = 500.0;
	
	glEnable(GL_DEPTH_TEST);
	
	VertexShader& DeferredVS = ResourcesManager::getInstance().getShader<VertexShader>("Deferred_VS");
	//DeferredVS.loadFromFile("src/GLSL/Deferred/deferred_vs.glsl");
	DeferredVS.loadFromFile("src/GLSL/Deferred/deferred_instance_vs.glsl");
	DeferredVS.compile();

	FragmentShader& DeferredFS = ResourcesManager::getInstance().getShader<FragmentShader>("Deferred_FS");
	DeferredFS.loadFromFile("src/GLSL/Deferred/deferred_normal_map_fs.glsl");
	DeferredFS.compile();
	
	Program& Deferred = ResourcesManager::getInstance().getProgram("Deferred");
	Deferred.attachShader(DeferredVS);
	Deferred.attachShader(DeferredFS);
	Deferred.link();
	
	if(!Deferred) return 0;
	
	VertexShader& DeferredLightVS = ResourcesManager::getInstance().getShader<VertexShader>("DeferredLight_VS");
	DeferredLightVS.loadFromFile("src/GLSL/Deferred/deferred_light_vs.glsl");
	DeferredLightVS.compile();

	FragmentShader& DeferredLightFS = ResourcesManager::getInstance().getShader<FragmentShader>("DeferredLight_FS");
	DeferredLightFS.loadFromFile("src/GLSL/Deferred/deferred_light_fs.glsl");
	DeferredLightFS.compile();
	
	Program& DeferredLight = ResourcesManager::getInstance().getProgram("DeferredLight");
	DeferredLight.attachShader(DeferredLightVS);
	DeferredLight.attachShader(DeferredLightFS);
	DeferredLight.link();
	
	if(!DeferredLight) return 0;
	
	VertexShader& DeferredColorVS = ResourcesManager::getInstance().getShader<VertexShader>("DeferredColor_VS");
	DeferredColorVS.loadFromFile("src/GLSL/Deferred/deferred_forward_color_vs.glsl");
	DeferredColorVS.compile();

	FragmentShader& DeferredColorFS = ResourcesManager::getInstance().getShader<FragmentShader>("DeferredColor_FS");
	DeferredColorFS.loadFromFile("src/GLSL/Deferred/deferred_forward_color_fs.glsl");
	DeferredColorFS.compile();
	
	Program& DeferredColor = ResourcesManager::getInstance().getProgram("DeferredColor");
	DeferredColor.attachShader(DeferredColorVS);
	DeferredColor.attachShader(DeferredColorFS);
	DeferredColor.link();
	
	if(!DeferredColor) return 0;
	
	VertexShader& PostProcessVS = ResourcesManager::getInstance().getShader<VertexShader>("PostProcess_VS");
	PostProcessVS.loadFromFile("src/GLSL/vs.glsl");
	PostProcessVS.compile();

	FragmentShader& PostProcessFS = ResourcesManager::getInstance().getShader<FragmentShader>("PostProcess_FS");
	PostProcessFS.loadFromFile("src/GLSL/Deferred/phong_deferred_fs.glsl");
	PostProcessFS.compile();
	
	Program& PostProcess = ResourcesManager::getInstance().getProgram("PostProcess");
	PostProcess.attachShader(PostProcessVS);
	PostProcess.attachShader(PostProcessFS);
	PostProcess.link();
	
	if(!PostProcess) return 0;
	
	Material PostProcessMaterial(PostProcess);
	//PostProcessMaterial.setUniform("iResolution", &_resolution);
	PostProcessMaterial.setUniform("Color", _offscreenRender.getColor(0));
	PostProcessMaterial.setUniform("Position", _offscreenRender.getColor(1));
	PostProcessMaterial.setUniform("Normal", _offscreenRender.getColor(2));
	PostProcessMaterial.setUniform("lightRadius", &LightRadius);
	
	ComputeShader& DeferredCS = ResourcesManager::getInstance().getShader<ComputeShader>("DeferredCS");
	DeferredCS.loadFromFile("src/GLSL/Deferred/tiled_deferred_cs.glsl");
	DeferredCS.compile();
	
	if(!DeferredCS.getProgram()) return 0;
	
	Program& ParticleUpdate = ResourcesManager::getInstance().getProgram("ParticleUpdate");
	VertexShader& ParticleUpdateVS = ResourcesManager::getInstance().getShader<VertexShader>("ParticleUpdate_VS");
	ParticleUpdateVS.loadFromFile("src/GLSL/Particles/update_vs.glsl");
	ParticleUpdateVS.compile();
	GeometryShader& ParticleUpdateGS = ResourcesManager::getInstance().getShader<GeometryShader>("ParticleUpdate_GS");
	ParticleUpdateGS.loadFromFile("src/GLSL/Particles/update_gs.glsl");
	ParticleUpdateGS.compile();
	ParticleUpdate.attachShader(ParticleUpdateVS);
	ParticleUpdate.attachShader(ParticleUpdateGS);
	const char* varyings[2] = {"position_type", "speed_lifetime"};
	glTransformFeedbackVaryings(ParticleUpdate.getName(), 2, varyings, GL_INTERLEAVED_ATTRIBS);
	ParticleUpdate.link();
	
	if(!ParticleUpdate) return 0;
	
	Program& ParticleDraw = ResourcesManager::getInstance().getProgram("ParticleDraw");
	VertexShader& ParticleDrawVS = ResourcesManager::getInstance().getShader<VertexShader>("ParticleDraw_VS");
	ParticleDrawVS.loadFromFile("src/GLSL/Particles/draw_vs.glsl");
	ParticleDrawVS.compile();
	GeometryShader& ParticleDrawGS = ResourcesManager::getInstance().getShader<GeometryShader>("ParticleDraw_GS");
	ParticleDrawGS.loadFromFile("src/GLSL/Particles/draw_gs.glsl");
	ParticleDrawGS.compile();
	FragmentShader& ParticleDrawFS = ResourcesManager::getInstance().getShader<FragmentShader>("ParticleDraw_FS");
	ParticleDrawFS.loadFromFile("src/GLSL/Particles/draw_fs.glsl");
	ParticleDrawFS.compile();
	ParticleDraw.attachShader(ParticleDrawVS);
	ParticleDraw.attachShader(ParticleDrawGS);
	ParticleDraw.attachShader(ParticleDrawFS);
	ParticleDraw.link();
	 
	if(!ParticleDraw) return 0;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Camera Initialization
	
	Camera MainCamera;
	UniformBuffer CameraBuffer;
	CameraBuffer.init();
	CameraBuffer.bind(0);
	Deferred.bindUniformBlock("Camera", CameraBuffer); 
	DeferredLight.bindUniformBlock("Camera", CameraBuffer);
	DeferredColor.bindUniformBlock("Camera", CameraBuffer);
	ParticleDraw.bindUniformBlock("Camera", CameraBuffer);
	//DeferredShadowCS.getProgram().bindUniformBlock("Camera", CameraBuffer);
	
	PostProcessMaterial.setUniform("cameraPosition", &MainCamera.getPosition());
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Light initialization
	
	const size_t LightCount = 1;
	PostProcessMaterial.setUniform("lightCount", LightCount);
	DeferredCS.getProgram().setUniform("lightCount", LightCount);
	
	UniformBuffer LightBuffer;
	
	LightBuffer.init();
	LightBuffer.bind(1);

	PostProcess.bindUniformBlock("LightBlock", LightBuffer);
	DeferredCS.getProgram().bindUniformBlock("LightBlock", LightBuffer);
	
	LightStruct tmpLight[LightCount];
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Particles
	
	VertexArray particles_vao;
	particles_vao.init();
	particles_vao.bind();
	
	Buffer particles_buffers[2];
	particles_buffers[0].init(Buffer::VertexAttributes);
	particles_buffers[1].init(Buffer::VertexAttributes);
	std::vector<Particle> particles;
	
	for(int i = 0; i < 100; ++i)
		particles.push_back(Particle{glm::vec4{0.0, 0.0, i, 0.0}, glm::vec4{0.0, 1.0, 0.0, 10.0}});
	
	TransformFeedback particles_transform_feedback[2];
	for(int i = 0; i < 2; ++i)
	{
		particles_transform_feedback[i].init();
		particles_transform_feedback[i].bind();
		particles_buffers[i].bind();
		particles_buffers[i].data(particles.data(), sizeof(Particle) * particles.size(), Buffer::DynamicDraw);
		particles_transform_feedback[i].bindBuffer(0, particles_buffers[i]);
	}
	
	size_t ParticleStep = 0;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Light Sphere Mesh
	auto LightSphereV = Mesh::load("in/3DModels/sphere/sphere.obj");
	auto& LightSphere = LightSphereV[0];
	LightSphere->createVAO();
	
	resize_callback(window, _width, _height);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Main Loop
	
	MainCamera.updateView();
	bool firstStep = true;
	
	while(!glfwWindowShouldClose(window))
	{	
		// Time Management 
		TimeManager::getInstance().update();
		_frameTime = TimeManager::getInstance().getRealDeltaTime();
		_frameRate = TimeManager::getInstance().getInstantFrameRate();
		if(!_paused)
			_time += _timescale * _frameTime;
		
		// Camera Management
		if(_controlCamera)
		{
			if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				MainCamera.moveForward(_frameTime);
				
			if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
				MainCamera.strafeLeft(_frameTime);
					
			if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				MainCamera.moveBackward(_frameTime);
					
			if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
				MainCamera.strafeRight(_frameTime);
					
			if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
				MainCamera.moveDown(_frameTime);
					
			if(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
				MainCamera.moveUp(_frameTime);
				
			double mx = _mouse_x, my = _mouse_y;
			glfwGetCursorPos(window, &_mouse_x, &_mouse_y);
			MainCamera.look(glm::vec2(_mouse_x - mx, my - _mouse_y));
		}
		MainCamera.updateView();
		// Uploading camera data to the corresponding camera buffer
		CameraStruct CamS = {MainCamera.getMatrix(), _projection};
		CameraBuffer.data(&CamS, sizeof(CameraStruct), Buffer::DynamicDraw);
		
		// (Updating window title)
		std::ostringstream oss;
		oss << _frameRate;
		glfwSetWindowTitle(window, ((std::string("OpenGL ToolBox Test - FPS: ") + oss.str()).c_str()));
	
		////////////////////////////////////////////////////////////////////////////////////////////
		// Light Management
		
		for(size_t i = 0; i < LightCount; ++i)
		{
			tmpLight[i] = {
				glm::vec4(0.0, 10.0, 0.0, 1.0), 	// Position
				glm::vec4(1.0, 1.0, 1.0, 1.0)		// Color
			};
		}
		LightBuffer.data(&tmpLight, LightCount * sizeof(LightStruct), Buffer::DynamicDraw);
		////////////////////////////////////////////////////////////////////////////////////////////		
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Particle Update
		
		ParticleUpdate.setUniform("time", _frameTime);
		ParticleUpdate.use();
		
		TransformFeedback::disableRasterization();
		particles_buffers[ParticleStep].bind();
		particles_transform_feedback[(ParticleStep + 1) % 2].bind();
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), 0); // position_type
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*) sizeof(glm::vec4)); // speed_lifetime
	 
		TransformFeedback::begin(Points);
		
		if(firstStep)
		{
			glDrawArrays(GL_POINTS, 0, particles.size());
			firstStep = false;
		} else {
			particles_transform_feedback[ParticleStep].draw(Points);
		}
		
		TransformFeedback::end();
		TransformFeedback::enableRasterization();
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Actual drawing
		
		// Offscreen
		_offscreenRender.bind();
		_offscreenRender.clear();
		
		ParticleDraw.use();
		
		particles_buffers[(ParticleStep + 1) % 2].bind();
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), 0); // position_type
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*) sizeof(glm::vec4)); // speed_lifetime
		
		glPointSize(8.0);
		particles_transform_feedback[(ParticleStep + 1) % 2].draw(Points);
		
		_offscreenRender.unbind();		
		
		ParticleStep = (ParticleStep + 1) % 2;
		
		// Post processing
		// Restore Viewport (binding the framebuffer modifies it - should I make the unbind call restore it ? How ?)
		glViewport(0, 0, _width, _height);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		/*
		_offscreenRender.getColor(0).bindImage(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
		_offscreenRender.getColor(1).bindImage(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		_offscreenRender.getColor(2).bindImage(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		DeferredCS.getProgram().setUniform("ColorDepth", (int) 0);
		DeferredCS.getProgram().setUniform("Position", (int) 1);
		DeferredCS.getProgram().setUniform("Normal", (int) 2);	
		DeferredCS.getProgram().setUniform("cameraPosition", MainCamera.getPosition());
		DeferredCS.getProgram().setUniform("lightRadius", LightRadius);
		DeferredCS.compute(_resolution.x / DeferredCS.getWorkgroupSize().x + 1, _resolution.y / DeferredCS.getWorkgroupSize().y + 1, 1);
		DeferredCS.memoryBarrier();
		
		// Blitting
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		_offscreenRender.bind(FramebufferTarget::Read);
		glBlitFramebuffer(0, 0, _resolution.x, _resolution.y, 0, 0, _resolution.x, _resolution.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		*/
		
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		_offscreenRender.bind(FramebufferTarget::Read);
		glReadBuffer(GL_COLOR_ATTACHMENT0 + (0));
		glBlitFramebuffer(0, 0, _resolution.x, _resolution.y, 0, 0, _resolution.x, _resolution.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		
		////////////////////////////////////////////////////////////////////////////////////////////
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	glfwDestroyWindow(window);
}

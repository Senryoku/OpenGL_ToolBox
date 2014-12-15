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
#include <AntTweakBar.h>

#include <TimeManager.hpp>
#include <ResourcesManager.hpp>
#include <StringConversion.hpp>
#include <Material.hpp>
#include <Texture2D.hpp>
#include <Texture3D.hpp>
#include <Framebuffer.hpp>
#include <Buffer.hpp>
#include <MeshInstance.hpp>
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

float 		_ballsDiffuseReflection = 0.4f;

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

int			_colorToRender = 0;

Framebuffer<Texture2D, 3>	_offscreenRender;
	
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
	_offscreenRender.getColor(1).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.getColor(2).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.init();
	
	TwWindowSize(_width, _height);
	std::cout << "Reshaped to " << width << "*" << height  << " (" << ((GLfloat) _width)/_height << ")" << std::endl;
}

// Hackish way to add basic support of GLFW3 to AntTweakBar
// There may have some problems: http://sourceforge.net/p/anttweakbar/tickets/11/

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// Quick GLFW3 compatibility hack...
	int twkey = key;
	switch(twkey)
	{
		case GLFW_KEY_LEFT: twkey = TW_KEY_LEFT; break;
		case GLFW_KEY_RIGHT: twkey = TW_KEY_RIGHT; break;
		case GLFW_KEY_UP: twkey = TW_KEY_UP; break;
		case GLFW_KEY_DOWN: twkey = TW_KEY_DOWN; break;
		case GLFW_KEY_ESCAPE: twkey = 256 + 1; break;
		case GLFW_KEY_ENTER: twkey = 256 + 38; break;
		case GLFW_KEY_KP_ENTER: twkey = 62; break;
		case GLFW_KEY_BACKSPACE: twkey = 256 + 39; break;
		case GLFW_KEY_SPACE: twkey = 32; break;
		default: break;
	}
	
	if(!TwEventKeyGLFW(twkey, action))
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
					_colorToRender = (_colorToRender + 1) % 5;
					break;
				}
			}
		}
	}
}

inline void TwEventMouseButtonGLFW3(GLFWwindow* window, int button, int action, int mods)
{	
	if(!TwEventMouseButtonGLFW(button, action))
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
}

inline void TwEventMousePosGLFW3(GLFWwindow* window, double xpos, double ypos)
{
	if(!TwMouseMotion(int(xpos), int(ypos)))
	{
		_mouse = glm::vec4(xpos, ypos, _mouse.z, _mouse.w);
	}
}

inline void TwEventMouseWheelGLFW3(GLFWwindow* window, double xoffset, double yoffset) { TwEventMouseWheelGLFW(yoffset); }

inline void TwEventKeyGLFW3(GLFWwindow* window, int key, int scancode, int action, int mods) { TwEventKeyGLFW(key, action); }

inline void TwEventCharGLFW3(GLFWwindow* window, int codepoint) { TwEventCharGLFW(codepoint, GLFW_PRESS); }

void screen(const std::string& path)
{
	GLubyte* pixels = new GLubyte[4 * _width * _height];

	glReadPixels(0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	
	stbi_write_png(path.c_str(), _width, _height, 4, pixels, 0);
	
	delete[] pixels;
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

// Checks whether the provided bounding bound is visible after its transformation by MVPMatrix
// Todo: Place in... MeshInstance ?
bool isVisible(const glm::mat4& MVPMatrix, const BoundingBox& bbox)
{
	const glm::vec3& a = bbox.min;
	const glm::vec3& b = bbox.max;
	
	std::array<glm::vec4, 8> p = {MVPMatrix * glm::vec4{a.x, a.y, a.z, 1.0},
								  MVPMatrix * glm::vec4{a.x, a.y, b.z, 1.0},
								  MVPMatrix * glm::vec4{a.x, b.y, a.z, 1.0},
								  MVPMatrix * glm::vec4{a.x, b.y, b.z, 1.0},
								  MVPMatrix * glm::vec4{b.x, a.y, a.z, 1.0},
								  MVPMatrix * glm::vec4{b.x, a.y, b.z, 1.0},
								  MVPMatrix * glm::vec4{b.x, b.y, a.z, 1.0},
								  MVPMatrix * glm::vec4{b.x, b.y, b.z, 1.0}};

	glm::vec2 min = glm::vec2(0.0), max = glm::vec2(0.0);
								  
	for(auto& t : p)
	{
		t /= t.w;
		min.x = std::min(min.x, t.x);
		min.y = std::min(min.y, t.y);
		max.x = std::max(max.x, t.x);
		max.y = std::max(max.y, t.y);
	}
	
	return !(max.x < -1.0 || max.y < -1.0 ||
			 min.x > 1.0  || min.y > 1.0);
}

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
	glfwSetMouseButtonCallback(window, (GLFWmousebuttonfun) TwEventMouseButtonGLFW3);
	glfwSetCursorPosCallback(window, (GLFWcursorposfun) TwEventMousePosGLFW3);
	glfwSetScrollCallback(window, (GLFWscrollfun) TwEventMouseWheelGLFW3);
	glfwSetCharCallback(window, (GLFWcharfun)TwEventCharGLFW3);
	
	glfwSetWindowSizeCallback(window, resize_callback);
	
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	
	TwInit(TW_OPENGL, nullptr);
	TwWindowSize(_width, _height);
			
	TwBar* bar = TwNewBar("Global Tweaks");
	TwDefine("'Global Tweaks' color='0 0 0' ");
	TwDefine("'Global Tweaks' iconified=true ");
	TwAddVarRO(bar, "FrameTime", TW_TYPE_FLOAT, &_frameTime, "");
	TwAddVarRO(bar, "FrameRate", TW_TYPE_FLOAT, &_frameRate, "");
	TwAddVarRW(bar, "TimeScale", TW_TYPE_FLOAT, &_timescale, "min=0.0 step=0.1");
	TwAddVarRW(bar, "FOV", TW_TYPE_FLOAT, &_fov, "min=0.0 step=0.1");
	TwAddVarRO(bar, "Fullscreen (V to toogle)", TW_TYPE_BOOLCPP, &_fullscreen, "");
	TwAddVarRO(bar, "MSAA (X to toogle)", TW_TYPE_BOOLCPP, &_msaa, "");
	TwAddVarRW(bar, "Ball Diffuse Reflection", TW_TYPE_FLOAT, &_ballsDiffuseReflection, "min=0 max=1 step=0.05");
	
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	VertexShader& DeferredVS = ResourcesManager::getInstance().getShader<VertexShader>("Deferred_VS");
	DeferredVS.loadFromFile("src/GLSL/Deferred/deferred_vs.glsl");
	DeferredVS.compile();

	FragmentShader& DeferredFS = ResourcesManager::getInstance().getShader<FragmentShader>("Deferred_FS");
	DeferredFS.loadFromFile("src/GLSL/Deferred/deferred_fs.glsl");
	DeferredFS.compile();
	
	Program& Deferred = ResourcesManager::getInstance().getProgram("Deferred");
	Deferred.attachShader(DeferredVS);
	Deferred.attachShader(DeferredFS);
	Deferred.link();
	
	if(!Deferred) return 0;
	
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
	PostProcessMaterial.setUniform("iResolution", &_resolution);
	PostProcessMaterial.setUniform("Color", _offscreenRender.getColor(0));
	PostProcessMaterial.setUniform("Position", _offscreenRender.getColor(1));
	PostProcessMaterial.setUniform("Normal", _offscreenRender.getColor(2));
	
	FragmentShader& FullScreenTextureFS = ResourcesManager::getInstance().getShader<FragmentShader>("FullScreenTextureFS");
	FullScreenTextureFS.loadFromFile("src/GLSL/DisplayTexture.glsl");
	FullScreenTextureFS.compile();
	
	Program& FullScreenTexture = ResourcesManager::getInstance().getProgram("FullScreenTexture");
	FullScreenTexture.attachShader(PostProcessVS);
	FullScreenTexture.attachShader(FullScreenTextureFS);
	FullScreenTexture.link();
	
	if(!FullScreenTexture) return 0;
	FullScreenTexture.setUniform("Texture", 0);
	
	ComputeShader& DeferredCS = ResourcesManager::getInstance().getShader<ComputeShader>("DeferredCS");
	DeferredCS.loadFromFile("src/GLSL/Deferred/tiled_deferred_cs.glsl");
	DeferredCS.compile();
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Camera Initialization
	
	Camera MainCamera;
	UniformBuffer CameraBuffer;
	CameraBuffer.init();
	CameraBuffer.bind(0);
	Deferred.bindUniformBlock("Camera", CameraBuffer); 
	//PostProcess.bindUniformBlock("Camera", CameraBuffer);
	DeferredCS.getProgram().bindUniformBlock("Camera", CameraBuffer);
	
	PostProcessMaterial.setUniform("cameraPosition", &MainCamera.getPosition());
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Light initialization
	
	const size_t LightCount = 75;
	Deferred.setUniform("lightCount", LightCount);
	PostProcessMaterial.setUniform("lightCount", LightCount);
	DeferredCS.getProgram().setUniform("lightCount", LightCount);
	
	UniformBuffer LightBuffer; //(Buffer::ShaderStorage);
	
	LightBuffer.init();
	LightBuffer.bind(1);

	PostProcess.bindUniformBlock("LightBlock", LightBuffer);
	DeferredCS.getProgram().bindUniformBlock("LightBlock", LightBuffer);
	
	LightStruct tmpLight[LightCount];
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Loading Meshes and declaring instances
	
	std::vector<MeshInstance>							_meshInstances;
	std::vector<std::pair<size_t, std::string>>			_tweakbars;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Ground
	
	Texture2D GroundTexture;
	GroundTexture.load("in/Textures/stone/cracked_c.png");
	
	Mesh Plane;
	float s = 1000.f;
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(-s, 0.f, -s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(0.f, 10.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(-s, 0.f, s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(0.f, 0.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(s, 0.f, s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(10.f, 0.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(s, 0.f, -s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(10.f, 10.f)));
	Plane.getTriangles().push_back(Mesh::Triangle(0, 1, 2));
	Plane.getTriangles().push_back(Mesh::Triangle(0, 2, 3));
	Plane.createVAO();
	Plane.getMaterial().setShadingProgram(Deferred);
	Plane.getMaterial().setUniform("Texture", GroundTexture);
	Plane.getMaterial().setUniform("ModelMatrix", glm::mat4(1.0));
	
	_meshInstances.push_back(MeshInstance(Plane));
	_tweakbars.push_back(std::make_pair(_meshInstances.size() - 1, "Plane"));
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Balls
	
	auto BallV = Mesh::load("in/3DModels/poolball/Ball1.obj");
	auto& Ball = BallV[0];
	Texture2D BallTex;
	BallTex.load(std::string("in/3DModels/poolball/lawl.jpg"));
	
	Ball->getMaterial().setShadingProgram(Deferred);
	Ball->getMaterial().setUniform("Texture", BallTex);
	
	Ball->createVAO();
	
	size_t row_ball_count = 10;
	size_t col_ball_count = 10;
	for(size_t i = 0; i < row_ball_count; ++i)
		for(size_t j = 0; j < col_ball_count; ++j)
		{
			_meshInstances.push_back(MeshInstance(*Ball, 
				glm::scale(glm::translate(glm::mat4(1.0), 
					glm::vec3(40.0 * (i - 0.5 * row_ball_count), 20.0, 40.0 * (j - 0.5 * col_ball_count))), 
					glm::vec3(10.0))));
		}
	
	// Creating requested AntTweakBars
	//for(auto& p : _tweakbars)
	//	_meshInstances[p.first].getMaterial().createAntTweakBar(p.second);
	
	resize_callback(window, _width, _height);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Main Loop
	
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
				((float) i) * 10.0f * glm::vec4(std::cos(i + _time), 0.0, std::sin(i + _time), 1.0)
				 + glm::vec4(0.0, 20.0 * std::sin(i + 0.71 * _time) + 30.0, 0.0 , 1.0), 	// Position
				glm::vec4(i % 2, (i % 3) / 2.0, (i % 5)/4.0, 1.0)		// Color
			};
		}
		LightBuffer.data(&tmpLight, LightCount * sizeof(LightStruct), Buffer::DynamicDraw);
		////////////////////////////////////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Actual drawing
		
		// Offscreen
		_offscreenRender.bind();
		for(auto& b : _meshInstances)
		{
			if(isVisible(_projection * MainCamera.getMatrix(), b.getMesh().getBoundingBox()))
			{
				b.draw();
			}
		}
		_offscreenRender.unbind();
		
		// Post processing
		// Restore Viewport (binding the framebuffer modifies it - should I make the unbind call restore it ? How ?)
		glViewport(0, 0, _width, _height);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		if(_colorToRender == 0)
		{
			PostProcessMaterial.use();
		} else if(_colorToRender == 1) {	
			_offscreenRender.getColor(0).bindImage(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			_offscreenRender.getColor(1).bindImage(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
			_offscreenRender.getColor(2).bindImage(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
			DeferredCS.getProgram().setUniform("ColorDepth", (int) 0);
			DeferredCS.getProgram().setUniform("Position", 1);
			DeferredCS.getProgram().setUniform("Normal", 2);	
			DeferredCS.getProgram().setUniform("cameraPosition", MainCamera.getPosition());
			DeferredCS.use();
			DeferredCS.dispatchCompute(_resolution.x / 32 + 1, _resolution.y / 32 + 1, 1);
			DeferredCS.memoryBarrier();
		
			_offscreenRender.getColor(0).bind(0);
			FullScreenTexture.use();
		} else { 
			_offscreenRender.getColor(_colorToRender - 2).bind(0);
			FullScreenTexture.use();
		}
		
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		PostProcessMaterial.useNone();
		
		////////////////////////////////////////////////////////////////////////////////////////////
		
		// Quick Cleanup for AntTweakBar...
		for(int i = 0; i < 8; ++i)
		{
			Texture::activeUnit(i);
			glBindTexture(GL_TEXTURE_2D, 0);
			glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
		}
		
		TwDraw();
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	TwTerminate();
	
	glfwDestroyWindow(window);
}

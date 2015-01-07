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

// Temp
struct LightStruct
{
	glm::vec4	position;
	glm::vec4	color;
};

struct ShadowStruct
{
	glm::vec4	position;
	glm::vec4	color;
	glm::mat4	depthMVP;
};

struct CameraStruct
{
	glm::mat4	view;
	glm::mat4	projection;
};

// Checks whether the provided bounding bound is visible after its transformation by MVPMatrix
// Todo: Place in... MeshInstance ?
// Todo: There is some overdraw (object right behind the camera are reported as visible)
//		 but I don't know why :(
bool isVisible(const glm::mat4& ProjectionMatrix, const glm::mat4& ViewMatrix, const glm::mat4& ModelMatrix, const BoundingBox& bbox)
{
	const glm::vec4 a = ModelMatrix * glm::vec4(bbox.min, 1.0);
	const glm::vec4 b = ModelMatrix * glm::vec4(bbox.max, 1.0);
	
	std::array<glm::vec4, 8> p = {glm::vec4{a.x, a.y, a.z, 1.0},
								  glm::vec4{a.x, a.y, b.z, 1.0},
								  glm::vec4{a.x, b.y, a.z, 1.0},
								  glm::vec4{a.x, b.y, b.z, 1.0},
								  glm::vec4{b.x, a.y, a.z, 1.0},
								  glm::vec4{b.x, a.y, b.z, 1.0},
								  glm::vec4{b.x, b.y, a.z, 1.0},
								  glm::vec4{b.x, b.y, b.z, 1.0}};
						
	bool front = false;
	for(auto& t : p)
	{
		t = ViewMatrix * t;
		front = front || t.z < 0.0;
	}

	if(!front) return false;
	
	glm::vec2 min = glm::vec2(2.0, 2.0);
	glm::vec2 max = glm::vec2(-2.0, -2.0);
						
	for(auto& t : p)
	{
		t = ProjectionMatrix * t;
		if(t.w > 0.0) t /= t.w;
		min.x = std::min(min.x, t.x);
		min.y = std::min(min.y, t.y);
		max.x = std::max(max.x, t.x);
		max.y = std::max(max.y, t.y);
	}
	
	return !(max.x < -1.0 || max.y < -1.0 ||
			 min.x >  1.0 || min.y >  1.0);
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
			
	float LightRadius = 75.0;
	bool DrawLights = false;
	
	TwBar* bar = TwNewBar("Global Tweaks");
	TwDefine("'Global Tweaks' color='0 0 0' ");
	TwDefine("'Global Tweaks' iconified=true ");
	TwAddVarRO(bar, "FrameTime", TW_TYPE_FLOAT, &_frameTime, "");
	TwAddVarRO(bar, "FrameRate", TW_TYPE_FLOAT, &_frameRate, "");
	TwAddVarRW(bar, "TimeScale", TW_TYPE_FLOAT, &_timescale, "min=0.0 step=0.1");
	TwAddVarRW(bar, "FOV", TW_TYPE_FLOAT, &_fov, "min=0.0 step=0.1");
	TwAddVarRW(bar, "LightRadius", TW_TYPE_FLOAT, &LightRadius, "min=0.0 step=1.0");
	TwAddVarRO(bar, "Fullscreen (V to toogle)", TW_TYPE_BOOLCPP, &_fullscreen, "");
	TwAddVarRO(bar, "MSAA (X to toogle)", TW_TYPE_BOOLCPP, &_msaa, "");
	TwAddVarRW(bar, "Update VFC", TW_TYPE_BOOLCPP, &_updateVFC, "");
	TwAddVarRW(bar, "Draw Lights", TW_TYPE_BOOLCPP, &DrawLights, "");
	
	glEnable(GL_DEPTH_TEST);
	
	VertexShader& DeferredVS = ResourcesManager::getInstance().getShader<VertexShader>("Deferred_VS");
	DeferredVS.loadFromFile("src/GLSL/Deferred/deferred_vs.glsl");
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
	
	/*
	FragmentShader& FullScreenTextureFS = ResourcesManager::getInstance().getShader<FragmentShader>("FullScreenTextureFS");
	FullScreenTextureFS.loadFromFile("src/GLSL/DisplayTexture.glsl");
	FullScreenTextureFS.compile();
	
	Program& FullScreenTexture = ResourcesManager::getInstance().getProgram("FullScreenTexture");
	FullScreenTexture.attachShader(PostProcessVS);
	FullScreenTexture.attachShader(FullScreenTextureFS);
	FullScreenTexture.link();
	
	if(!FullScreenTexture) return 0;
	FullScreenTexture.setUniform("Texture", 0);
	*/
	
	ComputeShader& DeferredCS = ResourcesManager::getInstance().getShader<ComputeShader>("DeferredCS");
	DeferredCS.loadFromFile("src/GLSL/Deferred/tiled_deferred_cs.glsl");
	DeferredCS.compile();
	
	if(!DeferredCS.getProgram()) return 0;
	
	ComputeShader& DeferredShadowCS = ResourcesManager::getInstance().getShader<ComputeShader>("DeferredShadowCS");
	DeferredShadowCS.loadFromFile("src/GLSL/Deferred/tiled_deferred_shadow_cs.glsl");
	DeferredShadowCS.compile();
	
	if(!DeferredShadowCS.getProgram()) return 0;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Camera Initialization
	
	Camera MainCamera;
	UniformBuffer CameraBuffer;
	CameraBuffer.init();
	CameraBuffer.bind(0);
	Deferred.bindUniformBlock("Camera", CameraBuffer); 
	DeferredLight.bindUniformBlock("Camera", CameraBuffer);
	DeferredColor.bindUniformBlock("Camera", CameraBuffer);
	//DeferredShadowCS.getProgram().bindUniformBlock("Camera", CameraBuffer);
	
	PostProcessMaterial.setUniform("cameraPosition", &MainCamera.getPosition());
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Light initialization
	
	const size_t LightCount = 250;
	PostProcessMaterial.setUniform("lightCount", LightCount);
	DeferredCS.getProgram().setUniform("lightCount", LightCount);
	DeferredShadowCS.getProgram().setUniform("lightCount", LightCount);
	
	UniformBuffer LightBuffer;
	
	LightBuffer.init();
	LightBuffer.bind(1);

	PostProcess.bindUniformBlock("LightBlock", LightBuffer);
	DeferredCS.getProgram().bindUniformBlock("LightBlock", LightBuffer);
	DeferredShadowCS.getProgram().bindUniformBlock("LightBlock", LightBuffer);
	
	LightStruct tmpLight[LightCount];
	
	// Shadow casting lights ---------------------------------------------------
	const size_t ShadowCount = 3;
	UniformBuffer ShadowBuffers[3];
	DeferredShadowCS.getProgram().setUniform("shadowCount", ShadowCount);
	
	Light MainLights[ShadowCount];
	MainLights[0].init();
	MainLights[0].setColor(glm::vec4(0.5, 0.75, 0.5, 1.0));
	MainLights[0].setPosition(glm::vec3(100.0, 800.0, 100.0));
	MainLights[0].lookAt(glm::vec3(0.0));
	
	MainLights[1].init();
	MainLights[1].setColor(glm::vec4(0.75, 0.5, 0.5, 1.0));
	MainLights[1].setPosition(glm::vec3(-100.0, 800.0, 100.0));
	MainLights[1].lookAt(glm::vec3(0.0));
	
	MainLights[2].init();
	MainLights[2].setColor(glm::vec4(0.5, 0.5, 0.75, 1.0));
	MainLights[2].setPosition(glm::vec3(0.0, 800.0, -100.0));
	MainLights[2].lookAt(glm::vec3(0.0));
	
	for(size_t i = 0; i < ShadowCount; ++i)
	{
		ShadowBuffers[i].init();
		ShadowBuffers[i].bind(i + 2);
		MainLights[i].updateMatrices();
		ShadowStruct tmpShadows = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
		ShadowBuffers[i].data(&tmpShadows, sizeof(ShadowStruct), Buffer::DynamicDraw);
		
		//DeferredShadowCS.getProgram().bindUniformBlock(std::string("ShadowBlock[").append(StringConversion::to_string(i)).append("]"), ShadowBuffers[i]);
		DeferredShadowCS.getProgram().setUniform(std::string("ShadowMaps[").append(StringConversion::to_string(i)).append("]"), (int) i + 3);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Loading Meshes and declaring instances
	
	std::vector<MeshInstance>							_meshInstances;
	std::vector<std::pair<size_t, std::string>>			_tweakbars;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Ground
	
	Texture2D GroundTexture;
	GroundTexture.load("in/Textures/Tex0.jpg");
	Texture2D GroundNormalMap;
	GroundNormalMap.load("in/Textures/Tex0_n.jpg");
	
	Mesh Plane;
	float s = 1000.f;
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(-s, 0.f, -s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(0.f, 20.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(-s, 0.f, s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(0.f, 0.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(s, 0.f, s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(20.f, 0.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(s, 0.f, -s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(20.f, 20.f)));
	Plane.getTriangles().push_back(Mesh::Triangle(0, 1, 2));
	Plane.getTriangles().push_back(Mesh::Triangle(0, 2, 3));
	Plane.setBoundingBox({glm::vec3(-s, 0.f, -s), glm::vec3(s, 0.f, s)});
	Plane.createVAO();
	Plane.getMaterial().setShadingProgram(Deferred);
	Plane.getMaterial().setUniform("Texture", GroundTexture);
	Plane.getMaterial().setUniform("NormalMap", GroundNormalMap);
	
	_meshInstances.push_back(MeshInstance(Plane));
	_tweakbars.push_back(std::make_pair(_meshInstances.size() - 1, "Plane"));
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Balls
	
	auto BallV = Mesh::load("in/3DModels/poolball/Ball1.obj");
	auto& Ball = BallV[0];
	Texture2D BallTex;
	BallTex.load(std::string("in/3DModels/poolball/lawl.jpg"));
	Texture2D BallNormalMap;
	BallNormalMap.load(std::string("in/3DModels/poolball/lawl_n.jpg"));
	
	Ball->getMaterial().setShadingProgram(Deferred);
	Ball->getMaterial().setUniform("Texture", BallTex);
	Ball->getMaterial().setUniform("NormalMap", BallNormalMap);
	
	Ball->createVAO();
	
	size_t row_ball_count = 20;
	size_t col_ball_count = 20;
	for(size_t i = 0; i < row_ball_count; ++i)
		for(size_t j = 0; j < col_ball_count; ++j)
		{
			_meshInstances.push_back(MeshInstance(*Ball, 
				glm::scale(glm::translate(glm::mat4(1.0), 
					glm::vec3(40.0 * (i - 0.5 * row_ball_count), 20.0, 40.0 * (j - 0.5 * col_ball_count))), 
					glm::vec3(10.0))));
		}
		
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Light Sphere Mesh
	auto LightSphereV = Mesh::load("in/3DModels/sphere/sphere.obj");
	auto& LightSphere = LightSphereV[0];
	LightSphere->createVAO();
	
	// Creating requested AntTweakBars
	//for(auto& p : _tweakbars)
	//	_meshInstances[p.first].getMaterial().createAntTweakBar(p.second);
	
	resize_callback(window, _width, _height);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Main Loop
	
	MainCamera.updateView();
	glm::mat4 VFC_ViewMatrix = MainCamera.getMatrix();
	
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
				((float) i) * 5.0f * glm::vec4(std::cos(i + _time), 0.0, std::sin(i + _time), 1.0)
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
		_offscreenRender.clear();
		if(_updateVFC)
			VFC_ViewMatrix = MainCamera.getMatrix();
			
		for(auto& b : _meshInstances)
		{
			if(isVisible(_projection, VFC_ViewMatrix, b.getModelMatrix(), b.getMesh().getBoundingBox()))
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
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
			PostProcessMaterial.useNone();
		} else if(_colorToRender == 1) {			
			// Cull Front and Disable Depth Test : It doesn't matter for a sphere :]
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
			glDisable(GL_DEPTH_TEST);
			glDepthMask(GL_FALSE);
			
			glEnable(GL_BLEND);
			glBlendEquation(GL_FUNC_ADD);
			glBlendFunc(GL_ONE, GL_ONE);
			_offscreenRender.getColor(0).bind(0);
			_offscreenRender.getColor(1).bind(1);
			_offscreenRender.getColor(2).bind(2);
			DeferredLight.setUniform("Color", (int) 0);
			DeferredLight.setUniform("Position", (int) 1);
			DeferredLight.setUniform("Normal", (int) 2);	
			DeferredLight.setUniform("cameraPosition", MainCamera.getPosition());
			DeferredLight.setUniform("lightRadius", LightRadius);
			DeferredLight.use();
			for(int l = 0; l < (int) LightCount; ++l)
			{
				//LightBuffer.bindRange(2, sizeof(LightStruct) * l, sizeof(LightStruct));
				DeferredLight.setUniform("LightPosition", tmpLight[l].position);			
				DeferredLight.setUniform("LightColor", tmpLight[l].color);
				glm::mat4 model = glm::scale(glm::translate(glm::mat4(1.0), glm::vec3(tmpLight[l].position)), glm::vec3(LightRadius));
				if(isVisible(_projection, MainCamera.getMatrix(), model, LightSphere->getBoundingBox()))
				{
					DeferredLight.setUniform("ModelMatrix", model);
					LightSphere->draw();
				}
			}
			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);
			glEnable(GL_DEPTH_TEST);
			glCullFace(GL_BACK);
		} else if(_colorToRender == 2) {	
			_offscreenRender.getColor(0).bindImage(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			_offscreenRender.getColor(1).bindImage(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
			_offscreenRender.getColor(2).bindImage(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
			DeferredCS.getProgram().setUniform("ColorMaterial", (int) 0);
			DeferredCS.getProgram().setUniform("PositionDepth", (int) 1);
			DeferredCS.getProgram().setUniform("Normal", (int) 2);	
			DeferredCS.getProgram().setUniform("cameraPosition", MainCamera.getPosition());
			DeferredCS.getProgram().setUniform("lightRadius", LightRadius);
			DeferredCS.compute(_resolution.x / DeferredCS.getWorkgroupSize().x + 1, _resolution.y / DeferredCS.getWorkgroupSize().y + 1, 1);
			DeferredCS.memoryBarrier();
		
			// Blitting
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
			_offscreenRender.bind(FramebufferTarget::Read);
			glBlitFramebuffer(0, 0, _resolution.x, _resolution.y, 0, 0, _resolution.x, _resolution.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		} else if(_colorToRender == 3) {	
			////////////////////////////////////////////////////////////////////////////////////////
			// ShadowMaps update
			
			MainLights[0].setPosition(300.0f * glm::vec3(std::sin(_time * 0.25), 0.0, std::cos(_time * 0.25)) + glm::vec3(0.0, 500.0 , 0.0));
			MainLights[1].setPosition(400.0f * glm::vec3(std::sin(_time * 0.4), 0.0, std::cos(_time * 0.4)) + glm::vec3(0.0, 400.0 , 0.0));
			MainLights[2].setPosition(100.0f * glm::vec3(std::sin(_time * 0.1), 0.0, std::cos(_time * 0.1)) + glm::vec3(0.0, 700.0 , 0.0));
			
			for(size_t i = 0; i < ShadowCount; ++i)
			{
				MainLights[i].lookAt(glm::vec3(0.0, 0.0, 0.0));
				MainLights[i].updateMatrices();
				ShadowStruct tmpShadows = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
				ShadowBuffers[i].data(&tmpShadows, sizeof(ShadowStruct), Buffer::DynamicDraw);
				
				MainLights[i].bind();
				
				for(auto& b : _meshInstances)
					if(isVisible(MainLights[i].getProjectionMatrix(), MainLights[i].getViewMatrix(), b.getModelMatrix(), b.getMesh().getBoundingBox()))
					{
						Light::getShadowMapProgram().setUniform("ModelMatrix", b.getModelMatrix());
						b.getMesh().draw();
					}
				
				MainLights[i].unbind();
			}
		
			_offscreenRender.getColor(0).bindImage(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
			_offscreenRender.getColor(1).bindImage(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
			_offscreenRender.getColor(2).bindImage(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
			for(size_t i = 0; i < ShadowCount; ++i)
				MainLights[i].getShadowMap().bind(i + 3);
			DeferredShadowCS.getProgram().setUniform("ColorMaterial", (int) 0);
			DeferredShadowCS.getProgram().setUniform("PositionDepth", (int) 1);
			DeferredShadowCS.getProgram().setUniform("Normal", (int) 2);	
			DeferredShadowCS.getProgram().setUniform("cameraPosition", MainCamera.getPosition());
			DeferredShadowCS.getProgram().setUniform("lightRadius", LightRadius);
			DeferredShadowCS.compute(_resolution.x / DeferredCS.getWorkgroupSize().x + 1, _resolution.y / DeferredCS.getWorkgroupSize().y + 1, 1);
			DeferredShadowCS.memoryBarrier();
		
			// Blitting
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
			_offscreenRender.bind(FramebufferTarget::Read);
			glBlitFramebuffer(0, 0, _resolution.x, _resolution.y, 0, 0, _resolution.x, _resolution.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		} else { 
			// Blitting
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
			_offscreenRender.bind(FramebufferTarget::Read);
			glReadBuffer(GL_COLOR_ATTACHMENT0 + (_colorToRender - 4));
			glBlitFramebuffer(0, 0, _resolution.x, _resolution.y, 0, 0, _resolution.x, _resolution.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		}
		
		if(DrawLights)
		{
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendEquation(GL_FUNC_ADD);
			glBlendFunc(GL_ONE, GL_ONE);
			_offscreenRender.getColor(0).bind(0);
			DeferredColor.setUniform("ColorMaterial", 0);
			for(const auto& l : tmpLight)
			{
				glm::mat4 model = glm::translate(glm::mat4(1.0), glm::vec3(l.position));
				if(isVisible(_projection, VFC_ViewMatrix, model, LightSphere->getBoundingBox()))
				{
					DeferredColor.setUniform("Color", l.color);
					DeferredColor.setUniform("ModelMatrix", model);
					DeferredColor.use();
					LightSphere->draw();
				}
			}
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
		}
		
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

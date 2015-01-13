#include <cstdlib>
#include <ctime>
#include <sstream>
#include <map>
#include <random>
#include <iomanip>

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
#include <CubeMap.hpp>
#include <Framebuffer.hpp>
#include <Buffer.hpp>
#include <TransformFeedback.hpp>
#include <MeshInstance.hpp>
#include <MathTools.hpp>
#include <Camera.hpp>
#include <Skybox.hpp>
#include <Light.hpp>
#include <stb_image_write.hpp>
#include <Query.hpp>

int			_width = 1366;
int			_height = 720;

Camera MainCamera;
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

float		_timescale = 0.5;
float 		_time = 0.f;
float		_frameTime;
float		_frameRate;
bool		_paused = false;

int			_colorToRender = 3;

bool		_video = false;

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
	
	_offscreenRender = Framebuffer<Texture2D, 3>(_width, _height);
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
				break;
			}
			case GLFW_KEY_N:
			{
				_video = true;
				break;
			}
			case GLFW_KEY_KP_ADD:
			{
				if(MainCamera.speed() < 1)
					MainCamera.speed() += .1;
				else
					MainCamera.speed() += 1;
				std::cout << "Camera Speed: " << MainCamera.speed() << std::endl;
				break;
			}
			case GLFW_KEY_KP_SUBTRACT:
			{
				if(MainCamera.speed() <= 1)
					MainCamera.speed() -= .1;
				else
					MainCamera.speed() -= 1;
				std::cout << "Camera Speed: " << MainCamera.speed() << std::endl;
				break;
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
	
	Particle(float type, const glm::vec3& position, const glm::vec3& speed, float lifetime) :
		position_type(position, type),
		speed_lifetime(speed, lifetime)
	{
	}
};

struct ShadowStruct
{
	glm::vec4	position;
	glm::vec4	color;
	glm::mat4	depthMVP;
};

struct WaterCell
{
	glm::vec4	data; // water height, ground height, speed
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
	
	// Debug Context (Does it work? =.=)
	//glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	//glEnable(GL_DEBUG_OUTPUT);
    
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
			
	float LightRadius = 5.0;
	
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
	
	ComputeShader& DeferredShadowCS = ResourcesManager::getInstance().getShader<ComputeShader>("DeferredShadowCS");
	DeferredShadowCS.loadFromFile("src/GLSL/Deferred/tiled_deferred_shadow_cs.glsl");
	DeferredShadowCS.compile();
	
	if(!DeferredShadowCS.getProgram()) return 0;
	
	Program& ParticleUpdate = ResourcesManager::getInstance().getProgram("ParticleUpdate");
	VertexShader& ParticleUpdateVS = ResourcesManager::getInstance().getShader<VertexShader>("ParticleUpdate_VS");
	ParticleUpdateVS.loadFromFile("src/GLSL/Particles/update_vs.glsl");
	ParticleUpdateVS.compile();
	GeometryShader& ParticleUpdateGS = ResourcesManager::getInstance().getShader<GeometryShader>("ParticleUpdate_GS");
	ParticleUpdateGS.loadFromFile("src/GLSL/Particles/update_heightmap_gs.glsl");
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
	
	ComputeShader& WaterUpdate = ResourcesManager::getInstance().getShader<ComputeShader>("WaterUpdate");
	WaterUpdate.loadFromFile("src/GLSL/Water/update_cs.glsl");
	WaterUpdate.compile();
	
	if(!WaterUpdate.getProgram()) return 0;
	
	Program& WaterDraw = ResourcesManager::getInstance().getProgram("WaterDraw");
	VertexShader& WaterDrawVS = ResourcesManager::getInstance().getShader<VertexShader>("WaterDraw_VS");
	WaterDrawVS.loadFromFile("src/GLSL/Water/draw_vs.glsl");
	WaterDrawVS.compile();
	GeometryShader& WaterDrawGS = ResourcesManager::getInstance().getShader<GeometryShader>("WaterDraw_GS");
	WaterDrawGS.loadFromFile("src/GLSL/Water/draw_gs.glsl");
	WaterDrawGS.compile();
	FragmentShader& WaterDrawFS = ResourcesManager::getInstance().getShader<FragmentShader>("WaterDraw_FS");
	WaterDrawFS.loadFromFile("src/GLSL/Water/draw_fs.glsl");
	WaterDrawFS.compile();
	WaterDraw.attachShader(WaterDrawVS);
	WaterDraw.attachShader(WaterDrawGS);
	WaterDraw.attachShader(WaterDrawFS);
	WaterDraw.link();
	 
	if(!WaterDraw) return 0;
		
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Camera Initialization
	
	MainCamera.speed() = 15;
	MainCamera.setPosition(glm::vec3(0.0, 15.0, -20.0));
	MainCamera.lookAt(glm::vec3(0.0, 5.0, 0.0));
	UniformBuffer CameraBuffer;
	CameraBuffer.init();
	CameraBuffer.bind(0);
	Deferred.bindUniformBlock("Camera", CameraBuffer); 
	ParticleDraw.bindUniformBlock("Camera", CameraBuffer);
	WaterDraw.bindUniformBlock("Camera", CameraBuffer);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Loading Meshes and declaring instances
	
	std::vector<MeshInstance>							_meshInstances;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Ground
	
	Texture2D GroundTexture;
	GroundTexture.load("in/Textures/Tex0.jpg");
	Texture2D GroundNormalMap;
	GroundNormalMap.load("in/Textures/Tex0_n.jpg");
	
	Mesh Plane;
	float s = 100.f;
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
	Plane.getMaterial().setUniform("useNormalMap", 1);
	Plane.getMaterial().setUniform("NormalMap", GroundNormalMap);
	
	_meshInstances.push_back(MeshInstance(Plane));
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Particles
	
	const size_t ParticleCount = 100;
	const float ParticleSize = 0.1;
	std::vector<Particle> particles;
	for(int i = 0; i < (int) ParticleCount; ++i)
		particles.push_back(Particle(i, glm::vec3{i * 0.01, 10.0, i * 0.02}, 4.0f * std::cos(1.0f * i) * glm::vec3{std::cos(3.14 * 0.02 * i), (i % 10) * 0.25, std::sin(3.14 * 0.02 * i)}, 10.0));
	
	Buffer particles_buffers[2];
	TransformFeedback particles_transform_feedback[2];
	for(int i = 0; i < 2; ++i)
	{
		particles_transform_feedback[i].init();
		particles_transform_feedback[i].bind();
		particles_buffers[i].init(Buffer::Target::VertexAttributes);
		particles_buffers[i].bind();
		particles_buffers[i].data(particles.data(), sizeof(Particle) * particles.size(), Buffer::Usage::DynamicDraw);
		particles_transform_feedback[i].bindBuffer(0, particles_buffers[i]);
		
		//particles_buffers[i].bind(Buffer::Uniform, (GLuint) i + 1); // Using them as light sources. Yeah.
	}
	
	size_t ParticleStep = 0;
	ParticleUpdate.setUniform("particle_size", ParticleSize);
	ParticleDraw.setUniform("particle_size", ParticleSize);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Water
	std::vector<WaterCell> water;
	size_t water_x = 200;
	size_t water_z = 200;
	float water_cellsize = 0.05;
	float water_moyheight = 2.0;
			
	for(size_t i = 0; i < water_x; ++i)
		for(size_t j = 0; j < water_z; ++j)
			water.push_back(WaterCell{glm::vec4{
										//water_moyheight,
										water_moyheight + 0.1 * std::cos(0.1 * std::sqrt(((double) i - water_x * 0.5)*((double) i - water_x * 0.5) + ((double) j - water_z/2.0) *((double) j - water_z/2.0))), 
										0.0,
										0.0,
										0.0}});

	ShaderStorage water_buffer;
	water_buffer.init();
	water_buffer.bind(4);
	water_buffer.data(water.data(), sizeof(WaterCell) * water.size(), Buffer::Usage::DynamicDraw);
	
	glm::mat4 WaterModelMatrix = glm::translate(glm::mat4(1.0), - glm::vec3(water_x * water_cellsize * 0.5, 0.0, water_z * water_cellsize * 0.5));
	
	WaterUpdate.getProgram().setUniform("size_x", (int) water_x);
	WaterUpdate.getProgram().setUniform("size_y", (int) water_z);
	WaterUpdate.getProgram().setUniform("cell_size", water_cellsize);
	WaterUpdate.getProgram().setUniform("moyheight", water_moyheight);
	WaterUpdate.getProgram().setUniform("iterations", 10);
	WaterUpdate.getProgram().bindShaderStorageBlock("InBuffer", water_buffer);
	
	ParticleUpdate.setUniform("size_x", (int) water_x);
	ParticleUpdate.setUniform("size_y", (int) water_z);
	ParticleUpdate.setUniform("cell_size", water_cellsize);
	ParticleUpdate.bindShaderStorageBlock("InBuffer", water_buffer);
	ParticleUpdate.setUniform("HeightmapModelMatrix", WaterModelMatrix);
	
	WaterDraw.setUniform("size_x", (int) water_x);
	WaterDraw.setUniform("size_y", (int) water_z);
	WaterDraw.setUniform("cell_size", water_cellsize);
	WaterDraw.setUniform("ModelMatrix", WaterModelMatrix);
	WaterDraw.bindShaderStorageBlock("InBuffer", water_buffer);

	resize_callback(window, _width, _height);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Light initialization
	
	const size_t LightCount = particles.size();
	DeferredShadowCS.getProgram().setUniform("lightCount", LightCount);
	DeferredShadowCS.getProgram().setUniform("lightRadius", LightRadius);
	
	UniformBuffer LightBuffer;
	
	LightBuffer.init();
	LightBuffer.bind(1);

	DeferredShadowCS.getProgram().bindUniformBlock("LightBlock", LightBuffer);
	
	LightStruct tmpLight[LightCount];
	for(size_t i = 0; i < LightCount; ++i)
	{
		tmpLight[i] = {
			glm::vec4(0.0), 	// Position
			glm::vec4(0.5 + 0.5 * (i % 2), 0.5 + 0.5 * (i % 3), 0.5 + 0.5 * (i % 5), 1.0)		// Color
		};
	}
	LightBuffer.data(&tmpLight, LightCount * sizeof(LightStruct), Buffer::Usage::DynamicDraw);
		
	// Shadow casting lights ---------------------------------------------------
	const size_t ShadowCount = 4;
	UniformBuffer ShadowBuffers[ShadowCount];
	DeferredShadowCS.getProgram().setUniform("shadowCount", ShadowCount);
	
	Light MainLights[ShadowCount];
	MainLights[0].init();
	MainLights[0].setColor(glm::vec4(0.3));
	MainLights[0].setPosition(glm::vec3(0.0, 40.0, 100.0));
	MainLights[0].lookAt(glm::vec3(0.0, 10.0, 0.0));
	
	for(size_t i = 0; i < ShadowCount; ++i)
	{
		ShadowBuffers[i].init();
		ShadowBuffers[i].bind(i + 2);
		MainLights[i].updateMatrices();
		ShadowStruct tmpShadows = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
		ShadowBuffers[i].data(&tmpShadows, sizeof(ShadowStruct), Buffer::Usage::DynamicDraw);
		
		DeferredShadowCS.getProgram().setUniform(std::string("ShadowMaps[").append(StringConversion::to_string(i)).append("]"), (int) i + 3);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Main Loop
	
	MainCamera.updateView();
	bool firstStep = true;
	
	Query LightQuery, ParticleQuery, ParticleDrawQuery, WaterQuery, WaterDrawQuery, ShadowMapQuery;
			
	glfwGetCursorPos(window, &_mouse_x, &_mouse_y); // init mouse position
	while(!glfwWindowShouldClose(window))
	{	
		// Time Management 
		TimeManager::getInstance().update();
		_frameTime = TimeManager::getInstance().getRealDeltaTime();
		_frameRate = TimeManager::getInstance().getInstantFrameRate();
		if(!_paused)
		{
			_time += _timescale * _frameTime;
			_frameTime *= _timescale;
		} else _frameTime = 0.0;
		
		// Camera Management
		if(_controlCamera)
		{
			float _frameTime = TimeManager::getInstance().getRealDeltaTime(); // Should move even on pause :)
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
			if(_mouse_x != mx || _mouse_y != my)
				MainCamera.look(glm::vec2(_mouse_x - mx, my - _mouse_y));
		}
		MainCamera.updateView();
		// Uploading camera data to the corresponding camera buffer
		CameraStruct CamS = {MainCamera.getMatrix(), _projection};
		CameraBuffer.data(&CamS, sizeof(CameraStruct), Buffer::Usage::DynamicDraw);
		
		// (Updating window title)
		std::ostringstream oss;
		oss.setf(std::ios::fixed, std:: ios::floatfield);
		oss.precision(2);
		oss << std::setw(6) << std::setfill('0') << _frameRate;
		oss << " - ShadowMap: " << std::setw(6) << std::setfill('0') << ShadowMapQuery.get<GLuint64>() / 1000000.0 << " ms";
		oss << " - Light: " << std::setw(6) << std::setfill('0') << LightQuery.get<GLuint64>() / 1000000.0 << " ms";
		oss << " - Particles: " << std::setw(6) << std::setfill('0') << ParticleQuery.get<GLuint64>() / 1000000.0 << " ms";
		oss << " (" << std::setw(6) << std::setfill('0') << ParticleDrawQuery.get<GLuint64>() / 1000000.0 << " ms)";
		oss << " - Water: " << std::setw(6) << std::setfill('0') << WaterQuery.get<GLuint64>() / 1000000.0 << " ms";
		oss << " (" << std::setw(6) << std::setfill('0') << WaterDrawQuery.get<GLuint64>() / 1000000.0 << " ms)";
		glfwSetWindowTitle(window, ((std::string("OpenGL ToolBox Test - FPS: ") + oss.str()).c_str()));
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Particle Update
		
		TransformFeedback::disableRasterization();
		ParticleQuery.begin(Query::Target::TimeElapsed);
		ParticleUpdate.setUniform("time", _frameTime);
		ParticleUpdate.use();
		
		particles_buffers[ParticleStep].bind();
		particles_transform_feedback[(ParticleStep + 1) % 2].bind();
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid *) offsetof(struct Particle, position_type)); // position_type
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid *) offsetof(struct Particle, speed_lifetime)); // speed_lifetime
	 
		TransformFeedback::begin(Primitive::Points);
		
		if(firstStep)
		{
			glDrawArrays(GL_POINTS, 0, particles.size());
		} else {
			particles_transform_feedback[ParticleStep].draw(Primitive::Points);
		}
		
		TransformFeedback::end();
		ParticleQuery.end();
		TransformFeedback::enableRasterization();
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Water Update
		WaterUpdate.use();
	 
		const int WaterInterations = 1;
		WaterUpdate.getProgram().setUniform("time", _frameTime / WaterInterations);
		WaterQuery.begin(Query::Target::TimeElapsed);
		for(int i = 0; i < WaterInterations; ++i)
		{
			WaterUpdate.compute(water_x / WaterUpdate.getWorkgroupSize().x + 1, water_z / WaterUpdate.getWorkgroupSize().y + 1, 1);
			WaterUpdate.memoryBarrier();
		}
		WaterQuery.end();
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// ShadowMap drawing
		
		ShadowMapQuery.begin(Query::Target::TimeElapsed);
		for(size_t i = 0; i < ShadowCount; ++i)
		{
			MainLights[i].lookAt(glm::vec3(0.0, 0.0, 0.0));
			MainLights[i].updateMatrices();
			ShadowStruct tmpShadows = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
			ShadowBuffers[i].data(&tmpShadows, sizeof(ShadowStruct), Buffer::Usage::DynamicDraw);
			
			MainLights[i].bind();
			
			for(auto& b : _meshInstances)
				if(isVisible(MainLights[i].getProjectionMatrix(), MainLights[i].getViewMatrix(), b.getModelMatrix(), b.getMesh().getBoundingBox()))
				{
					Light::getShadowMapProgram().setUniform("ModelMatrix", b.getModelMatrix());
					b.getMesh().draw();
				}
		
			MainLights[i].unbind();
		}
		ShadowMapQuery.end();
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Actual drawing
		
		// Offscreen
		_offscreenRender.bind();
		_offscreenRender.clear();
		
		// Particles
		ParticleDrawQuery.begin(Query::Target::TimeElapsed);
		ParticleDraw.setUniform("cameraPosition", MainCamera.getPosition());
		ParticleDraw.setUniform("cameraRight", MainCamera.getRight());
		ParticleDraw.use();
		
		particles_buffers[(ParticleStep + 1) % 2].bind();
		
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid *) offsetof(struct Particle, position_type)); // position_type
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid *) offsetof(struct Particle, speed_lifetime)); // speed_lifetime
		
		particles_transform_feedback[(ParticleStep + 1) % 2].draw(Primitive::Points);
		ParticleDrawQuery.end();
		
		// Water
		WaterDrawQuery.begin(Query::Target::TimeElapsed);
		WaterDraw.use();
		glDrawArrays(GL_POINTS, 0, water.size());
		WaterDrawQuery.end();
		
		// Meshes
			
		for(auto& b : _meshInstances)
		{
			if(isVisible(_projection, MainCamera.getMatrix(), b.getModelMatrix(), b.getMesh().getBoundingBox()))
				b.draw();
		}
		
		_offscreenRender.unbind();		
		
		// Use particles as lights, really sub optimal, but the light and particle structures were not designed to work together :)
		//DeferredCS.getProgram().bindUniformBlock("LightBlock", particles_buffers[ParticleStep]); // Not anymore, but it was cool.
		for(size_t i = 0; i < particles.size(); ++i)
			Buffer::copySubData(particles_buffers[(ParticleStep + 1) % 2], LightBuffer, sizeof(Particle) * i, sizeof(LightStruct) * i, sizeof(glm::vec4));
		
		ParticleStep = (ParticleStep + 1) % 2;
		
		// Post processing
		// Restore Viewport (binding the framebuffer modifies it - should I make the unbind call restore it ? How ?)
		glViewport(0, 0, _width, _height);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
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
		LightQuery.begin(Query::Target::TimeElapsed);
		DeferredShadowCS.compute(_resolution.x / DeferredShadowCS.getWorkgroupSize().x + 1, _resolution.y / DeferredShadowCS.getWorkgroupSize().y + 1, 1);
		DeferredShadowCS.memoryBarrier();
		LightQuery.end();
		
		// Blitting
		Framebuffer<>::unbind(FramebufferTarget::Draw);
		_offscreenRender.bind(FramebufferTarget::Read);
		glBlitFramebuffer(0, 0, _resolution.x, _resolution.y, 0, 0, _resolution.x, _resolution.y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
		
		////////////////////////////////////////////////////////////////////////////////////////////
		
		glfwSwapBuffers(window);
		glfwPollEvents();
		
		if(_video)
		{
			static int framecount = 0;
			screen("out/video/" + StringConversion::to_string(framecount) + ".png");
			framecount++;
			if(framecount > 60)
			{
				framecount = 0;
				_video = false;
			}
		}
		
		firstStep = false;
	}
	
	glfwDestroyWindow(window);
}

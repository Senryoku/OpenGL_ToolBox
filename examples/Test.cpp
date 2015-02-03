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

bool		_video = false;

Framebuffer<Texture2D, 3>	_offscreenRender;
Framebuffer<Texture2D, 3>	_offscreenRenderTransparency;
	
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
	_offscreenRender.getColor(0).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.getColor(1).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.getColor(2).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRender.init();
	
	_offscreenRenderTransparency = Framebuffer<Texture2D, 3>(_width, _height);
	_offscreenRenderTransparency.getColor(0).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRenderTransparency.getColor(1).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRenderTransparency.getColor(2).create(nullptr, _width, _height, GL_RGBA32F, GL_RGBA, false);
	_offscreenRenderTransparency.getDepth() = _offscreenRender.getDepth();
	_offscreenRenderTransparency.init();
	
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
	if(argc > 1)
	{
		_fullscreen = true;
	}
	
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
	
	GLFWwindow* window = nullptr;
	if(!_fullscreen)
	{
		window = glfwCreateWindow(_width, _height, "OpenGL ToolBox Test", nullptr, nullptr);
	} else {
		auto videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		_width = videoMode->width;
		_height = videoMode->height;
		window = glfwCreateWindow(_width, _height, "OpenGL ToolBox Test", glfwGetPrimaryMonitor(), nullptr);
	}
	
	if (!window)
	{
		std::cerr << "Error: couldn't create window." << std::endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	
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
	Deferred.attach(DeferredVS);
	Deferred.attach(DeferredFS);
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
	ParticleUpdate.attach(ParticleUpdateVS);
	ParticleUpdate.attach(ParticleUpdateGS);
	ParticleUpdate.setTransformFeedbackVaryings<2>({"position_type", "speed_lifetime"}, true);
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
	ParticleDraw.attach(ParticleDrawVS);
	ParticleDraw.attach(ParticleDrawGS);
	ParticleDraw.attach(ParticleDrawFS);
	ParticleDraw.link();
	 
	if(!ParticleDraw) return 0;
	
	ComputeShader& WaterUpdate = ResourcesManager::getInstance().getShader<ComputeShader>("WaterUpdate");
	WaterUpdate.loadFromFile("src/GLSL/Water/update_cs.glsl");
	WaterUpdate.compile();
	
	if(!WaterUpdate.getProgram()) return 0;
	
	Program& WaterDrawTF = ResourcesManager::getInstance().getProgram("WaterDrawTF");
	VertexShader& WaterDrawTFVS = ResourcesManager::getInstance().getShader<VertexShader>("WaterDraw_TF_VS");
	WaterDrawTFVS.loadFromFile("src/GLSL/Water/draw_tf_vs.glsl");
	WaterDrawTFVS.compile();
	GeometryShader& WaterDrawTFGS = ResourcesManager::getInstance().getShader<GeometryShader>("WaterDraw_TF_GS");
	WaterDrawTFGS.loadFromFile("src/GLSL/Water/draw_tf_gs.glsl");
	WaterDrawTFGS.compile();
	WaterDrawTF.attach(WaterDrawTFVS);
	WaterDrawTF.attach(WaterDrawTFGS);
	WaterDrawTF.setTransformFeedbackVaryings<2>({"water", "ground"}, false);
	WaterDrawTF.link();
	
	if(!WaterDrawTF) return 0;
	
	ComputeShader& WaterComputeNormals = ResourcesManager::getInstance().getShader<ComputeShader>("WaterComputeNormals");
	WaterComputeNormals.loadFromFile("src/GLSL/normals_cs.glsl");
	WaterComputeNormals.compile();
	
	if(!WaterComputeNormals.getProgram()) return 0;
	
	Program& WaterDraw = ResourcesManager::getInstance().getProgram("WaterDraw");
	VertexShader& WaterDrawVS = ResourcesManager::getInstance().getShader<VertexShader>("WaterDraw_VS");
	WaterDrawVS.loadFromFile("src/GLSL/Water/draw_indexed_vs.glsl");
	WaterDrawVS.compile();
	FragmentShader& WaterDrawFS = ResourcesManager::getInstance().getShader<FragmentShader>("WaterDraw_FS");
	WaterDrawFS.loadFromFile("src/GLSL/Water/draw_indexed_fs.glsl");
	WaterDrawFS.compile();
	WaterDraw.attach(WaterDrawVS);
	WaterDraw.attach(WaterDrawFS);
	WaterDraw.link();
	 
	if(!WaterDraw) return 0;
	
	VertexShader& BlendVS = ResourcesManager::getInstance().getShader<VertexShader>("BlendVS");
	BlendVS.loadFromFile("src/GLSL/vs.glsl");
	BlendVS.compile();

	FragmentShader& BlendFS = ResourcesManager::getInstance().getShader<FragmentShader>("BlendFS");
	BlendFS.loadFromFile("src/GLSL/blend_fs.glsl");
	BlendFS.compile();
	
	Program& Blend = ResourcesManager::getInstance().getProgram("Blend");
	Blend.attach(BlendVS);
	Blend.attach(BlendFS);
	Blend.link();
	
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
	
	Texture2D GroundTexture("in/Textures/Tex0.jpg");
	Texture2D GroundNormalMap("in/Textures/Tex0_n.jpg");
	GroundTexture.bind(0);
	Deferred.setUniform("Texture", (int) 0);
	GroundNormalMap.bind(1);
	Deferred.setUniform("NormalMap", (int) 1);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Particles
	
	const size_t ParticleCount = 250;
	const float ParticleSize = 0.2;
	std::vector<Particle> particles;
	for(int i = 0; i < (int) ParticleCount; ++i)
		particles.push_back(Particle(i, glm::vec3{i * 0.01, 10.0, i * 0.02}, 20.0f * std::cos(1.0f * i) * glm::vec3{std::cos(3.14 * 0.02 * i), (i % 10) * 0.25, std::sin(3.14 * 0.02 * i)}, 10.0));
	
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
	ParticleUpdate.setUniform("respawn_speed", 6.0f);
	ParticleDraw.setUniform("particle_size", ParticleSize);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Water
	std::vector<WaterCell> water;
	size_t water_x = 300;
	size_t water_z = 300;
	float water_cellsize = 0.2;
	float water_moyheight = 2.0;
			
	for(size_t i = 0; i < water_x; ++i)
		for(size_t j = 0; j < water_z; ++j)
			water.push_back(WaterCell{glm::vec4{
										//water_moyheight,
										water_moyheight + 1.5 * std::cos(0.05 * std::sqrt(((double) i - water_x * 0.2)*((double) i - water_x * 0.2) + ((double) j - water_z * 0.2) *((double) j - water_z * 0.2))), 
										0.2 + 0.2 * std::cos(0.1 * std::sqrt(((double) i - water_x * 0.5)*((double) i - water_x * 0.5) + ((double) j - water_z/2.0) *((double) j - water_z/2.0))), 
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
	
	WaterComputeNormals.getProgram().setUniform("size_x", (int) water_x);
	WaterComputeNormals.getProgram().setUniform("size_y", (int) water_z);
	
	WaterDrawTF.setUniform("size_x", (int) water_x);
	WaterDrawTF.setUniform("size_y", (int) water_z);
	WaterDrawTF.setUniform("cell_size", water_cellsize);
	WaterDrawTF.setUniform("ModelMatrix", WaterModelMatrix);

	TransformFeedback water_transform_feedback;
	water_transform_feedback.init();
	water_transform_feedback.bind();
	
	Buffer water_vertex_buffer;
	water_vertex_buffer.init(Buffer::Target::VertexAttributes);
	water_vertex_buffer.bind();
	water_vertex_buffer.data(nullptr, sizeof(WaterCell) * water.size(), Buffer::Usage::DynamicDraw);
	
	water_transform_feedback.bindBuffer(0, water_vertex_buffer);
	Buffer ground_vertex_buffer;
	ground_vertex_buffer.init(Buffer::Target::VertexAttributes);
	ground_vertex_buffer.bind();
	ground_vertex_buffer.data(nullptr, sizeof(WaterCell) * water.size(), Buffer::Usage::DynamicDraw);
	water_transform_feedback.bindBuffer(1, ground_vertex_buffer);
	
	water_transform_feedback.unbind();

	// Normal Buffers
	Buffer water_normal_buffer;
	water_normal_buffer.init(Buffer::Target::ShaderStorage);
	water_normal_buffer.bind();
	water_normal_buffer.data(nullptr, sizeof(glm::vec4) * water.size(), Buffer::Usage::DynamicDraw);

	Buffer ground_normal_buffer;
	ground_normal_buffer.init(Buffer::Target::ShaderStorage);
	ground_normal_buffer.bind();
	ground_normal_buffer.data(nullptr, sizeof(glm::vec4) * water.size(), Buffer::Usage::DynamicDraw);
	
	Buffer water_indices;
	water_indices.init(Buffer::Target::VertexIndices);
	water_indices.bind();
	std::vector<size_t> tmp_water_indices;
	tmp_water_indices.reserve((water_x - 1) * (water_z - 1) * 3 * 2);
	for(size_t i = 0; i < water_x - 1; ++i)
	{
		for(size_t j = 0; j < water_z - 1; ++j)
		{
			tmp_water_indices.push_back(i * water_z + j);
			tmp_water_indices.push_back(i * water_z + j + 1);
			tmp_water_indices.push_back((i + 1) * water_z + j);
			
			tmp_water_indices.push_back(i * water_z + j + 1);
			tmp_water_indices.push_back((i + 1) * water_z + j + 1);
			tmp_water_indices.push_back((i + 1) * water_z + j);
		}
	}
	water_indices.data(tmp_water_indices.data(), sizeof(size_t) * tmp_water_indices.size(), Buffer::Usage::StaticDraw);
	tmp_water_indices.clear();
	
	Buffer water_texcoord_buffer;
	water_texcoord_buffer.init(Buffer::Target::VertexIndices);
	water_texcoord_buffer.bind();
	std::vector<glm::vec2> tmp_water_texcoord;
	tmp_water_texcoord.resize(water_x * water_z);
	for(size_t i = 0; i < water_x; ++i)
	{
		for(size_t j = 0; j < water_z; ++j)
		{
				tmp_water_texcoord[i*water_z + j] = glm::vec2(((float) i) / water_x, ((float) j) / water_z);
		}
	}
	water_texcoord_buffer.data(tmp_water_texcoord.data(), sizeof(glm::vec2) * tmp_water_texcoord.size(), Buffer::Usage::StaticDraw);
	tmp_water_texcoord.clear();
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
	MainLights[0].setColor(glm::vec4(1.0));
	MainLights[0].setPosition(glm::vec3(0.0, 40.0, 100.0));
	MainLights[0].lookAt(glm::vec3(0.0, 10.0, 0.0));
	
	for(size_t i = 0; i < ShadowCount; ++i)
	{
		ShadowBuffers[i].init();
		ShadowBuffers[i].bind(i + 2);
		MainLights[i].updateMatrices();
		ShadowStruct tmpShadows = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
		ShadowBuffers[i].data(&tmpShadows, sizeof(ShadowStruct), Buffer::Usage::DynamicDraw);
		
		DeferredShadowCS.getProgram().setUniform(std::string("ShadowMaps[").append(std::to_string(i)).append("]"), (int) i + 3);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Main Loop
	
	MainCamera.updateView();
	bool firstStep = true;
	
	Query LightQuery, ParticleQuery, ParticleDrawQuery, WaterQuery, WaterTFQuery, WaterDrawQuery, ShadowMapQuery;
			
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
	
	Skybox Sky({"in/Textures/cubemaps/Park2/posx.jpg",
				"in/Textures/cubemaps/Park2/negx.jpg",
				"in/Textures/cubemaps/Park2/posy.jpg",
				"in/Textures/cubemaps/Park2/negy.jpg",
				"in/Textures/cubemaps/Park2/posz.jpg",
				"in/Textures/cubemaps/Park2/negz.jpg"
	});
		
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
			if(_frameTime > 1.0/60.0) _frameTime = 1.0/60.0; // In case the window is moved
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
		oss << " (" << std::setw(6) << std::setfill('0') << WaterTFQuery.get<GLuint64>() / 1000000.0 << " + ";
		oss << std::setw(6) << std::setfill('0') << WaterDrawQuery.get<GLuint64>() / 1000000.0 << " ms)";
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
		particles_transform_feedback[(ParticleStep + 1) % 2].unbind();
		TransformFeedback::enableRasterization();
		
		// Use particles as lights, really sub optimal, but the light and particle structures were not designed to work together :)
		for(size_t i = 0; i < particles.size(); ++i)
			Buffer::copySubData(particles_buffers[(ParticleStep + 1) % 2], LightBuffer, sizeof(Particle) * i, sizeof(LightStruct) * i, sizeof(glm::vec4));
		
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
		
		// Updating vertices
		TransformFeedback::disableRasterization();
		WaterTFQuery.begin(Query::Target::TimeElapsed);
		WaterDrawTF.use();
		
		water_buffer.bind(Buffer::Target::VertexAttributes);
		water_transform_feedback.bind();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(WaterCell), (const GLvoid *) offsetof(struct WaterCell, data));
		
		TransformFeedback::begin(Primitive::Points);
		glDrawArrays(GL_POINTS, 0, water.size());
		TransformFeedback::end();
		
		water_transform_feedback.unbind();
		
		water_vertex_buffer.bind(Buffer::Target::ShaderStorage, 5);
		water_normal_buffer.bind(Buffer::Target::ShaderStorage, 6);
		WaterComputeNormals.use();
		WaterComputeNormals.compute(water_x / WaterComputeNormals.getWorkgroupSize().x + 1, water_z / WaterComputeNormals.getWorkgroupSize().y + 1, 1);
		WaterComputeNormals.memoryBarrier();
		
		ground_vertex_buffer.bind(Buffer::Target::ShaderStorage, 5);
		ground_normal_buffer.bind(Buffer::Target::ShaderStorage, 6);
		WaterComputeNormals.use();
		WaterComputeNormals.compute(water_x / WaterComputeNormals.getWorkgroupSize().x + 1, water_z / WaterComputeNormals.getWorkgroupSize().y + 1, 1);
		WaterComputeNormals.memoryBarrier();
		
		WaterTFQuery.end();
		TransformFeedback::enableRasterization();
		
		////////////////////////////////////////////////////////////////////////////////////////////
		// Actual drawing
		
		// Offscreen
		_offscreenRender.bind();
		glDepthMask(GL_TRUE);
		_offscreenRender.clear();
		
		Sky.draw(_projection, MainCamera.getMatrix());
		
		// Particles
		glDisable(GL_CULL_FACE);
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
		
		ParticleStep = (ParticleStep + 1) % 2;
		glEnable(GL_CULL_FACE);
		
		// Ground
		GroundTexture.bind(0);
		Deferred.setUniform("Texture", (int) 0);
		GroundNormalMap.bind(1);
		Deferred.setUniform("NormalMap", (int) 1);
		Deferred.setUniform("ModelMatrix", glm::mat4(1.0));
		Deferred.use();
		ground_vertex_buffer.bind();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (const GLvoid *) 0);
		
		ground_normal_buffer.bind(Buffer::Target::VertexAttributes);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (const GLvoid *) 0);
		
		water_texcoord_buffer.bind(Buffer::Target::VertexAttributes);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (const GLvoid *) 0);
		
		water_indices.bind();
		glDrawElements(GL_TRIANGLES, (water_x - 1) * (water_z - 1) * 3 * 2, GL_UNSIGNED_INT, 0);
		
		// Meshes
			
		for(auto& b : _meshInstances)
		{
			if(isVisible(_projection, MainCamera.getMatrix(), b.getModelMatrix(), b.getMesh().getBoundingBox()))
				b.draw();
		}
		
		_offscreenRender.unbind();		
		
		////////////////////////////////////////////////////////////////
		// Transparency Offscreen
		
		_offscreenRenderTransparency.bind();	
		_offscreenRenderTransparency.clear(BufferBit::Color); // Keep depth buffer from opaque pass
		glDepthMask(GL_FALSE);
		// Water
		WaterDrawQuery.begin(Query::Target::TimeElapsed);
		Sky.getCubemap().bind(0);
		WaterDraw.setUniform("EnvMap", 0);
		WaterDraw.setUniform("cameraPosition", MainCamera.getPosition());
		WaterDraw.use();
		
		water_vertex_buffer.bind();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (const GLvoid *) 0);
		
		water_normal_buffer.bind(Buffer::Target::VertexAttributes);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (const GLvoid *) 0);
		
		water_indices.bind();
		glDrawElements(GL_TRIANGLES, (water_x - 1) * (water_z - 1) * 3 * 2, GL_UNSIGNED_INT, 0);
		WaterDrawQuery.end();
		_offscreenRenderTransparency.unbind();
				
		////////////////////////////////////////////////////////////////
		// Lightning
		LightQuery.begin(Query::Target::TimeElapsed);
		for(size_t i = 0; i < ShadowCount; ++i)
			MainLights[i].getShadowMap().bind(i + 3);
		DeferredShadowCS.getProgram().setUniform("ColorMaterial", (int) 0);
		DeferredShadowCS.getProgram().setUniform("PositionDepth", (int) 1);
		DeferredShadowCS.getProgram().setUniform("Normal", (int) 2);	
		DeferredShadowCS.getProgram().setUniform("cameraPosition", MainCamera.getPosition());
		DeferredShadowCS.getProgram().setUniform("lightRadius", LightRadius);
		
		// Opaque lightning
		_offscreenRender.getColor(0).bindImage(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
		_offscreenRender.getColor(1).bindImage(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		_offscreenRender.getColor(2).bindImage(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		DeferredShadowCS.compute(_resolution.x / DeferredShadowCS.getWorkgroupSize().x + 1, _resolution.y / DeferredShadowCS.getWorkgroupSize().y + 1, 1);
		
		// Transparent lightning
		_offscreenRenderTransparency.getColor(0).bindImage(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
		_offscreenRenderTransparency.getColor(1).bindImage(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		_offscreenRenderTransparency.getColor(2).bindImage(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
		DeferredShadowCS.compute(_resolution.x / DeferredShadowCS.getWorkgroupSize().x + 1, _resolution.y / DeferredShadowCS.getWorkgroupSize().y + 1, 1);
		
		ComputeShader::memoryBarrier();
		LightQuery.end();
		
		// Blending to default framebuffer
		_offscreenRender.getColor(0).bind(0);
		_offscreenRenderTransparency.getColor(0).bind(1);
		Blend.use();
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		////////////////////////////////////////////////////////////////////////////////////////////
		
		glfwSwapBuffers(window);
		glfwPollEvents();
		
		if(_video)
		{
			static int framecount = 0;
			screen("out/video/" + std::to_string(framecount) + ".png");
			framecount++;
			if(framecount > 5 * 60)
			{
				framecount = 0;
				_video = false;
			}
		}
		
		firstStep = false;
	}
	
	glfwDestroyWindow(window);
}

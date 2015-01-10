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

int		_width = 1366;
int		_height = 720;

glm::vec3 _resolution(_width, _height, 0.0);
glm::mat4 _projection;
glm::vec4 _mouse(0.0);

glm::vec4 _ambiant = glm::vec4(0.05f, 0.05f, 0.05f, 1.f);

float _ballsDiffuseReflection = 0.4f;

int 	_poissonSamples = 4;
float 	_poissonDiskRadius = 2500.f;

bool _fullscreen = false;
bool _msaa = false;
		
bool controlCamera = true;
double mouse_x, mouse_y;

float 	_time = 0.f;
float	_frameTime;
float	_frameRate;
bool	_paused = false;
	
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
	
	float inRad = 60.0 * pi()/180.f;
	_projection = glm::perspective(inRad, (float) _width/_height, 0.1f, 1000.0f);
	
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
					if(!controlCamera)
					{
						glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
					} else {
						glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
					}
					controlCamera = !controlCamera;
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
	glm::mat4	depthMVP;
};

struct CameraStruct
{
	glm::mat4	view;
	glm::mat4	projection;
};

// Todo: Place in... MeshInstance ?
// Check.
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
	
	// TODO: Construct and test the 8 points !!!
	
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
	TwAddVarRO(bar, "Fullscreen (V to toogle)", TW_TYPE_BOOLCPP, &_fullscreen, "");
	TwAddVarRO(bar, "MSAA (X to toogle)", TW_TYPE_BOOLCPP, &_msaa, "");
	TwAddVarRW(bar, "Ball Diffuse Reflection", TW_TYPE_FLOAT, &_ballsDiffuseReflection, "min=0 max=1 step=0.05");
	
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	CubeMap Tex;
	#define CUBEMAP_FOLDER "brudslojan"
	Tex.load({"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posz.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negz.jpg"
	});

	VertexShader& VS = ResourcesManager::getInstance().getShader<VertexShader>("NormalMap_VS");
	VS.loadFromFile("src/GLSL/NormalMap/normalmap_vs.glsl");
	VS.compile();

	FragmentShader& FS = ResourcesManager::getInstance().getShader<FragmentShader>("NormalMap_FS");
	FS.loadFromFile("src/GLSL/NormalMap/normalmap_fs.glsl");
	FS.compile();
	
	Program& NormalMap = ResourcesManager::getInstance().getProgram("NormalMap");
	NormalMap.attachShader(VS);
	NormalMap.attachShader(FS);
	NormalMap.link();

	if(!NormalMap) return 0;
	
	VertexShader& InstantiatedBallsVS = ResourcesManager::getInstance().getShader<VertexShader>("InstantiatedBalls_VS");
	InstantiatedBallsVS.loadFromFile("src/GLSL/InstantiatedBalls/normalmap_vs.glsl");
	InstantiatedBallsVS.compile();

	FragmentShader& InstantiatedBallsFS = ResourcesManager::getInstance().getShader<FragmentShader>("InstantiatedBalls_FS");
	InstantiatedBallsFS.loadFromFile("src/GLSL/InstantiatedBalls/normalmap_fs.glsl");
	InstantiatedBallsFS.compile();
	
	Program& InstantiatedBalls = ResourcesManager::getInstance().getProgram("InstantiatedBalls");
	InstantiatedBalls.attachShader(InstantiatedBallsVS);
	InstantiatedBalls.attachShader(InstantiatedBallsFS);
	InstantiatedBalls.link();
	
	if(!InstantiatedBalls) return 0;
	
	Light MainLights[3];
	MainLights[0].setColor(glm::vec4(1.0));
	MainLights[0].init();
	MainLights[0].setPosition(glm::vec3(100.0, 300.0, 100.0));
	MainLights[0].lookAt(glm::vec3(0.0));
	
	MainLights[1].setColor(glm::vec4(1.0, 0.9, 0.9, 1.0));
	MainLights[1].init();
	MainLights[1].setPosition(glm::vec3(100.0, 300.0, 100.0));
	MainLights[1].lookAt(glm::vec3(0.0));
	
	MainLights[2].setColor(glm::vec4(0.9, 0.9, 1.0, 1.0));
	MainLights[2].init();
	MainLights[2].setPosition(glm::vec3(100.0, 300.0, 100.0));
	MainLights[2].lookAt(glm::vec3(0.0));
	
	// TODO: Try using only one UBO
	size_t LightCount = 3;
	TwAddVarRW(bar, "LightCount", TW_TYPE_UINT8, &LightCount, "min=0 max=3");
	UniformBuffer LightBuffers[3];
	
	for(size_t i = 0; i < 3; ++i)
	{
		LightBuffers[i].init();
		LightBuffers[i].bind(i);
		MainLights[i].updateMatrices();
		LightStruct tmpLight = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
		LightBuffers[i].data(&tmpLight, sizeof(LightStruct), Buffer::Usage::DynamicDraw);
	}
	
	Camera MainCamera;
	UniformBuffer CameraBuffer;
	CameraBuffer.init();
	CameraBuffer.bind(LightCount);
	NormalMap.bindUniformBlock("Camera", CameraBuffer);
	InstantiatedBalls.bindUniformBlock("Camera", CameraBuffer);
	
	std::vector<MeshInstance>							_meshInstances;
	std::vector<std::pair<size_t, std::string>>			_tweakbars;
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// GladOS
	
	NormalMap.setUniform("lightCount", LightCount);
	InstantiatedBalls.setUniform("lightCount", LightCount);
	for(size_t i = 0; i < LightCount; ++i)
	{
		NormalMap.bindUniformBlock(std::string("LightBlock[").append(StringConversion::to_string(i)).append("]"), LightBuffers[i]);
		InstantiatedBalls.bindUniformBlock(std::string("LightBlock[").append(StringConversion::to_string(i)).append("]"), LightBuffers[i]);
		
		NormalMap.setUniform(std::string("ShadowMap[").append(StringConversion::to_string(i)).append("]"), (int) i + 2);
		InstantiatedBalls.setUniform(std::string("ShadowMap[").append(StringConversion::to_string(i)).append("]"), (int) i + 2);
		
		MainLights[i].bind();
		MainLights[i].unbind();
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Ground
	
	Texture2D GroundTexture;
	GroundTexture.load("in/Textures/stone/cracked_c.png");
	Texture2D GroundNormalMap;
	GroundNormalMap.load("in/Textures/stone/cracked_n.png");
	
	Mesh Plane;
	float s = 1000.f;
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(-s, 0.f, -s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(0.f, 10.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(-s, 0.f, s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(0.f, 0.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(s, 0.f, s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(10.f, 0.f)));
	Plane.getVertices().push_back(Mesh::Vertex(glm::vec3(s, 0.f, -s), glm::vec3(0.f, 1.0f, 0.0f), glm::vec2(10.f, 10.f)));
	Plane.getTriangles().push_back(Mesh::Triangle(0, 1, 2));
	Plane.getTriangles().push_back(Mesh::Triangle(0, 2, 3));
	Plane.createVAO();
	Plane.getMaterial().setShadingProgram(NormalMap);
	Plane.getMaterial().setUniform("Texture", GroundTexture);
	Plane.getMaterial().setUniform("NormalMap", GroundNormalMap);
	Plane.getMaterial().setUniform("ModelMatrix", glm::mat4(1.0));
	
	Plane.getMaterial().setUniform("ambiant", &_ambiant);
	
	Plane.getMaterial().setUniform("roughness", 0.2f);
	Plane.getMaterial().setUniform("F0", 0.2f);
	Plane.getMaterial().setUniform("diffuseReflection", 0.3f);
	
	Plane.getMaterial().setUniform("poissonSamples", &_poissonSamples);
	Plane.getMaterial().setUniform("poissonDiskRadius", &_poissonDiskRadius);
	
	_meshInstances.push_back(MeshInstance(Plane));
	_tweakbars.push_back(std::make_pair(_meshInstances.size() - 1, "Plane"));
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Balls
	
	auto BallV = Mesh::load("in/3DModels/poolball/Ball1.obj");
	auto& Ball = BallV[0];
	Texture2D BallTex;
	BallTex.load(std::string("in/3DModels/poolball/lawl.jpg"));
	
	Ball->getMaterial().setShadingProgram(InstantiatedBalls);
	
	Ball->getMaterial().setUniform("Texture", BallTex);
	Ball->getMaterial().setUniform("NormalMap", GroundNormalMap);
	
	Ball->getMaterial().setUniform("ambiant", &_ambiant);
	
	Ball->getMaterial().setUniform("diffuseReflection", &_ballsDiffuseReflection);
	
	Ball->getMaterial().setUniform("poissonSamples", &_poissonSamples);
	Ball->getMaterial().setUniform("poissonDiskRadius", &_poissonDiskRadius);
	
	Ball->createVAO();
	
	Buffer BallsModelMatrices(Buffer::Target::VertexAttributes);
	BallsModelMatrices.init();
	std::vector<glm::mat4> tmp_modelmatrices;
	tmp_modelmatrices.reserve(100);
	
	size_t row_ball_count = 10;
	size_t col_ball_count = 10;
	for(size_t i = 0; i < row_ball_count; ++i)
		for(size_t j = 0; j < col_ball_count; ++j)
		{
			tmp_modelmatrices.push_back(glm::scale(glm::translate(glm::mat4(1.0), glm::vec3(40.0 * (i - 0.5 * row_ball_count), 20.0, 40.0 * (j - 0.5 * col_ball_count))), glm::vec3(10.0)));
		}
		
	BallsModelMatrices.data(tmp_modelmatrices.data(), sizeof(glm::mat4) * tmp_modelmatrices.size(), Buffer::Usage::StaticDraw);

	Skybox Sky({"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posz.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negz.jpg"
	});
	
	for(auto& p : _tweakbars)
		_meshInstances[p.first].getMaterial().createAntTweakBar(p.second);
	
	resize_callback(window, _width, _height);
	
	////////////////////////////////////////////////////////////////////////////////////////////////
	// Main Loop
	
	while(!glfwWindowShouldClose(window))
	{	
		TimeManager::getInstance().update();
		_frameTime = TimeManager::getInstance().getRealDeltaTime();
		_frameRate = TimeManager::getInstance().getInstantFrameRate();
		if(!_paused)
			_time += _frameTime;
		
		if(controlCamera)
		{
			if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				MainCamera.moveForward(_frameTime);
				
			if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
				MainCamera.strafeLeft(_frameTime);
					
			if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				MainCamera.moveBackward(_frameTime);
					
			if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
				MainCamera.strafeRight(_frameTime);
				
			double mx = mouse_x, my = mouse_y;
			glfwGetCursorPos(window, &mouse_x, &mouse_y);
			MainCamera.look(glm::vec2(mouse_x - mx, my - mouse_y));
		}
		MainCamera.updateView();
		
		CameraStruct CamS = {MainCamera.getMatrix(), _projection};
		CameraBuffer.data(&CamS, sizeof(CameraStruct), Buffer::Usage::DynamicDraw);
		
		std::ostringstream oss;
		oss << _frameRate;
		glfwSetWindowTitle(window, ((std::string("OpenGL ToolBox Test - FPS: ") + oss.str()).c_str()));
		
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
		MainLights[0].setPosition(300.0f * glm::vec3(std::sin(_time * 0.5), 0.0, std::cos(_time * 0.5)) + glm::vec3(0.0, 800.0 , 0.0));
		MainLights[0].lookAt(glm::vec3(0.0, 250.0, 0.0));
		
		MainLights[1].setPosition(-300.0f * glm::vec3(std::sin(_time * 0.8), 0.0, std::cos(_time * 0.8)) + glm::vec3(0.0, 400.0 , 0.0));
		MainLights[1].lookAt(glm::vec3(0.0, 150.0, 0.0));
		
		MainLights[2].setPosition(200.0f * glm::vec3(std::sin(_time * 0.2), 0.0, std::cos(_time * 0.2)) + glm::vec3(0.0, 800.0 , 0.0));
		MainLights[2].lookAt(100.0f * glm::vec3(std::sin(_time * 0.2), 0.0, std::cos(_time * 0.2)) + glm::vec3(0.0, 200.0 , 0.0));
		
		for(size_t i = 0; i < LightCount; ++i)
		{
			MainLights[i].updateMatrices();
			LightStruct tmpLight = {glm::vec4(MainLights[i].getPosition(), 1.0),  MainLights[i].getColor(), MainLights[i].getBiasedMatrix()};
			LightBuffers[i].data(&tmpLight, sizeof(LightStruct), Buffer::Usage::DynamicDraw);
		}
		
		// Restore Viewport (binding the framebuffer modifies it - should I make the unbind call restore it ? How ?)
		
		Sky.draw(_projection, MainCamera.getMatrix());
			
		for(size_t i = 0; i < LightCount; ++i)
			MainLights[i].getShadowMap().bind(i + 2);
			
		for(auto& b : _meshInstances)
		{
			if(isVisible(_projection * MainCamera.getMatrix(), b.getMesh().getBoundingBox()))
			{
				b.draw();
			}
		}
		
		// TEMP: Instance draw
		
		Ball->getVAO().bind();
		BallsModelMatrices.bind();
		Ball->getMaterial().use();
		for(unsigned int i = 0; i < 4; ++i)
		{
			glEnableVertexAttribArray(3 + i);
			glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (const GLvoid*) (sizeof(float) * i * 4));
			glVertexAttribDivisor(3 + i, 1);
		}
		glDrawElementsInstanced(GL_TRIANGLES, Ball->getTriangles().size()*3, GL_UNSIGNED_INT, 0, tmp_modelmatrices.size());
		BallsModelMatrices.unbind();
		Ball->getVAO().unbind();
			
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

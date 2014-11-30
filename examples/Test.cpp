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
#include <SkyBox.hpp>
#include <Light.hpp>
#include <stb_image_write.hpp>

int		_width = 1366;
int		_height = 720;

glm::vec3 _resolution(_width, _height, 0.0);
glm::mat4 _projection;
glm::vec4 _mouse(0.0);

		
bool controlCamera = true;
double mouse_x, mouse_y;

float _time = 0.f;
float	_frameTime;
float	_frameRate;
	
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
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GL_TRUE);
			
		if(key == GLFW_KEY_R && action == GLFW_PRESS)
		{
			ResourcesManager::getInstance().reloadShaders();
		}
		
		if(key == GLFW_KEY_SPACE && action == GLFW_PRESS)
		{
			if(!controlCamera)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			} else {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
			controlCamera = !controlCamera;
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

int main(int argc, char* argv[])
{
	if (glfwInit() == false)
	{
		std::cerr << "Error: couldn't initialize GLFW." << std::endl;
		exit(EXIT_FAILURE);
	}
	glfwSetErrorCallback(error_callback);
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
			
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	CubeMap Tex;
	#define CUBEMAP_FOLDER "Vasa"
	Tex.load({"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posz.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negz.jpg"
	});

	VertexShader& VS = ResourcesManager::getInstance().getShader<VertexShader>("NormalMap_VS");
	VS.loadFromFile("src/GLSL/normalmap_vs.glsl");
	VS.compile();

	FragmentShader& FS = ResourcesManager::getInstance().getShader<FragmentShader>("NormalMap_FS");
	FS.loadFromFile("src/GLSL/normalmap_fs.glsl");
	FS.compile();

	GeometryShader& GS = ResourcesManager::getInstance().getShader<GeometryShader>("Cubemap_GS");
	GS.loadFromFile("src/GLSL/cubemap_gs.glsl");
	GS.compile();
	
	Program& NormalMap = ResourcesManager::getInstance().getProgram("NormalMap");
	NormalMap.attachShader(VS);
	NormalMap.attachShader(FS);
	NormalMap.link();
	
	Program& CubeNormalMap = ResourcesManager::getInstance().getProgram("CubeNormalMap");
	CubeNormalMap.attachShader(VS);
	CubeNormalMap.attachShader(GS);
	CubeNormalMap.attachShader(FS);
	CubeNormalMap.link();

	VertexShader& ReflectiveVS = ResourcesManager::getInstance().getShader<VertexShader>("Reflective_VS");
	ReflectiveVS.loadFromFile("src/GLSL/Reflective/vs.glsl");
	ReflectiveVS.compile();

	FragmentShader& ReflectiveFS = ResourcesManager::getInstance().getShader<FragmentShader>("Reflective_FS");
	ReflectiveFS.loadFromFile("src/GLSL/Reflective/fs.glsl");
	ReflectiveFS.compile();
	
	
	VertexShader& FullscreenTextureVS = ResourcesManager::getInstance().getShader<VertexShader>("FullscreenTexture_VS");
	FullscreenTextureVS.loadFromFile("src/GLSL/vs.glsl");
	FullscreenTextureVS.compile();
	
	FragmentShader& FullscreenTextureFS = ResourcesManager::getInstance().getShader<FragmentShader>("FullscreenTexture_FS");
	FullscreenTextureFS.loadFromFile("src/GLSL/FullscreenTexture.glsl");
	FullscreenTextureFS.compile();
	
	Program& FullscreenTexture = ResourcesManager::getInstance().getProgram("FullscreenTexture");
	FullscreenTexture.attachShader(FullscreenTextureVS);
	FullscreenTexture.attachShader(FullscreenTextureFS);
	FullscreenTexture.link();
	
	Program& Reflective = ResourcesManager::getInstance().getProgram("Reflective");
	Reflective.attachShader(ReflectiveVS);
	Reflective.attachShader(ReflectiveFS);
	Reflective.link();
	
	Camera MainCamera;
	Light MainLight;
	MainLight.init();
	MainLight.setPosition(glm::vec3(100.0, 300.0, 100.0));
	MainLight.lookAt(glm::vec3(0.0));
	
	glm::mat3 _normalMatrix;
	
	auto Glados = Mesh::load("in/3DModels/Glados/Glados.obj");
	std::vector<MeshInstance> _meshInstances;
	size_t meshNum = 0;
	std::array<std::string, 4> GladosTex {"caf160b2", "fa08434c", "09f5cc6b", "72c40a8a"};
	std::vector<Texture2D> GladosTextures;
	GladosTextures.resize(4);
	std::vector<Texture2D> GladosNormalMaps;
	GladosNormalMaps.resize(4);
	for(size_t i = 0; i < 4; ++i)
	{
		GladosTextures[i].load(std::string("in/3DModels/Glados/").append(GladosTex[i]).append(".jpg"));
		GladosNormalMaps[i].load(std::string("in/3DModels/Glados/").append(GladosTex[i]).append("_n.jpg"));
		GladosNormalMaps[i].set(Texture::MinFilter, GL_LINEAR);
	}
	
	for(Mesh* m : Glados)
	{
		m->getMaterial().setShadingProgram(NormalMap);
		m->getMaterial().setUniform("lightPosition", &MainLight.getPosition());
		m->getMaterial().setUniform("DepthMVP", &MainLight.getBiasedMatrix());
		m->getMaterial().setUniform("ModelViewMatrix", &MainCamera.getMatrix());
		m->getMaterial().setUniform("ProjectionMatrix", &_projection);
		m->getMaterial().setUniform("NormalMatrix", &_normalMatrix);
		m->getMaterial().setUniform("Texture", GladosTextures[meshNum]);
		m->getMaterial().setUniform("ShadowMap", MainLight.getShadowMap());
		m->getMaterial().setUniform("NormalMap", GladosNormalMaps[meshNum]);
		m->getMaterial().setUniform("Ns", 90.0f);
		m->getMaterial().setUniform("Ka", glm::vec4(0.0f, 0.0f, 0.0f, 1.f));
		m->getMaterial().setUniform("Ks", glm::vec4(1.0f, 1.0f, 1.0f, 1.f));
		m->getMaterial().createAntTweakBar("Material " + StringConversion::to_string(meshNum));
		
		m->createVAO();
		
		_meshInstances.push_back(MeshInstance(*m));
		++meshNum;
	}
	
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
	Plane.getMaterial().setUniform("lightPosition", &MainLight.getPosition());
	Plane.getMaterial().setUniform("DepthMVP", &MainLight.getBiasedMatrix());
	Plane.getMaterial().setUniform("ModelViewMatrix", &MainCamera.getMatrix());
	Plane.getMaterial().setUniform("ProjectionMatrix", &_projection);
	Plane.getMaterial().setUniform("NormalMatrix", &_normalMatrix);
	Plane.getMaterial().setUniform("Texture", GroundTexture);
	Plane.getMaterial().setUniform("ShadowMap", MainLight.getShadowMap());
	Plane.getMaterial().setUniform("NormalMap", GroundNormalMap);
	Plane.getMaterial().setUniform("Ns", 90.0f);
	Plane.getMaterial().setUniform("Ka", glm::vec4(0.1f, 0.1f, 0.1f, 1.f));
	Plane.getMaterial().setUniform("Ks", glm::vec4(1.0f, 1.0f, 1.0f, 1.f));
	Plane.getMaterial().createAntTweakBar("Plane");
	
	float inRad = 60.0 * pi()/180.f;
	_projection = glm::perspective(inRad, (float) _width/_height, 0.1f, 1000.0f);
	
	Framebuffer<CubeMap, 1> CubeFramebufferTest;
	CubeFramebufferTest.init();
	
	Camera SphereCamera;
	SphereCamera.setPosition(glm::vec3(0.0, 75.0, 0.0));
	SphereCamera.setDirection(glm::vec3(0.0, 0.0, 1.0));
	SphereCamera.updateView();
	auto Sphere = Mesh::load("in/3DModels/Robot/Robot.obj");
	meshNum = 0;
	for(Mesh* m : Sphere)
	{
		//for(auto& v : m->getVertices())
		//	v.position = 50.0f * v.position + glm::vec3(0.0, 75.0, 0.0);
			
		m->getMaterial().setShadingProgram(Reflective);
		m->getMaterial().setUniform("cameraPosition", &MainCamera.getPosition());
		m->getMaterial().setUniform("lightPosition", &MainLight.getPosition());
		m->getMaterial().setUniform("DepthMVP", &MainLight.getBiasedMatrix());
		m->getMaterial().setUniform("ModelViewMatrix", &MainCamera.getMatrix());
		m->getMaterial().setUniform("ProjectionMatrix", &_projection);
		m->getMaterial().setUniform("NormalMatrix", &_normalMatrix);
		m->getMaterial().setUniform("EnvMap", CubeFramebufferTest.getColor());
		m->getMaterial().setUniform("ShadowMap", MainLight.getShadowMap());
		m->getMaterial().setUniform("Ns", 90.0f);
		m->getMaterial().setUniform("Ka", glm::vec4(0.1f, 0.1f, 0.1f, 1.f));
		m->getMaterial().setUniform("Ks", glm::vec4(1.0f, 1.0f, 1.0f, 1.f));
		m->getMaterial().setUniform("diffuse", glm::vec4(0.1f, 0.1f, 0.1f, 1.f));
		m->getMaterial().createAntTweakBar("Reflective " + StringConversion::to_string(meshNum));
		
		//m->computeNormals();
		m->createVAO();
		
		_meshInstances.push_back(MeshInstance(*m));
		++meshNum;
	}

	Skybox Sky({"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negx.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negy.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/posz.jpg",
				"in/Textures/cubemaps/" CUBEMAP_FOLDER "/negz.jpg"
	});
	
	while(!glfwWindowShouldClose(window))
	{	
		TimeManager::getInstance().update();
		_frameTime = TimeManager::getInstance().getRealDeltaTime();
		_time += _frameTime;
		_frameRate = TimeManager::getInstance().getInstantFrameRate();
		
		std::ostringstream oss;
		oss << _frameRate;
		glfwSetWindowTitle(window, ((std::string("OpenGL ToolBox Test - FPS: ") + oss.str()).c_str()));
		
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
		MainLight.setPosition(300.0f * glm::vec3(std::sin(_time * 0.5), 0.0, std::cos(_time * 0.5)) + glm::vec3(0.0, 800.0 , 0.0));
		MainLight.lookAt(glm::vec3(0.0, 250.0, 0.0));
		
		MainLight.updateMatrices();
		MainLight.bind();
		for(Mesh* m : Glados)
		{
			m->draw();
		}
		Plane.draw();
		for(Mesh* m : Sphere)
		{
			m->draw();
		}
		MainLight.unbind();
		glCullFace(GL_BACK);
	
		_normalMatrix = glm::mat3(glm::transpose(glm::inverse(MainCamera.getMatrix())));
		
		// Render to CubeMap Test ! :)
		
		CubeFramebufferTest.bind(GL_DRAW_FRAMEBUFFER);
		
		Sky.cubedraw();
		glm::mat4 Perspective = glm::perspective<float>((float) pi() / 2.0f, 1.f, 0.5f, 1000.0f);
		for(Mesh* m : Glados)
		{
			m->getMaterial().setShadingProgram(CubeNormalMap);
			m->getMaterial().setUniform("ProjectionMatrix", &Perspective);
			m->getMaterial().setUniform("ModelViewMatrix", &SphereCamera.getMatrix());
			m->getMaterial().use();
			m->draw();
		}
		Plane.getMaterial().setShadingProgram(CubeNormalMap);
		Plane.getMaterial().setUniform("ProjectionMatrix", &Perspective);
		Plane.getMaterial().setUniform("ModelViewMatrix", &SphereCamera.getMatrix());
		Plane.getMaterial().use();
		Plane.draw();
		CubeFramebufferTest.unbind();
		
		// Save to disk (Debug)
		static bool done = false;
		if(!done) 
		{
			CubeFramebufferTest.getColor().dump("out/CubeFramebufferTest/cube_");
			done = true;
		}
		
		//MainCamera.setPosition(500.0f * glm::vec3(std::sin(_time * 0.1), 0.0, std::cos(_time * 0.1)) + glm::vec3(0.0, 250.0 , 0.0));
		//MainCamera.lookAt(glm::vec3(0.0, 250.0, 0.0));
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
	
		_normalMatrix = glm::mat3(glm::transpose(glm::inverse(MainCamera.getMatrix())));
		
		// Restore Viewport (binding the framebuffer modifies it - should I make the unbind call restore it ? How ?)
		glViewport(0, 0, _width, _height);
		Sky.draw(_projection, MainCamera.getMatrix());
		
		for(Mesh* m : Glados)
		{
			m->getMaterial().setShadingProgram(NormalMap);
			m->getMaterial().setUniform("ProjectionMatrix", &_projection);
			m->getMaterial().setUniform("ModelViewMatrix", &MainCamera.getMatrix());
			m->getMaterial().use();
			m->draw();
		}
		Plane.getMaterial().setShadingProgram(NormalMap);
		Plane.getMaterial().setUniform("ProjectionMatrix", &_projection);
		Plane.getMaterial().setUniform("ModelViewMatrix", &MainCamera.getMatrix());
		Plane.getMaterial().use();
		Plane.draw();
		
		for(Mesh* m : Sphere)
		{
			m->getMaterial().use();
			m->draw();
		}
		
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

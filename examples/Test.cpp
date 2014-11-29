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

#include <Core/TimeManager.hpp>
#include <Core/ResourcesManager.hpp>
#include <Tools/StringConversion.hpp>
#include <Graphics/Material.hpp>
#include <Graphics/Texture2D.hpp>
#include <Graphics/Texture3D.hpp>
#include <Graphics/Framebuffer.hpp>
#include <Graphics/Buffer.hpp>
#include <Graphics/MeshInstance.hpp>
#include <MathTools.hpp>
#include <Camera.hpp>
#include <Light.hpp>
#include <stb_image_write.hpp>

int		_width = 1366;
int		_height = 720;

glm::vec3 _resolution(_width, _height, 0.0);
glm::mat4 _projection;
glm::vec4 _mouse(0.0);

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
	
	Program& NormalMap = ResourcesManager::getInstance().getProgram("NormalMap");
	NormalMap.attachShader(VS);
	NormalMap.attachShader(FS);
	NormalMap.link();
	
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
		m->getMaterial().setUniform("minDiffuse", 0.05f);
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
	Plane.getMaterial().setUniform("Ka", glm::vec4(0.0f, 0.0f, 0.0f, 1.f));
	Plane.getMaterial().setUniform("Ks", glm::vec4(1.0f, 1.0f, 1.0f, 1.f));
	Plane.getMaterial().setUniform("minDiffuse", 0.05f);
	Plane.getMaterial().createAntTweakBar("Plane");
	
	float inRad = 60.0 * pi()/180.f;
	_projection = glm::perspective(inRad, (float) _width/_height, 0.1f, 1000.0f);

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
		
		MainCamera.setPosition(500.0f * glm::vec3(std::sin(_time * 0.1), 0.0, std::cos(_time * 0.1)) + glm::vec3(0.0, 250.0 , 0.0));
		MainCamera.lookAt(glm::vec3(0.0, 250.0, 0.0));
		MainCamera.updateView();
	
		_normalMatrix = glm::mat3(glm::transpose(glm::inverse(MainCamera.getMatrix())));
	
		MainLight.setPosition(300.0f * glm::vec3(std::sin(_time * 0.5), 0.0, std::cos(_time * 0.5)) + glm::vec3(0.0, 800.0 , 0.0));
		MainLight.lookAt(glm::vec3(0.0, 250.0, 0.0));
		
		MainLight.updateMatrices();
		MainLight.bind();
		for(Mesh* m : Glados)
		{
			m->draw();
		}
		Plane.draw();
		MainLight.unbind();
	
		// Restore Viewport (binding the framebuffer modifies it - should I make the unbind call restore it ? How ?)
		glViewport(0, 0, _width, _height);
		glCullFace(GL_BACK);
		for(Mesh* m : Glados)
		{
			m->getMaterial().use();
			m->draw();
		}
		Plane.getMaterial().use();
		Plane.draw();
		
		TwDraw();
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	TwTerminate();
	
	glfwDestroyWindow(window);
}

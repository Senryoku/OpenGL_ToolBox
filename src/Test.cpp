#include <cstdlib>
#include <ctime>
#include <sstream>
#include <map>
#include <random>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp> // glm::translate
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
#include <stb_image_write.hpp>

int		_width = 1366;
int		_height = 720;

glm::vec3 _resolution(_width, _height, 0.0);
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
	
	glOrtho(-1.0f, -1.0f, 1.0f, 1.0f, 0.1f, 10000.0f);
	
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
			
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	Texture3D Tex;
	Tex.create(nullptr, 128, 128, 128, 1);
	
	VertexShader& RayTracerVS = ResourcesManager::getInstance().getShader<VertexShader>("RayTracerVS");
	RayTracerVS.loadFromFile("src/GLSL/vs.glsl");
	RayTracerVS.compile();

	FragmentShader& FullscreenTextureFS = ResourcesManager::getInstance().getShader<FragmentShader>("FullscreenTextureFS");
	FullscreenTextureFS.loadFromFile("src/GLSL/FullscreenTexture.glsl");
	FullscreenTextureFS.compile();

	FragmentShader& ShaderToyFS = ResourcesManager::getInstance().getShader<FragmentShader>("ShaderToyFS");
	ShaderToyFS.loadFromFile("src/GLSL/Texture3DTest.glsl");
	ShaderToyFS.compile();
	
	Program& ShaderToy = ResourcesManager::getInstance().getProgram("Texture3D Test");
	ShaderToy.attachShader(RayTracerVS);
	ShaderToy.attachShader(ShaderToyFS);
	ShaderToy.link();
	
	Material Mat(ShaderToy);
	Mat.setUniform("iGlobalTime", &_time);
	Mat.setUniform("iResolution", glm::vec3(_width, _height, 0.0));
	Mat.setUniform("iMouse", &_mouse);
	Mat.setUniform("iChannel0", Tex);
	Mat.createAntTweakBar("Material");
	
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
		
		Mat.use();
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		Mat.useNone();
		
		TwDraw();
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	TwTerminate();
	
	glfwDestroyWindow(window);
}

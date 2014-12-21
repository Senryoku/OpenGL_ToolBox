#include "Shader.hpp"

Shader::~Shader()
{
	glDeleteShader(_handle);
}

void Shader::init()
{
	if(_handle != 0)
		return;
		
	_handle = glCreateShader(this->getType());
	if(_handle == 0)
	{
		std::cerr << __PRETTY_FUNCTION__ << " : Error glCreateShader(GL_FRAGMENT_SHADER)" << std::endl;
	}
	
	_compiled = false;
}


void Shader::loadFromFile(const std::string& path)
{
	this->init();
	
	//std::cout << "loadFromFile " << path << std::endl;
	std::ifstream Src(path);
	if(!Src.good())
	{
		std::cerr << __FUNCTION__ << " : Error opening shader " << path << std::endl;
		return;
	}
	GLint lengths[1024];
	GLchar* cSrc[1024];
	int l = 0;
	while(Src.good()) 
	{
		cSrc[l] = new GLchar[1024];
		Src.getline(cSrc[l], 1024);
		lengths[l] = Src.gcount();
		if(lengths[l] > 0)
		{
			cSrc[l][lengths[l] - 1] = '\n';
			cSrc[l][lengths[l]] = '\0';
			//std::cout << "Read (" << l << ", " << lengths[l] << ") : " << cSrc[l] << std::endl;
			++l;
		} else delete cSrc[l];
	}
	Src.close();
	glShaderSource(_handle, l, (const GLchar**) cSrc, lengths);
	for(int i = 0; i < l; ++i)
		delete cSrc[i];
		
	_srcPath = path;
	//std::cout << "... Done. " << std::endl;
}
	
void Shader::reload()
{
	assert(_srcPath != "");
	if(_srcPath == "")
		return;
	loadFromFile(_srcPath);
}

void Shader::compile()
{
	assert(_handle != 0);
    glCompileShader(_handle);
	
	// Check results
    int rvalue;
    glGetShaderiv(_handle, GL_COMPILE_STATUS, &rvalue);
    if (!rvalue)
	{
        std::cerr << __FUNCTION__ << " : Error while compiling shader " << _srcPath << std::endl;
        GLchar log[10240];
        GLsizei length;
        glGetShaderInfoLog(_handle, 10239, &length, log);
        std::cerr << "Compiler log " << log << std::endl;
		_compiled = false;
    } else {
		_compiled = true;
	}
}

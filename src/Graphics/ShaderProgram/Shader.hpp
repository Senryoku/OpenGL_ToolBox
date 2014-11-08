#pragma once

#include <iostream>
#include <fstream>
#include <string>
#define GLEW_STATIC
#include <GL/glew.h>

#include <OpenGLObject.hpp>

class Shader : public OpenGLObject
{
	public:
	Shader() =default;
	virtual ~Shader();
	
	void loadFromFile(const std::string& path);
	void reload();
	void setSource(const std::string& src);
	void compile();
	
	void compute(GLint x, GLint y = 1, GLint z = 1);
	
	inline bool isValid() const { return _compiled; }
	
	protected:
	std::string	_srcPath = "";
	bool				_compiled = false;
	
	virtual void initOGL() =0;
};


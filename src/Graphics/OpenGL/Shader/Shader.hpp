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
	
	inline bool isValid() const override;
	
	protected:
	std::string			_srcPath = "";
	bool				_compiled = false;
	
	virtual void init() =0;
};

inline bool Shader::isValid() const 
{
	return OpenGLObject::isValid() && _compiled;
}

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#define GLEW_STATIC
#include <GL/glew.h>

#include <OpenGLObject.hpp>

/**
 * Base class for all types of shaders
**/
class Shader : public OpenGLObject
{
public:
	Shader() =default;
	virtual ~Shader();
	
	void loadFromFile(const std::string& path);
	
	/**
	 * Reloads shader from file.
	 * (Should be used only after a successfull call to loadFromFile).
	 * @see loadFromFile
	**/
	void reload();

	void compile();
	
	/**
	 * @return True if properly initialized and compiled, false otherwise.
	**/
	inline bool isValid() const override;
	
protected:
	std::string			_srcPath = "";
	bool				_compiled = false;
	
	/**
	 * Initialize corresponding OpenGLObject
	**/
	void init();
	
	virtual GLenum getType() const =0;
};

inline bool Shader::isValid() const 
{
	return OpenGLObject::isValid() && _compiled;
}

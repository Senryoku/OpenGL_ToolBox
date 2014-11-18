#pragma once
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>
#include <ComputeShader.hpp>
#include <UniformBinding.hpp>

class Program : public OpenGLObject
{
public:
	Program();
	~Program();
	
	void attachShader(GLint shader);
	void attachShader(Shader& shader);
	void attachShader(ComputeShader& cshader);
	void link();
	void use();
	inline bool isValid() const { return _linked; }
	
	GLint getUniformLocation(const std::string& name) const;
	
	template<typename T>
	inline void setUniform(const std::string& name, const T& value)
	{
		::setUniform(getName(), getUniformLocation(name), value);
	}
	
	static void useNone();
	
private:
	bool	_linked = false;
};


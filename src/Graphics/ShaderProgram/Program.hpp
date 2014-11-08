#pragma once
#define GLEW_STATIC
#include <GL/glew.h>

#include <Shader.hpp>
#include <ComputeShader.hpp>

class Program
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
	
	inline const GLint& getID() const { return _program; }
	
	GLint getUniformLocation(const std::string& name) const;
	
	template<typename T>
	inline void setUniform(const std::string& name, const T& value)
	{
		
	}
	
	static void useNone();
	
	private:
	GLint	_program;
	bool	_linked = false;
};


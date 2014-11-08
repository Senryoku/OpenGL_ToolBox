#pragma once

#include <string>
#include <vector>
#include <iostream>

#define GLEW_STATIC
#include <GL/glew.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include <AllShader.hpp>
#include <Texture.hpp>
#include <CubeMap.hpp>

class Material
{
public:
	template<typename T>
	class Uniform
	{
	public:
		Uniform() =default;
		Uniform(const std::string& N, GLuint L, T& V) :
			_name(N),
			_location(L),
			_value(V)
		{ }
		
		inline const std::string& getName() const { return _name; }
		inline GLuint getLocation() const { return _location; }
		inline const T& getValue() const { return _value; }
		
		inline T& getRefToValue() { return _value; }
		
		inline void setValue(const T& val) { _value = val; }
		inline void setLocation(const GLuint val) { _location = val; }
	private:
		std::string	_name;
		GLuint			_location;
		T					_value;
	};

	//	Constructors
	Material() =default;
	
	Material(Program& P) :
		_shadingProgram(&P)
	{
	}
	
	Material(const Material&) =default;
	
	virtual ~Material() =default;

	//	Getters/Setters
	Program& getShadingProgram() { return *_shadingProgram; }
	void setShadingProgram(Program& P) { _shadingProgram = &P; updateLocations(); }
	
	// Scalars
	void setAttribute(const std::string& Name, const float& Value)
	{ setAttribute<float>(_uniform1f, Name, Value); }
	void setAttribute(const std::string& Name, const int& Value)
	{ setAttribute<int>(_uniform1i, Name, Value); }
	void setAttribute(const std::string& Name, const unsigned int& Value)
	{ setAttribute<unsigned int>(_uniform1ui, Name, Value); }

	// Vectors
	void setAttribute(const std::string& Name, const glm::vec2& Value)
	{ setAttribute<glm::vec2>(_uniform2fv, Name, Value); }
	void setAttribute(const std::string& Name, const glm::vec3& Value)
	{ setAttribute<glm::vec3>(_uniform3fv, Name, Value); }
	void setAttribute(const std::string& Name, const glm::vec4& Value)
	{ setAttribute<glm::vec4>(_uniform4fv, Name, Value); }
	
	// Matrices
	void setAttribute(const std::string& Name, const glm::mat2& Value)
	{ setAttribute<glm::mat2>(_uniformMatrix2fv, Name, Value); }
	void setAttribute(const std::string& Name, const glm::mat3& Value)
	{ setAttribute<glm::mat3>(_uniformMatrix3fv, Name, Value); }
	void setAttribute(const std::string& Name, const glm::mat4& Value)
	{ setAttribute<glm::mat4>(_uniformMatrix4fv, Name, Value); }
	
	///////////////////////////////////////////////////////////////////////
	// Referenced Attributes
	//   These are not copied locally but fetched from source each frame
	//   as we only hold a reference to them.
	
	// Scalars
	void setAttributeRef(const std::string& Name, float& Value)
	{ setAttribute<float*>(_refUniform1f, Name, &Value); }
	void setAttributeRef(const std::string& Name, int& Value)
	{ setAttribute<int*>(_refUniform1i, Name, &Value); }
	void setAttributeRef(const std::string& Name, unsigned int& Value)
	{ setAttribute<unsigned int*>(_refUniform1ui, Name, &Value); }

	// Vectors
	void setAttributeRef(const std::string& Name, glm::vec2& Value)
	{ setAttribute<float*>(_refUniform2fv, Name, glm::value_ptr(Value)); }
	void setAttributeRef(const std::string& Name, glm::vec3& Value)
	{ setAttribute<float*>(_refUniform3fv, Name, glm::value_ptr(Value)); }
	void setAttributeRef(const std::string& Name, glm::vec4& Value)
	{ setAttribute<float*>(_refUniform4fv, Name, glm::value_ptr(Value)); }
	
	// Matrices
	void setAttributeRef(const std::string& Name, glm::mat2& Value)
	{ setAttribute<float*>(_refUniformMatrix2fv, Name, glm::value_ptr(Value)); }
	void setAttributeRef(const std::string& Name, glm::mat3& Value)
	{ setAttribute<float*>(_refUniformMatrix3fv, Name, glm::value_ptr(Value)); }
	void setAttributeRef(const std::string& Name, glm::mat4& Value)
	{ setAttribute<float*>(_refUniformMatrix4fv, Name, glm::value_ptr(Value)); }
	
	// Samplers
	void setAttributeRef(const std::string& Name, Texture& Value)
	{ setAttribute<Texture*>(_refUniformSampler2D, Name, &Value); }
	void setAttributeRef(const std::string& Name, CubeMap& Value)
	{ setAttribute<CubeMap*>(_refUniformCubemap, Name, &Value); }
	///////////////////////////////////////////////////////////////////////

	//	Methods
	void virtual use() const
	{
		if(_shadingProgram != nullptr)
			_shadingProgram->use();
			
		bind();
	}
	
	void virtual bind() const;
	void updateLocations();
	
	// Debug Helper
	void createAntTweakBar(const std::string& Name);

private:
	Program*	_shadingProgram = nullptr;
	
	// Attributes
	std::vector<Uniform<float>>			_uniform1f;
	std::vector<Uniform<int>>				_uniform1i;
	std::vector<Uniform<unsigned int>>	_uniform1ui;
	
	std::vector<Uniform<glm::vec2>>		_uniform2fv;
	std::vector<Uniform<glm::vec3>>		_uniform3fv;
	std::vector<Uniform<glm::vec4>>		_uniform4fv;
	
	std::vector<Uniform<glm::mat2>>		_uniformMatrix2fv;
	std::vector<Uniform<glm::mat3>>		_uniformMatrix3fv;
	std::vector<Uniform<glm::mat4>>		_uniformMatrix4fv;
	
	// Referenced Attributes
	std::vector<Uniform<float*>>				_refUniform1f;
	std::vector<Uniform<int*>>					_refUniform1i;
	std::vector<Uniform<unsigned int*>>	_refUniform1ui;
	
	std::vector<Uniform<float*>>		_refUniform2fv;
	std::vector<Uniform<float*>>		_refUniform3fv;
	std::vector<Uniform<float*>>		_refUniform4fv;
	
	std::vector<Uniform<float*>>		_refUniformMatrix2fv;
	std::vector<Uniform<float*>>		_refUniformMatrix3fv;
	std::vector<Uniform<float*>>		_refUniformMatrix4fv;
	
	std::vector<Uniform<Texture*>>		_refUniformSampler2D;
	std::vector<Uniform<CubeMap*>>		_refUniformCubemap;
	
	template<typename T>
	void setAttribute(typename std::vector<Uniform<T>>& Container, const std::string& Name, T Value)
	{
		GLuint Location = glGetUniformLocation(_shadingProgram->getID(), Name.c_str());
		
		if(Location >= 0)
		{
			for(Uniform<T>& U : Container)
			{
				if(U.getName() == Name)
				{
					U.setValue(Value);
					return;
				}
			}
			Container.push_back(Uniform<T>{Name, Location, Value});
		} else 
			std::cerr << "Material: Uniform " + Name + " not found." << std::endl;
	}
	
	template<typename T>
	void updateLocations(typename std::vector<Uniform<T>>& Container)
	{
		for(Uniform<T>& U : Container)
		{
			U.setLocation(glGetUniformLocation(_shadingProgram->getID(), U.getName().c_str()));
		}
	}
};

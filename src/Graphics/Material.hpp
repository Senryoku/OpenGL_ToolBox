#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>


#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

/*
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr
*/
#include <AllShader.hpp>
#include <Uniform.hpp>
#include <Texture2D.hpp>

class Material
{
public:
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
	
	////////////////////////////////////////////////////////////////
	// Special cases for textures
	
	void setUniform(const std::string& name, const Texture2D& value)
	{
		setUniform(name, static_cast<const Texture&>(value));
	}
	
	void setUniform(const std::string& name, const CubeMap& value)
	{
		setUniform(name, static_cast<const Texture&>(value));
	}
	
	void setUniform(const std::string& name, const Texture& value)
	{
		GLint Location = getLocation(name);
		
		if(Location >= 0)
		{
			for(auto& U : _uniforms)
			{
				if(U.get()->getName() == name)
				{
					static_cast<Uniform<Texture>*>(U.get())->setValue(value);
					return;
				}
			}
			_uniforms.push_back(std::unique_ptr<GenericUniform>(new Uniform<Texture>(name, Location, _textureCount, value)));
			++_textureCount;
		} else {
			std::cerr << "Material: Uniform " + name + " not found." << std::endl;
		}
	}
	
	////////////////////////////////////////////////////////////////
	// Generic Uniform setting
	
	template<typename T>
	void setUniform(const std::string& name, const T& value)
	{
		GLint Location = getLocation(name);
		
		if(Location >= 0)
		{
			for(auto& U : _uniforms)
			{
				if(U.get()->getName() == name)
				{
					static_cast<Uniform<T>*>(U.get())->setValue(value);
					return;
				}
			}
			_uniforms.push_back(std::unique_ptr<GenericUniform>(new Uniform<T>(name, Location, value)));
		} else {
			std::cerr << "Material: Uniform " + name + " not found." << std::endl;
		}
	}

	void use() const
	{
		if(_shadingProgram != nullptr)
			_shadingProgram->use();
			
		bind();
	}
	
	void bind() const;
		
#ifndef NO_ANTTWEAKBAR
	void createAntTweakBar(const std::string& Name);
#endif // NO_ANTTWEAKBAR
	
	void updateLocations();

private:
	Program*	_shadingProgram = nullptr;
	
	std::vector<std::unique_ptr<GenericUniform>>		_uniforms;
	GLuint 	_textureCount = 0;
	
	GLint getLocation(const std::string& name);
};

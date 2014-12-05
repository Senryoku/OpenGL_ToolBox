#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <memory>


#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <AllShader.hpp>
#include <Uniform.hpp>
#include <Texture2D.hpp>
#include <Texture3D.hpp>

/**
 * Material
 * Association of a Shader program and a set of Uniforms.
 * @see Program
 * @see Uniform
**/
class Material
{
public:
	//	Constructors
	Material() =default;
	
	Material(const Program& P) :
		_shadingProgram(&P)
	{
	}
	
	/**
	 * Copy constructor
	**/
	Material(const Material&);
	
	/**
	 * Default destructor
	**/
	~Material() =default;
	
	Material& operator=(const Material&);

	//	Getters/Setters
	const Program& getShadingProgram() { return *_shadingProgram; }
	void setShadingProgram(Program& P) { _shadingProgram = &P; updateLocations(); }
	
	////////////////////////////////////////////////////////////////
	// Special cases for textures
	
	void setUniform(const std::string& name, const Texture3D& value)
	{
		setUniform(name, static_cast<const Texture&>(value));
	}
	
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
					//static_cast<Uniform<T>*>(U.get())->setValue(value);
					// In case the types differs... We better create a shinny new one.
					U.reset(new Uniform<T>(name, Location, value));
					
					return;
				}
			}
			_uniforms.push_back(std::unique_ptr<GenericUniform>(new Uniform<T>(name, Location, value)));
		} else {
			std::cerr << "Material: Uniform " + name + " not found." << std::endl;
		}
	}

	inline void use() const
	{
		if(_shadingProgram != nullptr)
			_shadingProgram->use();
			
		bind();
	}
	
	inline void useNone() const
	{
		unbind();
		Program::useNone();
	}
	
	void bind() const;
	
	void unbind() const;
		
#ifndef NO_ANTTWEAKBAR
	void createAntTweakBar(const std::string& Name);
#endif // NO_ANTTWEAKBAR
	
	void updateLocations();

private:
	const Program*	_shadingProgram = nullptr;
	
	std::vector<std::unique_ptr<GenericUniform>>		_uniforms;
	GLuint 												_textureCount = 0;
	
	GLint getLocation(const std::string& name) const;
};

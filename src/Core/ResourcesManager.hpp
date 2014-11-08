#pragma once

#include <map>

#include <Tools/Singleton.hpp>
#include <Graphics/Texture.hpp>
#include <Graphics/CubeMap.hpp>
#include <Graphics/AllShader.hpp>

class ResourcesManager : public Singleton<ResourcesManager>
{
public:
	~ResourcesManager()
	{
		for(auto it : _shaders)
			delete it.second;
	}
	
	inline Shader& getShader(const std::string& Name) throw(std::runtime_error)
	{
		auto it = _shaders.find(Name);
		if(it != _shaders.end())
		{
			return *it->second;
		} else {
			throw std::runtime_error(Name + " shader not found. Use a specialized version of getShader or make sure you referenced it to the ResourcesManager before calling getShader.");
		}
	}
	
	template<typename ShaderType>
	inline ShaderType& getShader(const std::string& Name) throw(std::runtime_error)
	{
		auto it = _shaders.find(Name);
		if(it != _shaders.end())
		{
			return *static_cast<ShaderType*>(it->second);
		} else {
			auto newShader = new ShaderType();
			_shaders[Name] = newShader;
			return *static_cast<ShaderType*>(newShader);
		}
	} 
	
	inline Texture& getTexture(const std::string& Name) throw(std::runtime_error)
	{
		auto it = _textures.find(Name);
		if(it != _textures.end())
		{
			return *it->second;
		} else {
			throw std::runtime_error(Name + " texture not found. Use a specialized version of getTexture or make sure you referenced it to the ResourcesManager before calling getTexture.");
		}
	}
	
	template<typename T>
	inline T& getTexture(const std::string& Name) throw(std::runtime_error)
	{
		auto it = _textures.find(Name);
		if(it != _textures.end())
		{
			return *static_cast<T*>(it->second);
		} else {
			auto newTexture = new T();
			_textures[Name] = newTexture;
			return *static_cast<T*>(newTexture);
		}
	} 
	
	inline Program& getProgram(const std::string& Name)
	{ return _programs[Name]; }
	
	inline void reloadShaders()
	{
		for(auto& S : _shaders)
		{
			S.second->reload();
			S.second->compile();
		}
		
		for(auto& P : _programs)
			P.second.link();
	}

private:
	std::map<std::string, Texture*>	_textures;

	std::map<std::string, Shader*>	_shaders;
	std::map<std::string, Program>	_programs;
};

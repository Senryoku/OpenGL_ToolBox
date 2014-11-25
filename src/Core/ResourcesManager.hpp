#pragma once

#include <map>
#include <memory>

#include <Tools/Singleton.hpp>
#include <Graphics/Texture.hpp>
#include <Graphics/CubeMap.hpp>
#include <Graphics/AllShader.hpp>

class ResourcesManager : public Singleton<ResourcesManager>
{
public:
	ResourcesManager() =default;
	~ResourcesManager() =default;
	
	inline Shader& getShader(const std::string& Name) throw(std::runtime_error)
	{
		auto it = _shaders.find(Name);
		if(it != _shaders.end())
		{
			return *it->second.get();
		} else {
			throw std::runtime_error(Name + " shader not found. Use a specialized version of getShader or make sure you referenced it to the ResourcesManager before calling getShader.");
		}
	}
	
	template<typename ShaderType>
	inline ShaderType& getShader(const std::string& Name)
	{
		auto it = _shaders.find(Name);
		if(it != _shaders.end())
		{
			return *static_cast<ShaderType*>(it->second.get());
		} else {
			auto newShader = new ShaderType();
			_shaders[Name].reset(newShader);
			return *newShader;
		}
	} 
	
	inline Texture& getTexture(const std::string& Name) throw(std::runtime_error)
	{
		auto it = _textures.find(Name);
		if(it != _textures.end())
		{
			return *it->second.get();
		} else {
			throw std::runtime_error(Name + " texture not found. Use a specialized version of getTexture or make sure you referenced it to the ResourcesManager before calling getTexture.");
		}
	}
	
	template<typename T>
	inline T& getTexture(const std::string& Name)
	{
		auto it = _textures.find(Name);
		if(it != _textures.end())
		{
			return *static_cast<T*>(it->second.get());
		} else {
			auto newTexture = new T();
			_textures[Name].reset(newTexture);
			return *newTexture;
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
	std::map<std::string, std::unique_ptr<Texture>>	_textures;
	std::map<std::string, std::unique_ptr<Shader>>	_shaders;
	
	std::map<std::string, Program>							_programs;
};

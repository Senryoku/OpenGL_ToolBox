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
	
	Shader& getShader(const std::string& Name) throw(std::runtime_error);
	
	template<typename ShaderType>
	inline ShaderType& getShader(const std::string& Name);
	
	Texture& getTexture(const std::string& Name) throw(std::runtime_error);
	
	template<typename T>
	inline T& getTexture(const std::string& Name);
	
	Program& getProgram(const std::string& Name);
	
	void reloadShaders();

private:
	std::map<std::string, std::unique_ptr<Texture>>	_textures;
	std::map<std::string, std::unique_ptr<Shader>>	_shaders;
	
	std::map<std::string, Program>							_programs;
};

#include <ResourcesManager.tcc>

#include <ResourcesManager.hpp>

Shader& ResourcesManager::getShader(const std::string& Name) throw(std::runtime_error)
{
	auto it = _shaders.find(Name);
	if(it != _shaders.end())
	{
		return *it->second.get();
	} else {
		throw std::runtime_error(Name + " shader not found. Use a specialized version of getShader or make sure you referenced it to the ResourcesManager before calling getShader.");
	}
}

Texture& ResourcesManager::getTexture(const std::string& Name) throw(std::runtime_error)
{
	auto it = _textures.find(Name);
	if(it != _textures.end())
	{
		return *it->second.get();
	} else {
		throw std::runtime_error(Name + " texture not found. Use a specialized version of getTexture or make sure you referenced it to the ResourcesManager before calling getTexture.");
	}
}

Program& ResourcesManager::getProgram(const std::string& Name)
{ 
	return _programs[Name];
}

void ResourcesManager::reloadShaders()
{
	for(auto& S : _shaders)
	{
		S.second->reload();
		S.second->compile();
	}
	
	for(auto& P : _programs)
		P.second.link();
}
#pragma once

template<typename ShaderType>
inline ShaderType& ResourcesManager::getShader(const std::string& Name)
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

template<typename T>
inline T& ResourcesManager::getTexture(const std::string& Name)
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

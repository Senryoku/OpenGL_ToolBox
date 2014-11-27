#include <Material.hpp>

Material::Material(const Material& m)
{
	for(const auto& u : m._uniforms)
		_uniforms.push_back(std::unique_ptr<GenericUniform>(u.get()->clone()));
}

void Material::bind() const
{	
	for(const auto& U : _uniforms)
		U.get()->bind(_shadingProgram->getName());
}

void Material::unbind() const
{	
	for(auto& U : _uniforms)
		U.get()->unbind(_shadingProgram->getName());
}

void Material::updateLocations()
{
	for(auto& U : _uniforms)
	{
		U.get()->setLocation(getLocation(U.get()->getName()));
	}
}
	
GLint Material::getLocation(const std::string& name)
{
	return _shadingProgram->getUniformLocation(name);
}
	
#ifndef NO_ANTTWEAKBAR
	
#include <AntTweakBar.h>

void Material::createAntTweakBar(const std::string& Name)
{
	TwBar* Bar(TwNewBar(Name.c_str()));
	TwDefine(std::string("'" + Name + "' color='0 0 50' ").c_str());
	TwDefine(std::string("'" + Name + "' iconified=true ").c_str());
	
	for(auto& U : _uniforms)
	{
		U.get()->addTo(Bar);
	}
}
	
#endif // NO_ANTTWEAKBAR

#include <Material.hpp>

Material::Material(const Program& P) :
	_shadingProgram(&P)
{
}
	
Material::Material(const Material& m)
{
	_shadingProgram = m._shadingProgram;
	for(const auto& u : m._uniforms)
		_uniforms.push_back(std::unique_ptr<GenericUniform>(u.get()->clone()));
	_textureCount = m._textureCount;
}

Material& Material::operator=(const Material& m)
{
	_shadingProgram = m._shadingProgram;
	for(const auto& u : m._uniforms)
		_uniforms.push_back(std::unique_ptr<GenericUniform>(u.get()->clone()));
	_textureCount = m._textureCount;
	
	return *this;
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
	
GLint Material::getLocation(const std::string& name) const
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

#include <Material.hpp>

void Material::bind() const
{	
	// Scalars
	for(auto& U : _uniform1f)
		glProgramUniform1f(_shadingProgram->getID(), U.getLocation(), U.getValue());
	for(auto& U : _uniform1i)
		glProgramUniform1i(_shadingProgram->getID(), U.getLocation(), U.getValue());
	for(auto& U : _uniform1ui)
		glProgramUniform1ui(_shadingProgram->getID(), U.getLocation(), U.getValue());
		
	// Vectors
	for(auto& U : _uniform2fv)
		glProgramUniform2fv(_shadingProgram->getID(), U.getLocation(), 1, glm::value_ptr(U.getValue()));
	for(auto& U : _uniform3fv)
		glProgramUniform3fv(_shadingProgram->getID(), U.getLocation(), 1, glm::value_ptr(U.getValue()));
	for(auto& U : _uniform4fv)
		glProgramUniform4fv(_shadingProgram->getID(), U.getLocation(), 1, glm::value_ptr(U.getValue()));
		
	// Matrices
	for(auto& U : _uniformMatrix2fv)
		glProgramUniformMatrix2fv(_shadingProgram->getID(), U.getLocation(), 1, GL_FALSE, glm::value_ptr(U.getValue()));
	for(auto& U : _uniformMatrix3fv)
		glProgramUniformMatrix3fv(_shadingProgram->getID(), U.getLocation(), 1, GL_FALSE, glm::value_ptr(U.getValue()));
	for(auto& U : _uniform4fv)
		glProgramUniformMatrix4fv(_shadingProgram->getID(), U.getLocation(), 1, GL_FALSE, glm::value_ptr(U.getValue()));
	
	// Referenced Attributes
	// Scalars
	for(auto& U : _refUniform1f)
		glProgramUniform1f(_shadingProgram->getID(), U.getLocation(), *U.getValue());
	for(auto& U : _refUniform1i)
		glProgramUniform1i(_shadingProgram->getID(), U.getLocation(), *U.getValue());
	for(auto& U : _refUniform1ui)
		glProgramUniform1ui(_shadingProgram->getID(), U.getLocation(), *U.getValue());
		
	// Vectors
	for(auto& U : _refUniform2fv)
		glProgramUniform2fv(_shadingProgram->getID(), U.getLocation(), 1, U.getValue());
	for(auto& U : _refUniform3fv)
		glProgramUniform3fv(_shadingProgram->getID(), U.getLocation(), 1, U.getValue());
	for(auto& U : _refUniform4fv)
		glProgramUniform4fv(_shadingProgram->getID(), U.getLocation(), 1, U.getValue());
		
	// Matrices
	for(auto& U : _refUniformMatrix2fv)
		glProgramUniformMatrix2fv(_shadingProgram->getID(), U.getLocation(), 1, GL_FALSE, U.getValue());
	for(auto& U : _refUniformMatrix3fv)
		glProgramUniformMatrix3fv(_shadingProgram->getID(), U.getLocation(), 1, GL_FALSE, U.getValue());
	for(auto& U : _refUniform4fv)
		glProgramUniformMatrix4fv(_shadingProgram->getID(), U.getLocation(), 1, GL_FALSE, U.getValue());
		
	// Textures
	GLuint TextureUnit = 0;
	for(auto& U : _refUniformSampler2D)
	{
		U.getValue()->bind(TextureUnit);
		glProgramUniform1i(_shadingProgram->getID(), U.getLocation(), TextureUnit);
		TextureUnit++;
	}
	
	//TextureUnit = 0;
	for(auto& U : _refUniformCubemap)
	{
		U.getValue()->bind(TextureUnit);
		glProgramUniform1i(_shadingProgram->getID(), U.getLocation(), TextureUnit);
		TextureUnit++;
	}
}

void Material::updateLocations()
{
	updateLocations(_uniform1f);
	updateLocations(_uniform1i);
	updateLocations(_uniform1ui);
	updateLocations(_uniform2fv);
	updateLocations(_uniform3fv);
	updateLocations(_uniform4fv);
	updateLocations(_uniformMatrix2fv);
	updateLocations(_uniformMatrix3fv);
	updateLocations(_uniformMatrix4fv);
	updateLocations(_refUniform1f);
	updateLocations(_refUniform1i);
	updateLocations(_refUniform1ui);
	updateLocations(_refUniform2fv);
	updateLocations(_refUniform3fv);
	updateLocations(_refUniform4fv);
	updateLocations(_refUniformMatrix2fv);
	updateLocations(_refUniformMatrix3fv);
	updateLocations(_refUniformMatrix4fv);
	updateLocations(_refUniformSampler2D);
	updateLocations(_refUniformCubemap);
}
	
#include <AntTweakBar.h>

void Material::createAntTweakBar(const std::string& Name)
{
	TwBar* Bar(TwNewBar(Name.c_str()));
	TwDefine(std::string("'" + Name + "' color='0 0 50' ").c_str());
	TwDefine(std::string("'" + Name + "' iconified=true ").c_str());
	
	for(auto& U : _uniform1i)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_INT32, &U.getRefToValue(), "");
	for(auto& U : _refUniform1i)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_INT32, U.getRefToValue(), "");
		
	for(auto& U : _uniform1f)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_FLOAT, &U.getRefToValue(), "");
	for(auto& U : _refUniform1f)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_FLOAT, U.getRefToValue(), "");
		
	for(auto& U : _uniform3fv)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_COLOR3F, &U.getRefToValue(), "");
	for(auto& U : _refUniform3fv)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_COLOR3F, U.getRefToValue(), "");
		
	for(auto& U : _uniform4fv)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_COLOR4F, &U.getRefToValue(), "");
	for(auto& U : _refUniform4fv)
		TwAddVarRW(Bar, U.getName().c_str(), TW_TYPE_COLOR4F, U.getRefToValue(), "");
}
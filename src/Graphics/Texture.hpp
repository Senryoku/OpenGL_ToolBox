#pragma once

#include <OpenGLObject.hpp>

class Texture : public OpenGLObject
{
public:
	Texture() =default;
	
	Texture(GLuint handle) : 
		OpenGLObject(handle)
	{
	}
	
	Texture(const Texture&) =default;
	Texture(Texture&&) =default;
	Texture& operator=(const Texture&) =default;
	Texture& operator=(Texture&&) =default;
		
	virtual ~Texture();

	virtual void bind(int UnitTexture = 0) const {};
protected:
	void cleanup();
};

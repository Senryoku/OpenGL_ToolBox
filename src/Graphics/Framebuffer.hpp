#pragma once

#include <iostream>

#include <Graphics/Texture2D.hpp>

template<typename TexType = Texture2D, unsigned int ColorCount = 1>
class Framebuffer : public OpenGLObject
{
public:
	Framebuffer(size_t size = 512, bool useDepth = true);
	Framebuffer(size_t width, size_t height, bool useDepth = true);
	virtual ~Framebuffer();
	
	virtual void init();
	void bind(GLenum target = GL_FRAMEBUFFER) const;
	
	inline TexType& getColor(unsigned int i = 0) { return _color[i]; }
	inline TexType& getDepth() { return _depth; }
	
	inline const TexType& getColor(unsigned int i = 0) const { return _color[i]; }
	inline const TexType& getDepth() const { return _depth; }
	
	inline size_t getWidth() const { return _width; }
	inline size_t getHeight() const { return _height; }

	static inline void unbind(GLenum target = GL_FRAMEBUFFER) { glBindFramebuffer(target, 0); };
private:
	size_t	_width = 512;
	size_t	_height = 512;
	
	std::array<TexType, ColorCount>	_color;
	
	bool 			_useDepth = true;
	TexType		_depth;
	
	void cleanup();
};
template<typename TexType, unsigned int ColorCount> 
Framebuffer<TexType, ColorCount>::Framebuffer(size_t size, bool useDepth) : 
	_width(size),
	_height(size),
	_useDepth(useDepth)
{
}
template<typename TexType, unsigned int ColorCount> 
Framebuffer<TexType, ColorCount>::Framebuffer(size_t width, size_t height, bool useDepth) : 
	_width(width),
	_height(height),
	_useDepth(useDepth)
{
}
template<typename TexType, unsigned int ColorCount> 
Framebuffer<TexType, ColorCount>::~Framebuffer()
{
	cleanup();
}
template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::cleanup()
{
	glDeleteFramebuffers(1, &_handle);
}
template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::init()
{
	glGenFramebuffers(1, &_handle);
	bind();
		
	GLenum DrawBuffers[ColorCount];
	for(size_t i = 0; i < ColorCount; ++i)
	{
		DrawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
		_color[i].create(nullptr, _width, _height, GL_RGBA, GL_RGBA, false);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, _color[i].getName(), 0);
	}
	
	glDrawBuffers(ColorCount, DrawBuffers);
	
	if(_useDepth)
	{
		_depth.create(nullptr, _width, _height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, false);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _depth.getName(), 0);
	}
	
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cerr << "Error while creating Framebuffer !" << std::endl;
		cleanup();
	}
	
	unbind();
}
template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::bind(GLenum target) const
{
	glBindFramebuffer(target, _handle);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	if(!_useDepth)
		glClear(GL_COLOR_BUFFER_BIT);
	else
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, _width, _height);
}

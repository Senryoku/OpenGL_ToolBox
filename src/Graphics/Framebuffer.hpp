#pragma once

#include <iostream>

#include <Graphics/Texture2D.hpp>

template<typename TexType = Texture2D>
class Framebuffer : public OpenGLObject
{
public:
	Framebuffer(size_t size = 512, bool useDepth = true);
	Framebuffer(size_t width, size_t height, bool useDepth = true);
	virtual ~Framebuffer();
	
	virtual void init();
	void bind(GLenum target = GL_FRAMEBUFFER) const;
	
	inline TexType& getColor() { return _color; }
	inline TexType& getDepth() { return _depth; }

	static inline void unbind(GLenum target = GL_FRAMEBUFFER) { glBindFramebuffer(target, 0); };
private:
	size_t	_width = 512;
	size_t	_height = 512;
	TexType	_color;
	
	bool _useDepth = true;
	TexType	_depth;
	
	void cleanup();
};

template<typename TexType>
Framebuffer<TexType>::Framebuffer(size_t size, bool useDepth) : 
	_width(size),
	_height(size),
	_useDepth(useDepth)
{
}

template<typename TexType>
Framebuffer<TexType>::Framebuffer(size_t width, size_t height, bool useDepth) : 
	_width(width),
	_height(height),
	_useDepth(useDepth)
{
}

template<typename TexType>
Framebuffer<TexType>::~Framebuffer()
{
	//cleanup();
}

template<typename TexType>
void Framebuffer<TexType>::cleanup()
{
	glDeleteFramebuffers(1, &_handle);
}

template<typename TexType>
void Framebuffer<TexType>::init()
{
	_color.create(nullptr, _width, _height, 4);

	glGenFramebuffers(1, &_handle);
	glBindFramebuffer(GL_FRAMEBUFFER, _handle);
		
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _color.getName(), 0);
 
	GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
	
	if(_useDepth)
	{
		_depth.create(nullptr, _width, _height, 1);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _depth.getName(), 0);
	}
	
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cerr << "Error while creating Framebuffer !" << std::endl;
		cleanup();
	}
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

template<typename TexType>
void Framebuffer<TexType>::bind(GLenum target) const
{
	glBindFramebuffer(target, _handle);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glViewport(0, 0, _width, _height);
}

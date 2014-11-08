#pragma once

#include <Graphics/Framebuffer.hpp>
#include <Graphics/CubeMap.hpp>

template<>
class Framebuffer<CubeMap>
{
public:
	Framebuffer<CubeMap>(size_t size = 512);
	~Framebuffer<CubeMap>();
	
	void init();
	void bind(GLenum target = GL_DRAW_FRAMEBUFFER);
	
	inline CubeMap& getColorCubemap() { return _color; }
	inline CubeMap& getDepthCubemap() { return _depth; }

	static inline void unbind(GLenum target = GL_DRAW_FRAMEBUFFER) { glBindFramebuffer(target, 0); };
private:
	size_t	_size = 512;
	GLuint	_framebuffer = 0;
	CubeMap	_color;
	CubeMap	_depth;
};

Framebuffer<CubeMap>::Framebuffer(size_t size) : 
	_size(size)
{
}

Framebuffer<CubeMap>::~Framebuffer<CubeMap>()
{
	glDeleteFramebuffers(1, &_framebuffer);
}

void Framebuffer<CubeMap>::init()
{
	_depth.create({nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, _size, _size, 1);
	_color.create({nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, _size, _size, 4);

	glGenFramebuffers(1, &_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER_EXT, _framebuffer);
	glFramebufferTexture(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, _depth.getName(), 0);
	glFramebufferTexture(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, _color.getName(), 0);
}

void Framebuffer<CubeMap>::bind(GLenum target)
{
	glBindFramebuffer(target, _framebuffer);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glViewport(0, 0, _size, _size);
}

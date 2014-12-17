#pragma once

#include <iostream>

#include <Texture2D.hpp>

enum FramebufferTarget
{
	Both = GL_FRAMEBUFFER,
	Draw = GL_DRAW_FRAMEBUFFER,
	Read = GL_READ_FRAMEBUFFER
};

enum BufferBit
{
	Color = GL_COLOR_BUFFER_BIT,
	Depth = GL_DEPTH_BUFFER_BIT,
	Stencil = GL_STENCIL_BUFFER_BIT,
	All = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT
};

template<typename TexType = Texture2D, unsigned int ColorCount = 1>
class Framebuffer : public OpenGLObject
{
public:
	Framebuffer(size_t size = 512, bool useDepth = true);
	Framebuffer(size_t width, size_t height, bool useDepth = true);
	virtual ~Framebuffer();
	
	virtual void init();
	
	/**
	 * Bind this framebuffer
	 * @param target GL_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER or GL_READ_FRAMEBUFFER
	**/
	void bind(FramebufferTarget target = FramebufferTarget::Both) const;
	
	/**
	 * Clear the framebuffer.
	 * Equivalent to a call to clear(GLenum target) with
	 * Buffer::Color if any color buffer is attached to it, and
	 * Buffer::Depth if using depth.
	 * @see clear(GLenum target)
	 * @see glClear
	**/
	void clear() const;
	
	/**
	 * Clear the framebuffer.
	 * @param target Combinaison of Buffer
	 * @see clear()
	 * @see glClear
	**/
	void clear(BufferBit target) const;
	
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
	
	bool 		_useDepth = true;
	TexType		_depth;
	
	void cleanup();
};

#include <Framebuffer.tcc>

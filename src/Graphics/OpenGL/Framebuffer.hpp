#pragma once

#include <iostream>

#include <Texture2D.hpp>

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

#include <Framebuffer.tcc>

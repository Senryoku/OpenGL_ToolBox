#pragma once

#include <iostream>

#include <Texture2D.hpp>

/**
 * Describes the targets a framebuffer can be bound to.
**/
enum FramebufferTarget
{
	Both = GL_FRAMEBUFFER,		///< Draw & Read target
	Draw = GL_DRAW_FRAMEBUFFER,	///< Draw target
	Read = GL_READ_FRAMEBUFFER	///< Read target
};

/**
 * Masks identifying framebuffers' buffers.
**/
enum BufferBit
{
	Color = GL_COLOR_BUFFER_BIT,		///< Color Buffer
	Depth = GL_DEPTH_BUFFER_BIT,		///< Depth Buffer
	Stencil = GL_STENCIL_BUFFER_BIT,	///< Stencil Buffer
	All = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT
};

/**
 * Framebuffer Object (FBO)
 * @param TexType Type of the textures attached to the framebuffer.
 * @param ColorCount Number of Color buffers (attachments).
**/
template<typename TexType = Texture2D, unsigned int ColorCount = 1>
class Framebuffer : public OpenGLObject
{
public:
	/**
	 * Default constructor.
	 * @param size Resolution of the framebuffer (internal textures).
	 * @param useDepth Specify if this framebuffer as a Depth Buffer.
	**/
	Framebuffer(size_t size = 512, bool useDepth = true);
	
	/**
	 * Constructor for a rectangular framebuffer.
	 * @param width X Resolution of the framebuffer (internal textures).
	 * @param height Y Resolution of the framebuffer (internal textures).
	 * @param useDepth Specify if this framebuffer as a Depth Buffer.
	**/
	Framebuffer(size_t width, size_t height, bool useDepth = true);
	
	/**
	 * Destructor
	**/
	virtual ~Framebuffer();
	
	/**
	 * Initialize the OpenGLObject
	**/
	void init();
	
	/**
	 * Destroys OpenGL object.
	**/
	void cleanup();
	
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
	
	/**
	 * @param i Index of the color buffer.
	 * @return Texture attachment to ith color buffer.
	**/
	inline TexType& getColor(unsigned int i = 0) { return _color[i]; }
	
	/**
	 * @return Texture of the Depth buffer.
	**/
	inline TexType& getDepth() { return _depth; }
	
	/**
	 * @param i Index of the color buffer.
	 * @return Texture attachment to ith color buffer.
	**/
	inline const TexType& getColor(unsigned int i = 0) const { return _color[i]; }
	
	/**
	 * @return Texture of the Depth buffer.
	**/
	inline const TexType& getDepth() const { return _depth; }
	
	/**
	 * @return Width of the framebuffer.
	**/
	inline size_t getWidth() const { return _width; }
	
	/**
	 * @return Height of the framebuffer.
	**/
	inline size_t getHeight() const { return _height; }

	/**
	 * Unbind any currently bound to target FBO.
	 * (Restore default framebuffer)
	 * @param target Target
	**/
	static inline void unbind(GLenum target = GL_FRAMEBUFFER) { glBindFramebuffer(target, 0); };
private:
	size_t	_width = 512;	///< Width of the framebuffer.
	size_t	_height = 512;	///< Height of the framebuffer.
	
	std::array<TexType, ColorCount>	_color;	///< Textures attached to the color buffers.
	
	bool 		_useDepth = true;	///< Does it have a Depth Buffer ?
	TexType		_depth;				///< Texture attached to the Depth Buffer
};

#include <Framebuffer.tcc>



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
	_handle = 0;
}

template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::init()
{
	if(_handle != 0)
		cleanup();

	glGenFramebuffers(1, &_handle);
	glBindFramebuffer(GL_FRAMEBUFFER, _handle);
		
	if(ColorCount > 0)
	{
		GLenum DrawBuffers[ColorCount];
		for(size_t i = 0; i < ColorCount; ++i)
		{
			DrawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
			if(!_color[i])
				_color[i].create(nullptr, _width, _height, GL_RGBA, GL_RGBA, false);
			glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, _color[i].getName(), 0);
		}
		
		glDrawBuffers(ColorCount, DrawBuffers);
	} else {
		glDrawBuffer(GL_NONE);
	}
	
	if(_useDepth)
	{
		if(!_depth)
			_depth.create(nullptr, _width, _height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, false);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _depth.getName(), 0);
	}
	
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cerr << "Error while creating Framebuffer !" << std::endl;
		cleanup();
	}
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::bind(FramebufferTarget target) const
{
	glBindFramebuffer(static_cast<GLenum>(target), _handle);
	if(target == Read)
	{
		glReadBuffer(GL_COLOR_ATTACHMENT0); ///< @todo Do Better.
	} else {
		glViewport(0, 0, _width, _height);
	}
}

template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::clear() const
{
	GLbitfield target = (ColorCount > 0) ? GL_COLOR_BUFFER_BIT : 0;
	if(_useDepth) target |= GL_DEPTH_BUFFER_BIT;
	glClear(target);
}
	
template<typename TexType, unsigned int ColorCount> 
void Framebuffer<TexType, ColorCount>::clear(BufferBit target) const
{
	glClear(static_cast<GLbitfield>(target));
}

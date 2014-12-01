#include <Buffer.hpp>

#include <cassert>

Buffer::Buffer(Type type, GLuint handle) : 
	OpenGLObject(handle),
	_type(type)
{
}
	
Buffer::~Buffer()
{
	cleanup();
}

void Buffer::cleanup()
{
	glDeleteBuffers(1, &_handle);
	_handle = 0;
}

void Buffer::bind() const
{
	assert(isValid());
    glBindBuffer(static_cast<GLenum>(_type), getName());
}

void Buffer::bindBase(GLuint bindingPoint) const
{
	assert(_type == AtomicCounter || _type == TransformFeedback || _type == Uniform || _type == ShaderStorage);
	glBindBufferBase(static_cast<GLenum>(_type), bindingPoint, getName());
}

void Buffer::bindRange(GLuint bindingPoint, GLintptr offset, GLsizeiptr size) const
{
	assert(_type == AtomicCounter || _type == TransformFeedback || _type == Uniform || _type == ShaderStorage);
	glBindBufferRange(static_cast<GLenum>(_type), bindingPoint, getName(), offset, size);
}
	
void Buffer::unbind() const
{
    glBindBuffer(static_cast<GLenum>(_type), 0);
}

void Buffer::init()
{
	if(_handle != 0)
		cleanup();
	glGenBuffers(1, &_handle);
}

void Buffer::init(Type t)
{
	setType(t);
	init();
}

void Buffer::data(const void* data, size_t size, Buffer::Usage usage) const
{
	bind();
    glBufferData(_type, size, data, static_cast<GLenum>(usage));
}

void Buffer::store(const void* data, size_t size, Buffer::StorageUsage flags) const
{
	bind();
    glBufferStorage(_type, size, data, static_cast<GLbitfield>(flags));
}

void Buffer::subData(size_t offset, size_t size, const void* data) const
{
	bind();
	glBufferSubData(_type, offset, size, data);
}

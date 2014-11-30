#include <Buffer.hpp>

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
    glBindBuffer(static_cast<GLenum>(_type), _handle);
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

void Buffer::data(const void* data, size_t size, Buffer::Usage usage) const
{
    glBufferData(_type, size, data, static_cast<GLenum>(usage));
}

void Buffer::store(const void* data, size_t size, Buffer::StorageUsage flags) const
{
    glBufferStorage(_type, size, data, static_cast<GLbitfield>(flags));
}

void Buffer::subData(size_t offset, size_t size, const void* data) const
{
	glBufferSubData(_type, offset, size, data);
}

#include <Buffer.hpp>

void Buffer::bind() const
{
    glBindBuffer(static_cast<GLenum>(_type), _handle);
}

void Buffer::generate()
{
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
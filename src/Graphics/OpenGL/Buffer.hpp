#pragma once

#include <OpenGLObject.hpp>

class Buffer : public OpenGLObject
{
public:
	/**
	 * Bind targets
	**/
	enum Type
	{
		VertexAttributes = GL_ARRAY_BUFFER,  									///<Vertex attributes
		AtomicCounter = GL_ATOMIC_COUNTER_BUFFER, 					///< Atomic counter storage
		CopyRead = GL_COPY_READ_BUFFER, 									///<Buffer copy source
		CopyWrite = GL_COPY_WRITE_BUFFER, 									///< Buffer copy destination
		IndirectDispatch = GL_DISPATCH_INDIRECT_BUFFER, 			///< Indirect compute dispatch commands
		IndirectDraw = GL_DRAW_INDIRECT_BUFFER, 						///< Indirect command arguments
		VertexIndices = GL_ELEMENT_ARRAY_BUFFER, 						///< Vertex array indices
		PixelPack = GL_PIXEL_PACK_BUFFER, 									///< Pixel read target
		PixelUnpack = GL_PIXEL_UNPACK_BUFFER,								///< Texture data source
		Query = GL_QUERY_BUFFER,													///< Query result buffer
		ShaderStorage = GL_SHADER_STORAGE_BUFFER, 					///< Read-write storage for shaders
		Texture = GL_TEXTURE_BUFFER, 											///< Texture data buffer
		TransformFeedback = GL_TRANSFORM_FEEDBACK_BUFFER, 	///< Transform feedback buffer
		Uniform = GL_UNIFORM_BUFFER 											///< Uniform block storage
	};
	
	/**
	 * Usage hint for mutable buffers
	 * 
	 * STATIC: The user will set the data once.
	 * DYNAMIC: The user will set the data occasionally.
	 * STREAM: The user will be changing the data after every use. Or almost every use.
	 *
	 * DRAW: The user will be writing data to the buffer, but the user will not read it.
	 * READ: The user will not be writing data, but the user will be reading it back.
	 * COPY: The user will be neither writing nor reading the data.
	 *
	**/
	enum Usage
	{
		StaticDraw = GL_STATIC_DRAW,
		StaticRead = GL_STATIC_READ,
		StaticCopy = GL_STATIC_COPY,
		DynamicDraw = GL_DYNAMIC_DRAW,
		DynamicRead = GL_DYNAMIC_READ,
		DynamicCopy = GL_DYNAMIC_COPY,
		StreamDraw = GL_STREAM_DRAW,
		StreamRead = GL_STREAM_READ,
		StreamCopy = GL_STREAM_COPY
	};
	
	/**
	 * Usage hint for immutable buffers
	**/
	enum StorageUsage
	{
		Unspecified = 0,
		DynamicStorageBit = GL_DYNAMIC_STORAGE_BIT, ///< The contents of the data store may be updated after creation through calls to glBufferSubData. If this bit is not set, the buffer content may not be directly updated by the client. The data argument may be used to specify the initial content of the buffer's data store regardless of the presence of the GL_DYNAMIC_STORAGE_BIT​. Regardless of the presence of this bit, buffers may always be updated with server-side calls such as glCopyBufferSubData​ and glClearBufferSubData​.
		MapReadBit = GL_MAP_READ_BIT, ///< The data store may be mapped by the client for read access and a pointer in the client's address space obtained that may be read from.
		MapWriteBit = GL_MAP_WRITE_BIT, ///< The data store may be mapped by the client for write access and a pointer in the client's address space obtained that may be written through.
		MapPersistentBit = GL_MAP_PERSISTENT_BIT, ///< The client may request that the server read from or write to the buffer while it is mapped. The client's pointer to the data store remains valid so long as the data store is mapped, even during execution of drawing or dispatch commands.
		MapCoherentBit = GL_MAP_COHERENT_BIT, ///< Shared access to buffers that are simultaneously mapped for client access and are used by the server will be coherent, so long as that mapping is performed using glMapBufferRange​. That is, data written to the store by either the client or server will be immediately visible to the other with no further action taken by the application. In particular,
		ClientStorageBit = GL_CLIENT_STORAGE_BIT /// A hint that the buffer object's data should be stored in client memory. Assuming that such a distinction exists in the driver.
	};
	
	Buffer() =default;
	
	Buffer(Type type, GLuint handle = 0) : 
		OpenGLObject(handle),
		_type(type)
	{
	}
	
	Buffer(const Buffer&) =default;
	Buffer(Buffer&&) =default;
	Buffer& operator=(const Buffer&) =default;
	Buffer& operator=(Buffer&&) =default;
		
	virtual ~Buffer();
	
	/**
	 * Generates a new buffer name to manage.
	**/
	void init();
	
	/**
	 * Destroy managed buffer.
	**/
	void cleanup();
	
	/**
	 * Binds the buffer to its assigned target (type)
	**/
	void bind() const;
	
	/**
	 * Unbinds the buffer.
	**/
	void unbind() const;
	
	/**
	 * glBufferData
	**/
	void data(const void* data, size_t size, Buffer::Usage usage) const;

	/**
	 * glBufferStorage
	**/
	void store(const void* data, size_t size, Buffer::StorageUsage flags) const;
	
	/**
	 * glBufferSubData​
	**/
	void subData(size_t offset, size_t size, const void* data) const;
	
private:
	Type	_type;
};


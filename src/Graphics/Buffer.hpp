#pragma once

#include <OpenGLObject.hpp>

class Buffer : public OpenGLObject
{
public:
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
	}

    glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Normals[0]) * Normals.size(), &Normals[0],
                    GL_STATIC_DRAW);
	
	// Here ??
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
private:
};


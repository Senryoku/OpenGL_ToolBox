#pragma once

#include <cassert>

#include <OpenGLObject.hpp>
#include <Enums.hpp>
#include <Buffer.hpp>

/**
 * OpenGL TransformFeedback Object
**/
class TransformFeedback : public OpenGLObject
{
public:
	/**
	 * Generates a new TransformFeedback name to manage.
	**/
	void init();
	
	/**
	 * Destroy managed TransformFeedback object.
	**/
	void cleanup();
	
	/**
	 * Binds the TransformFeedback object
	 * @see unbind()
	**/
	inline void bind() const;
	
	/**
	 * Binds the specified buffer to index for this TransformFeedback Object.
	 * Note: This effectively binds the TransformFeedback Object (calls bind()).
	 * @param index Binding Point
	 * @param buffer Buffer to bind
	 * @param offset Starting offset in basic machine unit (optional, default: 0)
	 * @param size Amount of data in machine unit to bind (optional, default: 0, means the entire buffer)
	 * @see bind()
	**/ 
	void bindBuffer(GLuint index, const Buffer& buffer, GLintptr offset = 0, GLsizeiptr size = 0);
	
	/**
	 * Draw using this transform feedback.
	 * @param primitiveType Type of primitives to draw.
	**/
	inline void draw(Primitive primitiveType) const;
	
	/**
	 * Draw using this transform feedback.
	 * @param primitiveType Type of primitives to draw.
	 * @param instanceCount Number of instances to draw.
	**/
	inline void drawInstanced(Primitive primitiveType, GLuint instanceCount) const;
	
	/**
	 * Draw using this transform feedback.
	 * @param primitiveType Type of primitives to draw.
	 * @param stream Stream transform buffer from which to retrieve a primitive count.
	**/
	inline void drawStream(Primitive primitiveType, GLuint stream) const;
	
	/**
	 * Draw using this transform feedback.
	 * @param primitiveType Type of primitives to draw.
	 * @param stream Stream transform buffer from which to retrieve a primitive count.
	 * @param instanceCount Number of instances to draw.
	**/
	inline void drawStreamInstanced(Primitive primitiveType, GLuint stream, GLuint instanceCount) const;
	
	// Static
	
	/**
	 * Unbinds any currently bound TransformFeedback object (back to the default one (0)).
	**/
	inline static void unbind();
	
	/**
	 * Begins a transform feedback operation, using the currently bound container object (TFO, default (0) is a valid option).
	 * @param primitiveType Type of primitives captured by the transform feedback. Must be Points, Lines or Triangles (default).
	 * @see end
	**/
	inline static void begin(Primitive primitiveType = Primitive::Triangles);
	
	/**
	 * Ends a transform feedback operation.
	 * @see begin
	**/
	inline static void end();
	
	/**
	 * Pauses a transform feedback operation.
	**/
	inline static void pause();
	
	/**
	 * Resumes a transform feedback operation.
	**/
	inline static void resume();
	
	/**
	 * Wrapper arround glDrawTransformFeedback
	 * @param primitiveType Type of primitives to draw.
	 * @param transformFeedbackObject Name of a TransformFeedback Object from which to retrieve a primitive count (0 for the default one is a valid value).
	**/
	inline static void draw(Primitive primitiveType, GLuint transformFeedbackObject);
	
	/**
	 * Wrapper arround glDrawTransformFeedbackInstanced
	 * @param primitiveType Type of primitives to draw.
	 * @param transformFeedbackObject Name of a TransformFeedback Object from which to retrieve a primitive count (0 for the default one is a valid value).
	 * @param instanceCount Number of instances to draw.
	**/
	inline static void drawInstanced(Primitive primitiveType, GLuint transformFeedbackObject, GLuint instanceCount);
	
	/**
	 * Wrapper arround glDrawTransformFeedbackStream
	 * @param primitiveType Type of primitives to draw.
	 * @param transformFeedbackObject Name of a TransformFeedback Object from which to retrieve a primitive count (0 for the default one is a valid value).
	 * @param stream Stream transform buffer from which to retrieve a primitive count.
	 * @todo add overload for Buffer (instead of raw stream).
	**/
	inline static void drawStream(Primitive primitiveType, GLuint transformFeedbackObject, GLuint stream);
	
	/**
	 * Wrapper arround glDrawTransformFeedbackStreamInstanced
	 * @param primitiveType Type of primitives to draw.
	 * @param transformFeedbackObject Name of a TransformFeedback Object from which to retrieve a primitive count (0 for the default one is a valid value).
	 * @param stream Stream transform buffer from which to retrieve a primitive count.
	 * @param instanceCount Number of instances to draw.
	 * @todo add overload for Buffer (instead of raw stream).
	**/
	inline static void drawStreamInstanced(Primitive primitiveType, GLuint transformFeedbackObject, GLuint stream, GLuint instanceCount);
	
	/**
	 * Enable rasterization (default behavior).
	 * @todo Move this elsewhere ? (If we have a context class one day...)
	 * @see GL_RASTERIZER_DISCARD 
	**/
	inline static void enableRasterization();
	
	/**
	 * Disable rasterization (useful for TransformFeedback).
	 * @todo Move this elsewhere ? (If we have a context class one day...)
	 * @see GL_RASTERIZER_DISCARD 
	**/
	inline static void disableRasterization();
};

inline void TransformFeedback::bind() const
{
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, _handle);
}

inline void TransformFeedback::unbind()
{
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
}

inline void TransformFeedback::draw(Primitive primitiveType) const
{
	draw(primitiveType, _handle);
}

inline void TransformFeedback::drawInstanced(Primitive primitiveType, GLuint instanceCount) const
{
	drawInstanced(primitiveType, _handle, instanceCount);
}

inline void TransformFeedback::drawStream(Primitive primitiveType, GLuint stream) const
{
	drawStream(primitiveType, _handle, stream);
}

inline void TransformFeedback::drawStreamInstanced(Primitive primitiveType, GLuint stream, GLuint instanceCount) const
{
	drawStreamInstanced(primitiveType, _handle, stream, instanceCount);
}

// Static

inline void TransformFeedback::begin(Primitive primitiveType)
{
	assert(primitiveType == Points || primitiveType == Lines || primitiveType == Triangles);
	glBeginTransformFeedback(glenum(primitiveType));
}

inline void TransformFeedback::end()
{
	glEndTransformFeedback();
}

inline void TransformFeedback::pause()
{
	glPauseTransformFeedback();
}

inline void TransformFeedback::resume()
{
	glResumeTransformFeedback();
}

inline void TransformFeedback::draw(Primitive primitiveType, GLuint transformFeedbackObject)
{
	glDrawTransformFeedback(glenum(primitiveType), transformFeedbackObject);
}

inline void TransformFeedback::drawInstanced(Primitive primitiveType, GLuint transformFeedbackObject, GLuint instanceCount)
{
	glDrawTransformFeedbackInstanced(glenum(primitiveType), transformFeedbackObject, instanceCount);
}

inline void TransformFeedback::drawStream(Primitive primitiveType, GLuint transformFeedbackObject, GLuint stream)
{
	glDrawTransformFeedbackStream(glenum(primitiveType), transformFeedbackObject, stream);
}

inline void TransformFeedback::drawStreamInstanced(Primitive primitiveType, GLuint transformFeedbackObject, GLuint stream, GLuint instanceCount)
{
	glDrawTransformFeedbackStreamInstanced(glenum(primitiveType), transformFeedbackObject, stream, instanceCount);
}

inline void TransformFeedback::enableRasterization()
{
	glDisable(GL_RASTERIZER_DISCARD);
}

inline void TransformFeedback::disableRasterization()
{
	glEnable(GL_RASTERIZER_DISCARD);
}

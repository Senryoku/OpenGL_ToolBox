#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

/**
 * Base class for every OpenGL Object
**/
class OpenGLObject
{
public:
	/**
	 * Generic Binder
	 * Does nothing at this stage.
	**/
	class Binder
	{
	public:
		Binder(const OpenGLObject& t);
		
		~Binder();
	private:
		const OpenGLObject&	_object;
	};
	
	OpenGLObject();
	/**
	 * Constructs a instance managing an existing OpenGL Object.
	 * (Copy-constructor is generally a better option)
	 * @param handle Object to manage.
	**/
	OpenGLObject(GLuint handle);
	OpenGLObject(const OpenGLObject&) =default;
	OpenGLObject(OpenGLObject&&) =default;
	OpenGLObject& operator=(const OpenGLObject&) =default;
	OpenGLObject& operator=(OpenGLObject&&) =default;
	virtual ~OpenGLObject();
	
	/** 
	 * Returns OpenGL name of the object
	**/
	inline GLuint getName() const { return _handle; }
	
	/** 
	 * Sets the OpenGL name of the object managed by this instance
	 * /!\ Use with caution ! (assignment operator is probably what you're looking for)
	 * @param n New object to manage.
	**/
	inline void setName(GLuint n) { _handle = n; }
	
	/**
	 * Returns true if the object seems valid.
	 * Can be overloaded for concrete types if there is other attributes to test.
	**/
	virtual inline bool isValid() const { return _handle != 0; }
	
	/**
	 * Syntactic sugar for isValid().
	 * Allow the use of quick tests :
	 * @code if(OpenGLObject_Instance) { ... }
	 * @endcode
	**/
	inline operator bool() const { return this->isValid(); }
	
protected:
	GLuint	_handle; ///< OpenGL name of the managed object
};

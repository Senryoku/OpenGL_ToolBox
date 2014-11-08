#pragma once

class GenericUniform
{
public:
	GenericUniform() =default;
	GenericUniform(const std::string& N, GLuint L) :
		_name(N),
		_location(L),
	{ }
	virtual ~GenericUniform() =0;
	
	inline const std::string& getName() const { return _name; }
	inline GLuint getLocation() const { return _location; }
	
	inline void setLocation(const GLuint val) { _location = val; }
	
private:
	std::string	_name;
	GLuint			_location;
};

template<typename T>
class Uniform : public GenericUniform
{
public:
	Uniform() =default;
	Uniform(const std::string& N, GLuint L, T& V) :
		GenericUniform(N, L),
		_value(V)
	{ }
	
	inline const T& getValue() const { return _value; }
	inline T& getRefToValue() { return _value; }
	
	inline void setValue(const T& val) { _value = val; }
	
private:
	T					_value;
};

#include <Texture.hpp>

template<>
class Uniform<Texture> : public GenericUniform
{
public:
	Uniform() =default;
	Uniform(const std::string& N, GLuint L, GLuint U, Texture& V) :
		GenericUniform(N, L),
		_value(V),
		_textureUnit(U)
	{ }
	
	inline const Texture& getValue() const { return _value; }
	inline Texture& getRefToValue() { return _value; }
	inline GLuint getTextureUnit() const { return _textureUnit; }
	
	inline void setValue(const Texture& val) { _value = val; }
	inline void setTextureUnit(GLuint U) { _textureUnit = U; }
	
private:
	Texture		_value;
	GLuint 		_textureUnit;
};

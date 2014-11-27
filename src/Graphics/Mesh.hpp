#pragma once

#include <string>

#include <OpenGLObject.hpp>
#include <Buffer.hpp>
#include <VertexArray.hpp>

class Mesh : public OpenGLObject
{
public:
	~Mesh();

	bool load(const std::string& path);

private:
	VertexArray		_vao;
	Buffer			_vertices;
	Buffer			_indices;
};

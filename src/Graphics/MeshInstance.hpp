#pragma once

#include <Mesh.hpp>

class MeshInstance
{
public:
	MeshInstance(const Mesh& mesh, const glm::mat4& modelMatrix = glm::mat4(1.0));
	
	void draw() const
	{
		_material.use();
		_mesh->draw();
	}
	
	Material& getMaterial() { return _material; }
	
	const Mesh& getMesh() const { return *_mesh; }
	const glm::mat4& getModelMatrix() const { return _modelMatrix; }

private:
	const Mesh*		_mesh = nullptr;	
	Material		_material;
	glm::mat4		_modelMatrix;
};

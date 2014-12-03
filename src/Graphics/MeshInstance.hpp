#pragma once

#include <Mesh.hpp>

class MeshInstance
{
public:
	MeshInstance(const Mesh& mesh);
	
	void draw() const
	{
		_material.use();
		_mesh->draw();
	}
	
	Material& getMaterial() { return _material; }
	const Mesh& getMesh() const { return *_mesh; }

private:
	const Mesh*		_mesh = nullptr;	
	Material		_material;
};

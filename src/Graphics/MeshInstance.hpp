#pragma once

#include <Mesh.hpp>

class MeshInstance
{
public:
	MeshInstance(const Mesh& mesh);

private:
	const Mesh*	_mesh = nullptr;	
	Material		_material;
};

#include <MeshInstance.hpp>

MeshInstance::MeshInstance(const Mesh& mesh) :
	_mesh(&mesh),
	_material(mesh.getMaterial())
{
}

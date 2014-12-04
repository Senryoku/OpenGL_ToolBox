#include <MeshInstance.hpp>

MeshInstance::MeshInstance(const Mesh& mesh, const glm::mat4& modelMatrix) :
	_mesh(&mesh),
	_material(mesh.getMaterial()),
	_modelMatrix(modelMatrix)
{
	// mat4 pointers doesn't work anymore ? -.-'
	_material.setUniform("ModelMatrix", _modelMatrix);
}

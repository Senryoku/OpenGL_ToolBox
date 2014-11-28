#include <Mesh.hpp>

#include <iostream>

#include <assimp/Importer.hpp> 	// C++ importer interface
#include <assimp/scene.h> 			// Output data structure
#include <assimp/postprocess.h> // Post processing flags

#include <StringConversion.hpp>
#include <ResourcesManager.hpp>

//////////////////////// Mesh::Triangle ////////////////////////////////////////

Mesh::Triangle::Triangle(size_t v1,
						 size_t v2,
						 size_t v3) :
	vertices{v1, v2, v3}
{
}

/////////////////////// Mesh::Vertex ///////////////////////////////////////////

Mesh::Vertex::Vertex(glm::vec3 pos,
					 glm::vec3 nor,
					 glm::vec2 tex) :
	position(pos),
	normal(nor),
	texcoord(tex)
{
}

/////////////////////// Mesh ///////////////////////////////////////////////////

Mesh::Mesh() :
	_vao(),
	_vertex_buffer(Buffer::VertexAttributes),
	_index_buffer(Buffer::VertexIndices)
{
}

void Mesh::createVAO()
{
	_vao.init();
	_vao.bind();
	
	_vertex_buffer.init();
	_vertex_buffer.bind();
	
	_vertex_buffer.data(&_vertices[0], sizeof(Vertex)*_vertices.size(), Buffer::StaticDraw);

    _vao.attribute(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid *) offsetof(struct Vertex, position));
    _vao.attribute(1, 3, GL_FLOAT, GL_TRUE, sizeof(Vertex), (GLvoid *) offsetof(struct Vertex, normal));
    _vao.attribute(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid *) offsetof(struct Vertex, texcoord));

	_index_buffer.init();
	_index_buffer.bind();
	_index_buffer.data(&_triangles[0], sizeof(size_t)*_triangles.size()*3, Buffer::StaticDraw);
	
	_vao.unbind(); // Unbind first on purpose :)
	_index_buffer.unbind();
	_vertex_buffer.unbind();
}
	
std::vector<Mesh*> Mesh::load(const std::string& path)
{
	std::cout << "Loading " << path << " using assimp..." << std::endl;
	
	std::vector<Mesh*> M;
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path.c_str(), 
																aiProcess_CalcTangentSpace |
																aiProcess_Triangulate |
																aiProcess_JoinIdenticalVertices |
																aiProcess_SortByPType |
																aiProcess_GenNormals | aiProcess_FlipUVs	 );
																
	 // If the import failed, report it
	if( !scene)
	{
		std::cerr << importer.GetErrorString() << std::endl;
		return M;
	}
	
	if(scene->HasMeshes())
	{
		M.resize(scene->mNumMeshes);
		for(unsigned int meshIdx = 0; meshIdx < scene->mNumMeshes; ++meshIdx)
		{
			aiMesh* LoadedMesh = scene->mMeshes[meshIdx];
			
			std::string name(path);
			name.append("::" + StringConversion::to_string(meshIdx));
			name.append(scene->mMeshes[meshIdx]->mName.C_Str());
			while(ResourcesManager::getInstance().isMesh(name))
			{
				std::cout << "Warning: Mesh '" << name << "' was already loaded. Re-loading it under the name '";
				name.append("_");
				std::cout << name << "'." << std::endl;
			}
			
			std::cout << "Loading '" << name << "'." << std::endl;
			M[meshIdx] = &ResourcesManager::getInstance().getMesh(name);
			
			//std::cout << "Material Index: " << LoadedMesh->mMaterialIndex << std::endl;
			aiVector3D* n = LoadedMesh->mNormals;
			aiVector3D** t = LoadedMesh->mTextureCoords;
			for(unsigned int i = 0; i < LoadedMesh->mNumVertices; ++i)
			{
				aiVector3D& v = LoadedMesh->mVertices[i];
				M[meshIdx]->getVertices().push_back(Mesh::Vertex(glm::vec3(v.x, v.y, v.z),
																				(n == nullptr) ? glm::vec3() : glm::vec3((*(n + i)).x, (*(n + i)).y, (*(n + i)).z),
																				(t == nullptr) ? glm::vec2() : glm::vec2(t[0][i].x, t[0][i].y)));
			}
			
			for(unsigned int i = 0; i < LoadedMesh->mNumFaces; ++i)
			{
				aiFace& f = LoadedMesh->mFaces[i];
				unsigned int* idx = f.mIndices;
				M[meshIdx]->getTriangles().push_back(Mesh::Triangle(idx[0], idx[1], idx[2]));
			}
		}
	}
	
	std::cout << "Loading of " << path << " done." << std::endl;
	
	return M;
}

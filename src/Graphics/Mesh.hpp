#pragma once

#include <string>
#include <vector>
#include <array>

#include <glm/glm.hpp>
#define GLM_FORCE_RADIANS
#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include <Buffer.hpp>
#include <VertexArray.hpp>
#include <BoundingShape.hpp>
#include <Material.hpp>

class Mesh
{
public:
	struct Triangle
	{
		Triangle(size_t v1,
		         size_t v2,
				 size_t v3);

		Triangle(const Triangle& T) =default;

		std::array<size_t, 3>	vertices;
	};

	struct Vertex
	{
		Vertex() {}
		Vertex(glm::vec3 pos,
				  glm::vec3 nor,
				  glm::vec2 tex);

		glm::vec3	position;
		glm::vec3	normal;
		glm::vec2	texcoord;
	};
	
	Mesh();
	~Mesh() =default;
	
	inline std::vector<Vertex>&	getVertices()		{ return _vertices; }
	inline std::vector<Triangle>&	getTriangles()	{ return _triangles; }
	inline Material& getMaterial()						{ return _material; }
	
	inline const std::vector<Vertex>&	getVertices()	const	{ return _vertices; }
	inline const std::vector<Triangle>&	getTriangles() const { return _triangles; }
	inline const Material& getMaterial() const { return _material; }
	
	void computeNormals();
	
	void createVAO();
	void draw() const;
	
	const BoundingBox& getBoundingBox() const { return _bbox; }

	static std::vector<Mesh*> load(const std::string& path);
	
private:
	std::vector<Vertex>			_vertices;
	std::vector<Triangle>		_triangles;
	
	VertexArray			_vao;
	Buffer				_vertex_buffer;
	Buffer				_index_buffer;
	
	Material 			_material; ///< Base (default) Material for this mesh
	
	BoundingBox			_bbox;
};

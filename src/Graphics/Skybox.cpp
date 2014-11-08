#include <Skybox.hpp>

#include <cmath>

bool	Skybox::s_init 			= false;
GLuint	Skybox::s_vertexBuffer	= 0;
GLuint	Skybox::s_indiceBuffer	= 0;

Skybox::Skybox()
{
}

Skybox::Skybox(const std::array<std::string, 6>& Paths) :
	Skybox()
{
	loadCubeMap(Paths);
}

void Skybox::loadCubeMap(const std::array<std::string, 6>& Paths)
{
	if(!s_init)
		init();
		
	_cubeMap.load(Paths);
}
	
void Skybox::draw()
{
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_indiceBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, s_vertexBuffer);
	_cubeMap.bind();
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawElements(GL_QUADS, 6 * 4, GL_UNSIGNED_BYTE, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
}

void Skybox::init()
{
	float boxsize =  std::sqrt(3)/3.f;

	float vertices[24] = {
		-boxsize,	-boxsize,	-boxsize,
		boxsize,	-boxsize,	-boxsize,
		-boxsize,	boxsize,	-boxsize,
		boxsize,	boxsize,	-boxsize,
		-boxsize,	-boxsize,	boxsize,
		boxsize,	-boxsize,	boxsize,
		-boxsize,	boxsize,	boxsize,
		boxsize,	boxsize,	boxsize
	};

	GLubyte indices[24] = {
		1, 5, 7, 3,
		2, 0, 4, 6,
		4, 5, 7, 6,
		0, 1, 3, 2,
		0, 1, 5, 4,
		3, 2, 6, 7
	};
	
	glGenBuffers(1, &s_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, s_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	
	glGenBuffers(1, &s_indiceBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_indiceBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	
	s_init = true;
}
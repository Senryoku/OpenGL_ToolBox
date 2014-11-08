#pragma once

#include <CubeMap.hpp>

class Skybox
{
public:
	Skybox();
	Skybox(const std::array<std::string, 6>& Paths);
	~Skybox() =default;
	
	void draw();
	void loadCubeMap(const std::array<std::string, 6>& Paths);
	
	CubeMap& getCubemap() { return _cubeMap; }
	
private:
	CubeMap	_cubeMap;
	
	static void		init();
	
	static bool		s_init;
	static GLuint	s_vertexBuffer;
	static GLuint	s_indiceBuffer;
};
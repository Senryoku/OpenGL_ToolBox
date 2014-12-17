#include "GeometryShader.hpp"

void GeometryShader::init()
{
	if(_handle != 0)
		return;
		
	_handle = glCreateShader(GL_GEOMETRY_SHADER);
	if(_handle == 0)
	{
		std::cerr << __PRETTY_FUNCTION__ << " : Error glCreateShader(GL_COMPUTE_SHADER)" << std::endl;
	}
}

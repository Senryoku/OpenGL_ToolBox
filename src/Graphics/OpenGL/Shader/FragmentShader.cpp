#include "FragmentShader.hpp"

void FragmentShader::initOGL()
{
	if(_handle != 0)
		return;
		
	_handle = glCreateShader(GL_FRAGMENT_SHADER);
	if(_handle == 0)
	{
		std::cerr << __FUNCTION__ << " : Error glCreateShader(GL_COMPUTE_SHADER)" << std::endl;
	}
}

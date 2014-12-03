#include <Sampler.hpp>

Sampler::~Sampler()
{
	cleanup();
}

void Sampler::bind(unsigned int unit) const 
{
	glBindSampler(unit, _handle);
}

void Sampler::cleanup()
{
	glDeleteSamplers(1, &_handle);
}

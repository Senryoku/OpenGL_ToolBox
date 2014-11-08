#include <AntTweakBarBinding.hpp>

void createTweak(TwBar* bar, const std::string& name, float* value)
{
	TwAddVarRW(bar, name.c_str(), TW_TYPE_FLOAT, value, "");
}

void createTweak(TwBar* bar, const std::string& name, glm::vec2* value)
{
	TwAddVarRW(bar, name.c_str(), TW_TYPE_FLOAT, &(*value).x, "");
	TwAddVarRW(bar, name.c_str(), TW_TYPE_FLOAT, &(*value).y, "");
}

void createTweak(TwBar* bar, const std::string& name, glm::vec3* value)
{
	TwAddVarRW(bar, name.c_str(), TW_TYPE_COLOR3F, value, "");
}

void createTweak(TwBar* bar, const std::string& name, glm::vec4* value)
{
	TwAddVarRW(bar, name.c_str(), TW_TYPE_COLOR4F, value, "");
}

void createTweak(TwBar* bar, const std::string& name, glm::mat2* value)
{
}

void createTweak(TwBar* bar, const std::string& name, glm::mat3* value)
{
}

void createTweak(TwBar* bar, const std::string& name, glm::mat4* value)
{
}

void createTweak(TwBar* bar, const std::string& name, int* value)
{
	TwAddVarRW(bar, name.c_str(), TW_TYPE_INT32, value, "");
}

void createTweak(TwBar* bar, const std::string& name, std::array<int, 2>* value)
{
}

void createTweak(TwBar* bar, const std::string& name, std::array<int, 3>* value)
{
}

void createTweak(TwBar* bar, const std::string& name, std::array<int, 4>* value)
{
}

void createTweak(TwBar* bar, const std::string& name, unsigned int* value)
{
	TwAddVarRW(bar, name.c_str(), TW_TYPE_UINT32, value, "");
}

void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 2>* value)
{
}

void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 3>* value)
{
}

void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 4>* value)
{
}

void createTweak(TwBar* bar, const std::string& name, float** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, glm::vec2** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, glm::vec3** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, glm::vec4** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, glm::mat2** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, glm::mat3** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, glm::mat4** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, int** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, std::array<int, 2>** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, std::array<int, 3>** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, std::array<int, 4>** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, unsigned int** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 2>** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 3>** value)
{ createTweak(bar, name, *value); }
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 4>** value)
{ createTweak(bar, name, *value); }

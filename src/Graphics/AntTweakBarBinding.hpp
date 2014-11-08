#pragma once

#include <array>

#define GLEW_STATIC
#include <GL/glew.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <AntTweakBar.h>

void createTweak(TwBar* bar, const std::string& name, float* value);
void createTweak(TwBar* bar, const std::string& name, glm::vec2* value);
void createTweak(TwBar* bar, const std::string& name, glm::vec3* value);
void createTweak(TwBar* bar, const std::string& name, glm::vec4* value);
void createTweak(TwBar* bar, const std::string& name, glm::mat2* value);
void createTweak(TwBar* bar, const std::string& name, glm::mat3* value);
void createTweak(TwBar* bar, const std::string& name, glm::mat4* value);
void createTweak(TwBar* bar, const std::string& name, int* value);
void createTweak(TwBar* bar, const std::string& name, std::array<int, 2>* value);
void createTweak(TwBar* bar, const std::string& name, std::array<int, 3>* value);
void createTweak(TwBar* bar, const std::string& name, std::array<int, 4>* value);
void createTweak(TwBar* bar, const std::string& name, unsigned int* value);
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 2>* value);
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 3>* value);
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 4>* value);

void createTweak(TwBar* bar, const std::string& name, float** value);
void createTweak(TwBar* bar, const std::string& name, glm::vec2** value);
void createTweak(TwBar* bar, const std::string& name, glm::vec3** value);
void createTweak(TwBar* bar, const std::string& name, glm::vec4** value);
void createTweak(TwBar* bar, const std::string& name, glm::mat2** value);
void createTweak(TwBar* bar, const std::string& name, glm::mat3** value);
void createTweak(TwBar* bar, const std::string& name, glm::mat4** value);
void createTweak(TwBar* bar, const std::string& name, int** value);
void createTweak(TwBar* bar, const std::string& name, std::array<int, 2>** value);
void createTweak(TwBar* bar, const std::string& name, std::array<int, 3>** value);
void createTweak(TwBar* bar, const std::string& name, std::array<int, 4>** value);
void createTweak(TwBar* bar, const std::string& name, unsigned int** value);
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 2>** value);
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 3>** value);
void createTweak(TwBar* bar, const std::string& name, std::array<unsigned int, 4>** value);

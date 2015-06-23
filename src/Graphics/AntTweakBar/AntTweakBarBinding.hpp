#pragma once

#include <array>

#include <GL/gl3w.h>

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

// Dummy function for const parameters
void createTweak(TwBar* bar, const std::string& name, const glm::vec2* value);
void createTweak(TwBar* bar, const std::string& name, const glm::vec3* value);
void createTweak(TwBar* bar, const std::string& name, const glm::vec4* value);
void createTweak(TwBar* bar, const std::string& name, const glm::mat2* value);
void createTweak(TwBar* bar, const std::string& name, const glm::mat3* value);
void createTweak(TwBar* bar, const std::string& name, const glm::mat4* value);

///////////////////////////////////////////////////////////////////
// Pointer versions

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

// Dummy function for const parameters
void createTweak(TwBar* bar, const std::string& name, const glm::vec2** value);
void createTweak(TwBar* bar, const std::string& name, const glm::vec3** value);
void createTweak(TwBar* bar, const std::string& name, const glm::vec4** value);
void createTweak(TwBar* bar, const std::string& name, const glm::mat2** value);
void createTweak(TwBar* bar, const std::string& name, const glm::mat3** value);
void createTweak(TwBar* bar, const std::string& name, const glm::mat4** value);
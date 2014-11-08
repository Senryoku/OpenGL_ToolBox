#pragma once

#include <array>

#define GLEW_STATIC
#include <GL/glew.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <Texture.hpp>
#include <CubeMap.hpp>

// Set of free functions (overloads) wrapping OpenGL calls

template<typename T>
void setUniform(GLint program, GLuint location, const T* value)
{
	setUniform(program, location, *value);
}

void setUniform(GLint program, GLuint location, const float& value);
void setUniform(GLint program, GLuint location, const glm::vec2& value);
void setUniform(GLint program, GLuint location, const glm::vec3& value);
void setUniform(GLint program, GLuint location, const glm::vec4& value);
void setUniform(GLint program, GLuint location, const glm::mat2& value);
void setUniform(GLint program, GLuint location, const glm::mat3& value);
void setUniform(GLint program, GLuint location, const glm::mat4& value);
void setUniform(GLint program, GLuint location, const int& value);
void setUniform(GLint program, GLuint location, const std::array<int, 2>& value);
void setUniform(GLint program, GLuint location, const std::array<int, 3>& value);
void setUniform(GLint program, GLuint location, const std::array<int, 4>& value);
void setUniform(GLint program, GLuint location, const unsigned int& value);
void setUniform(GLint program, GLuint location, const std::array<unsigned int, 2>& value);
void setUniform(GLint program, GLuint location, const std::array<unsigned int, 3>& value);
void setUniform(GLint program, GLuint location, const std::array<unsigned int, 4>& value);
void setUniform(GLint program, GLuint location, const Texture& value);
void setUniform(GLint program, GLuint location, const CubeMap& value);


void setUniform(GLuint location, const float& value);
void setUniform(GLuint location, const glm::vec2& value);
void setUniform(GLuint location, const glm::vec3& value);
void setUniform(GLuint location, const glm::vec4& value);
void setUniform(GLuint location, const glm::mat2& value);
void setUniform(GLuint location, const glm::mat3& value);
void setUniform(GLuint location, const glm::mat4& value);
void setUniform(GLuint location, const int& value);
void setUniform(GLuint location, const std::array<int, 2>& value);
void setUniform(GLuint location, const std::array<int, 3>& value);
void setUniform(GLuint location, const std::array<int, 4>& value);
void setUniform(GLuint location, const unsigned int& value);
void setUniform(GLuint location, const std::array<unsigned int, 2>& value);
void setUniform(GLuint location, const std::array<unsigned int, 3>& value);
void setUniform(GLuint location, const std::array<unsigned int, 4>& value);
void setUniform(GLuint location, const Texture& value);
void setUniform(GLuint location, const CubeMap& value);

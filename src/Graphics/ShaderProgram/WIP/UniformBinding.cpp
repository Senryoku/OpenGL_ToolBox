#include <UniformBinding.hpp>

#include <glm/gtc/type_ptr.hpp> // glm::value_ptr

#include <Material.hpp>

void setUniform(GLint program, GLuint location, const float& value)
{
	glProgramUniform1f(program, location, value);
}

void setUniform(GLint program, GLuint location, const glm::vec2& value)
{
	glProgramUniform2fv(program, location, 1, glm::value_ptr(value));
}

void setUniform(GLint program, GLuint location, const glm::vec3& value)
{
	glProgramUniform3fv(program, location, 1, glm::value_ptr(value));
}

void setUniform(GLint program, GLuint location, const glm::vec4& value)
{
	glProgramUniform4fv(program, location, 1, glm::value_ptr(value));
}

void setUniform(GLint program, GLuint location, const glm::mat2& value)
{
	glProgramUniformMatrix2fv(program, location, 1, false, glm::value_ptr(value));
}

void setUniform(GLint program, GLuint location, const glm::mat3& value)
{
	glProgramUniformMatrix3fv(program, location, 1, false, glm::value_ptr(value));
}

void setUniform(GLint program, GLuint location, const glm::mat4& value)
{
	glProgramUniformMatrix4fv(program, location, 1, false, glm::value_ptr(value));
}

void setUniform(GLint program, GLuint location, const int& value)
{
	glProgramUniform1i(program, location, value);
}

void setUniform(GLint program, GLuint location, const std::array<int, 2>& value)
{
	glProgramUniform2iv(program, location, 1, (const GLint*) value.data());
}

void setUniform(GLint program, GLuint location, const std::array<int, 3>& value)
{
	glProgramUniform3iv(program, location, 1, (const GLint*) value.data());
}

void setUniform(GLint program, GLuint location, const std::array<int, 4>& value)
{
	glProgramUniform4iv(program, location, 1, (const GLint*) value.data());
}

void setUniform(GLint program, GLuint location, const unsigned int& value)
{
	glProgramUniform1ui(program, location, value);
}

void setUniform(GLint program, GLuint location, const std::array<unsigned int, 2>& value)
{
	glProgramUniform2uiv(program, location, 1, (const GLuint*) value.data());
}

void setUniform(GLint program, GLuint location, const std::array<unsigned int, 3>& value)
{
	glProgramUniform2uiv(program, location, 1, (const GLuint*) value.data());
}

void setUniform(GLint program, GLuint location, const std::array<unsigned int, 4>& value)
{
	glProgramUniform2uiv(program, location, 1, (const GLuint*) value.data());
}

void setUniform(GLint program, GLuint location, GLuint textureUnit, const Texture& value)
{
	value.bind(textureUnit);
	glProgramUniform1i(program, location, textureUnit);
}

void setUniform(GLint program, GLuint location, GLuint textureUnit, const CubeMap& value)
{
	value.bind(textureUnit);
	glProgramUniform1i(program, location, textureUnit);
}

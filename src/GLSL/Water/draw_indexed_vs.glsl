#version 430 core

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

in  layout(location = 0) vec4 position;
in  layout(location = 1) vec4 normal;

out layout(location = 0) vec4 out_position;
out layout(location = 1) vec4 out_normal;

void main()
{
    gl_Position = ProjectionMatrix * ViewMatrix * position;
	out_position = position;
	out_normal = normal;
}
	
#version 430 core

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform unsigned int lightCount = 0;

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;
in layout(location = 3) mat4 ModelMatrix;

out layout(location = 0) vec3 world_position;
out layout(location = 1) vec3 world_normal;
out layout(location = 2) vec2 texcoord;

void main(void)
{
	vec4 P = ModelMatrix * vec4(in_position, 1.f);
    gl_Position =  ProjectionMatrix * ViewMatrix * P;
	
	world_position = P.xyz / P.w;
	world_normal = mat3(ModelMatrix) * in_normal;
	texcoord = in_texcoord;
}

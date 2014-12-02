#version 430 core

layout(location = 0)
uniform mat4 ModelViewMatrix;
layout(location = 1)
uniform mat4 ProjectionMatrix;
layout(location = 2)
uniform mat3 NormalMatrix;

uniform mat4 ModelMatrix = mat4(1.0);

uniform unsigned int lightCount = 0;

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
	//sampler2D	shadowmap;
} Lights[8];

uniform float Ns = 8.f;
uniform vec4 diffuse = vec4(0.3f, 0.3f, 0.3f, 1.f);
uniform vec4 Ka = vec4(0.2f, 0.2f, 0.2f, 1.f);

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;

out layout(location = 0) vec3 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;
out layout(location = 3) vec4 shadowcoord[8];

void main(void)
{
	vec4 P = ModelViewMatrix * vec4(in_position, 1.f);
    gl_Position = ProjectionMatrix * P;
	
	position = vec3(P);
	normal = normalize(NormalMatrix*in_normal);
	texcoord = in_texcoord;
	
	for(int i = 0; i < lightCount; ++i)
		shadowcoord[i] = Lights[i].depthMVP * ModelMatrix * vec4(in_position, 1.f);
}

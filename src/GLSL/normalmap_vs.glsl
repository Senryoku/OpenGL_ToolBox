#version 430 core

layout(location = 0)
uniform mat4 ModelViewMatrix;
layout(location = 1)
uniform mat4 ProjectionMatrix;
layout(location = 2)
uniform mat3 NormalMatrix;

uniform mat4 ModelMatrix = mat4(1.0);

uniform mat4 DepthMVP;

uniform vec3 lightPosition = vec3(25.f, 10.f, 25.f);
uniform float Ns = 8.f;
uniform vec4 diffuse = vec4(0.3f, 0.3f, 0.3f, 1.f);
uniform vec4 Ka = vec4(0.2f, 0.2f, 0.2f, 1.f);
uniform vec4 Ks = vec4(0.6f, 0.6f, 0.6f, 1.f);

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;

out layout(location = 0) struct VertexData
{
	vec3 position;
	vec3 normal;
	vec2 texcoord;
	vec4 shadowcoord;
} VertexOut;

void main(void)
{
	vec4 P = ModelViewMatrix * vec4(in_position, 1.f);
    gl_Position = ProjectionMatrix * P;
	
	VertexOut.position = vec3(P);
	VertexOut.normal = normalize(NormalMatrix*in_normal);
	VertexOut.texcoord = in_texcoord;
	
	VertexOut.shadowcoord = DepthMVP * ModelMatrix * vec4(in_position, 1.f);
}

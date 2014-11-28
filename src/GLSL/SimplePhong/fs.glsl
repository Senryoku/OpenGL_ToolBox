#version 430 core

layout(location = 0)
uniform mat4 ModelViewMatrix;
layout(location = 1)
uniform mat4 ProjectionMatrix;
layout(location = 2)
uniform mat3 NormalMatrix;

uniform vec3 lightPosition = vec3(25.f, 10.f, 25.f);

uniform float shininess = 8.f;
uniform vec4 diffuse = vec4(0.3f, 0.3f, 0.3f, 1.f);
uniform vec4 ambient = vec4(0.2f, 0.2f, 0.2f, 1.f);
uniform vec4 specular = vec4(0.6f, 0.6f, 0.6f, 1.f);

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;
 
out vec4 colorOut;

void main(void)
{
	vec3 N = normalize(in_normal);
	
	vec3 L = normalize(lightPosition - in_position);
	
	float diffuseFactor = max(dot(N, L), 0.f);
	
	vec3 V = normalize(-in_position);
	vec3 R = normalize(-reflect(L, N));
	
	float specularFactor = shininess > 0.f ?
								pow(max(dot(R, V), 0.f), shininess) :
								0.f;
		
	colorOut = ambient + diffuseFactor*diffuse + specularFactor*specular;
}

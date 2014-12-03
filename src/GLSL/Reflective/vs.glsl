#version 430 core

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
	mat3 NormalMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform vec3 cameraPosition = vec3(25.f, 10.f, 25.f);

uniform unsigned int lightCount = 0;

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
} Lights[8];

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;

out layout(location = 0) struct VertexData
{
	vec3 position;
	vec3 normal;
	vec2 texcoord;
	vec3 reflectDir;
	vec4 shadowcoord[8];
} VertexOut;

void main(void)
{
	vec4 P = ViewMatrix * ModelMatrix * vec4(in_position, 1.f);
    gl_Position = ProjectionMatrix * P;
	
	VertexOut.position = vec3(P);
	VertexOut.normal = normalize(NormalMatrix * in_normal);
	VertexOut.texcoord = in_texcoord;
	vec3 r = reflect(in_position - cameraPosition, in_normal);
	VertexOut.reflectDir = vec3(-r.x, r.y, -r.z); // Why ? Oô
	
	for(int i = 0; i < 8; ++i)
		VertexOut.shadowcoord[i] = Lights[i].depthMVP * ModelMatrix * vec4(in_position, 1.f);
}

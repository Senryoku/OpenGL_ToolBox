#version 430 core

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};


uniform unsigned int lightCount = 0;

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
} Lights[8];

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;
in layout(location = 3) mat4 ModelMatrix;

out layout(location = 0) vec3 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;
out layout(location = 3) vec4 shadowcoord[8];

flat out int instanceID;

void main(void)
{
	vec4 P = ViewMatrix * ModelMatrix * vec4(in_position, 1.f);
    gl_Position = ProjectionMatrix * P;
	
	position = vec3(P);
	// I guess I should precompute the Normal Matrix... Oh, well.
	normal = normalize(mat3(transpose(inverse(ViewMatrix * ModelMatrix))) * in_normal);
	texcoord = in_texcoord;
	
	for(int i = 0; i < lightCount; ++i)
		shadowcoord[i] = Lights[i].depthMVP * ModelMatrix * vec4(in_position, 1.f);
		
	instanceID = gl_InstanceID;
}

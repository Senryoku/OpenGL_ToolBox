#version 430

layout(points) in;
layout(triangle_strip) out;
layout(max_vertices = 4) out;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform float particle_size = 0.1;
uniform vec3 cameraPosition;

in layout(location = 0) vec4 in_position[1];

out layout(location = 0) vec4 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;

void main()
{
	mat4 VP = ProjectionMatrix * ViewMatrix;
	vec3 n = normalize(cameraPosition - in_position[0].xyz);
	vec3 up = vec3(0.0, 1.0, 0.0);
	vec3 right = 0.5 * particle_size * cross(up, n);
	up *= 0.5 * particle_size;
	
	gl_Position = VP * vec4(in_position[0].xyz - up - right, 1.0);
	position = in_position[0];
	normal = n;
	texcoord = vec2(0.0, 0.0);
	EmitVertex();
	
	gl_Position = VP * vec4(in_position[0].xyz - up + right, 1.0);
	position = in_position[0];
	normal = n;
	texcoord = vec2(1.0, 0.0);
	EmitVertex();
	
	gl_Position = VP * vec4(in_position[0].xyz + up - right, 1.0);
	position = in_position[0];
	normal = n;
	texcoord = vec2(0.0, 1.0);
	EmitVertex();
	
	gl_Position = VP * vec4(in_position[0].xyz + up + right, 1.0);
	position = in_position[0];
	normal = n;
	texcoord = vec2(1.0, 1.0);
	EmitVertex();
	
	EndPrimitive();
}

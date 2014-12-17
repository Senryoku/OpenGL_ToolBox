#version 430 core

#define Samples9

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform vec4 Color;

in layout(location = 0) vec3 world_position;
in layout(location = 1) vec3 world_normal;

out layout(location = 0) vec4 colorDepthOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

void main(void)
{	
	worldNormalOut.rgb = normalize(world_normal);
	worldNormalOut.a = 1.0;
	
	worldPositionOut.xyz = world_position;
	worldPositionOut.w = 1.0;
	
	colorDepthOut.rgb = Color.rgb;
	colorDepthOut.w = gl_FragCoord.z;
}

#version 430 core

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform sampler2D Texture;

in layout(location = 0) vec3 world_position;
in layout(location = 1) vec3 world_normal;
in layout(location = 2) vec2 texcoord;

out layout(location = 0) vec4 colorMaterialOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

void main(void)
{	
	worldNormalOut.rgb = normalize(world_normal);
	worldNormalOut.a = 1.0;
	
	worldPositionOut.xyz = world_position;
	worldPositionOut.w = gl_FragCoord.z;
	
	colorMaterialOut.rgb = texture(Texture, texcoord).rgb;
	colorMaterialOut.w = 0.0;
}

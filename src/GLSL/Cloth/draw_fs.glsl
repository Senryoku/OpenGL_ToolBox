#version 430

in layout(location = 0) vec4 in_position_fixed;
in layout(location = 1) vec4 in_speed_data1;
in layout(location = 2) vec3 normal;
in layout(location = 3) vec2 texcoord;

out layout(location = 0) vec4 colorMaterialOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

const float MATERIAL_UNLIT = 2.0;

void main()
{
	vec3 color = vec3(1.0, 0.0, 0.0);
	colorMaterialOut = vec4(color, MATERIAL_UNLIT);
	worldPositionOut = vec4(in_position_fixed.xyz, gl_FragCoord.z);
	worldNormalOut = vec4(normal, 1.0);
}

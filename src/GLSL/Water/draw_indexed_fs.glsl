#version 430

uniform samplerCube EnvMap;
uniform float reflectFactor = 0.8;
uniform vec3 color = vec3(0.4, 0.4, 0.8);
uniform float transparency = 0.8;

in layout(location = 0) vec4 in_position;
in layout(location = 1) vec4 in_normal;
in layout(location = 2) vec3 in_eye;

out layout(location = 0) vec4 colorMaterialOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

void main()
{	
	colorMaterialOut = vec4(color + reflectFactor * texture(EnvMap, reflect(in_eye, in_normal.xyz)).rgb, transparency);
	worldPositionOut.xyz = in_position.xyz;
	worldPositionOut.w = gl_FragCoord.z;
	worldNormalOut.xyz = normalize(in_normal.xyz);
	worldNormalOut.w = 0.0;
}

#version 430

in layout(location = 0) vec4 in_position;
in layout(location = 1) vec4 in_normal;

out layout(location = 0) vec4 colorMaterialOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

void main()
{	
	colorMaterialOut = vec4(0.2, 0.2, 0.8, 1.0);
	worldPositionOut.xyz = in_position.xyz;
	worldPositionOut.w = gl_FragCoord.z;
	worldNormalOut.xyz = normalize(in_normal.xyz);
	worldNormalOut.w = 0.0;
}

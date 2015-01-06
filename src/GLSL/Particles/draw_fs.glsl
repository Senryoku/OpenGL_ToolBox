#version 430

in layout(location = 0) vec4 in_position_type;
in layout(location = 1) vec4 in_speed_lifetime;

out layout(location = 0) vec4 colorDepthOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

void main()
{
	colorDepthOut = vec4(1.0, 0.0, 0.0, gl_FragCoord.z);
	worldPositionOut = vec4(in_position_type.xyz, 1.0);
	worldNormalOut = vec4(0.0, 0.0, 1.0, 1.0);
}

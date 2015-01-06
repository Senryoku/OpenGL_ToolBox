#version 430 core

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

in layout(location = 0) vec4 position_type;
in layout(location = 1) vec4 speed_lifetime;

out layout(location = 0) vec4 out_position_type;
out layout(location = 1) vec4 out_speed_lifetime;

void main()
{
	//gl_Position = ProjectionMatrix * ViewMatrix * vec4(position_type.xyz, 1.0);
	out_position_type = position_type;
    out_speed_lifetime = speed_lifetime;
}
	
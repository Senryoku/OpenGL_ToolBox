#version 430 core

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

in layout(location = 0) vec4 position_fixed;
in layout(location = 1) vec4 speed_data1;

out layout(location = 0) vec4 out_position_fixed;
out layout(location = 1) vec4 out_speed_data1;

void main()
{
	out_position_fixed = position_fixed;
    out_speed_data1 = speed_data1;
}
	
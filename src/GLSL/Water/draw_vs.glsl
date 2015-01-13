#version 430 core

uniform int size_x;
uniform int size_y;
uniform float cell_size = 1.0;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

in layout(location = 0) vec4 data;

out layout(location = 0) vec4 out_position;

void main()
{
	out_position = vec4(cell_size * (gl_VertexID / size_x), 
						1.0 + data.x + data.y, 
						cell_size * (gl_VertexID % size_y), 
						1.0);
}
	
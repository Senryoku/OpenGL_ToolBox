#version 430

layout(points) in;
layout(triangle_strip) out;
layout(max_vertices = 6) out;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform int size_x = 200;
uniform int size_y = 200;
uniform float cell_size;

in layout(location = 0) vec4 in_position[];
in layout(location = 1) int in_id[];

out layout(location = 0) vec4 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;

struct WaterCell
{
	vec4 data; // Water Height, Ground Height, Speed(xy)
};

layout(std140, binding = 4) buffer InBuffer
{
	WaterCell	Ins[];
};

vec2 o[4] = {vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, -1.0)};

vec2 to2D(int idx)
{
	return vec2(idx / size_x, idx % size_y);
}

int to1D(vec2 c)
{
	return int(c.x * size_y + c.y);
}

bool valid(vec2 c)
{
	return c.x >= 0 && c.y >= 0 &&
		c.x < size_x && c.y < size_y;
}

vec3 computeNormal(vec2 coord)
{
	vec3 neighbors[4];
	for(int i = 0; i < 4; ++i)
	{
		vec2 c = coord + o[i];
		if(!valid(c)) c = coord;
		neighbors[i].xz = c * cell_size;
		neighbors[i].y = Ins[to1D(c)].data.x;
	}
	
	return normalize(cross((neighbors[1] - neighbors[3])/(2.0 * cell_size), (neighbors[2] - neighbors[0])/(2.0 * cell_size)));
}

vec2 computeTexcoord(vec2 coord)
{
	return vec2(coord.x / size_x, coord.y / size_y);
}

void main()
{
	mat4 VP = ProjectionMatrix * ViewMatrix;
	vec2 coord = to2D(in_id[0]);
	vec3 pos;
	pos.xz = coord * cell_size;
	pos.y = Ins[to1D(coord)].data.x;
	pos = vec3(ModelMatrix * vec4(pos, 1.0));
	
	vec3 neighbors[4];
	vec3 neighbors_normal[4];
	for(int i = 0; i < 4; ++i)
	{
		vec2 c = coord + o[i];
		if(!valid(c)) c = coord;
		neighbors[i].xz = c * cell_size;
		neighbors[i].y = Ins[to1D(c)].data.x;
		neighbors[i] = vec3(ModelMatrix * vec4(neighbors[i], 1.0));
		neighbors_normal[i] = computeNormal(c);
	}
	
	vec3 n = normalize(cross((neighbors[1] - neighbors[3])/(2.0 * cell_size), (neighbors[2] - neighbors[0])/(2.0 * cell_size)));
	
	if(coord.x > 0 && coord.y < size_y - 1)
	{
		gl_Position = VP * vec4(pos, 1.0);
		position = vec4(pos, 1.0);
		normal = n;
		texcoord = computeTexcoord(coord);
		EmitVertex();
		
		gl_Position = VP * vec4(neighbors[0], 1.0);
		position = vec4(neighbors[0], 1.0);
		normal = neighbors_normal[0];
		texcoord = computeTexcoord(coord + o[0]);
		EmitVertex();
		
		gl_Position = VP * vec4(neighbors[1], 1.0);
		position = vec4(neighbors[1], 1.0);
		normal = neighbors_normal[1];
		texcoord = computeTexcoord(coord + o[1]);
		EmitVertex();
		
		EndPrimitive();
	}
	
	if(coord.y > 0 && coord.x < size_x - 1)
	{
		gl_Position = VP * vec4(pos, 1.0);
		position = vec4(pos, 1.0);
		normal = n;
		texcoord = computeTexcoord(coord);
		EmitVertex();
		
		gl_Position = VP * vec4(neighbors[3], 1.0);
		position = vec4(neighbors[3], 1.0);
		normal = neighbors_normal[3];
		texcoord = computeTexcoord(coord + o[3]);
		EmitVertex();
		
		gl_Position = VP * vec4(neighbors[2], 1.0);
		position = vec4(neighbors[2], 1.0);
		normal = neighbors_normal[2];
		texcoord = computeTexcoord(coord + o[2]);
		EmitVertex();
		
		EndPrimitive();
	}
}

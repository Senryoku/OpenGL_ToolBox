#version 430

layout(points) in;
layout(triangle_strip) out;
layout(max_vertices = 12) out;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform int size_x = 200;
uniform int size_y = 200;
uniform float cell_size;

uniform bool draw_ground = true;

in layout(location = 0) vec4 in_position[];
in layout(location = 1) int in_id[];

out layout(location = 0) vec4 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;
flat out layout(location = 3) float ground;

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

void computeNormals(vec2 coord, out vec3 n, out vec3 n_g)
{
	vec3 neighbors[4];
	vec3 neighbors_g[4];
	for(int i = 0; i < 4; ++i)
	{
		vec2 c = coord + o[i];
		if(!valid(c)) c = coord;
		vec2 h = Ins[to1D(c)].data.xy;
		neighbors[i].xz = c * cell_size;
		neighbors[i].y = h.x;
		if(draw_ground)
		{
			neighbors_g[i].xz = c * cell_size;
			neighbors_g[i].y = h.y;
		}
	}
	
	n = normalize(cross((neighbors[1] - neighbors[3]), (neighbors[2] - neighbors[0])));
	if(draw_ground)
		n_g = normalize(cross((neighbors_g[1] - neighbors_g[3]), (neighbors_g[2] - neighbors_g[0])));
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
	
	vec3 pos_g;
	pos_g.xz = coord * cell_size;
	pos_g.y = Ins[to1D(coord)].data.y;
	pos_g = vec3(ModelMatrix * vec4(pos_g, 1.0));
	
	vec3 neighbors[4];
	vec3 neighbors_normal[4];
	vec3 neighbors_g[4];
	vec3 neighbors_normal_g[4];
	
	bool b0 = (coord.x > 0 && coord.y < size_y - 1);
	bool b1 = (coord.y > 0 && coord.x < size_x - 1);
	
	for(int i = 0; i < 4; ++i)
	{
		vec2 c = coord + o[i];
		if(!valid(c)) c = coord;
		vec2 h = Ins[to1D(c)].data.xy;
		
		neighbors[i].xz = c * cell_size;
		neighbors[i].y = h.x;
		
		neighbors[i] = vec3(ModelMatrix * vec4(neighbors[i], 1.0));
		
		if(draw_ground)
		{
			neighbors_g[i].xz = c * cell_size;
			neighbors_g[i].y = h.y;
			neighbors_g[i] = vec3(ModelMatrix * vec4(neighbors_g[i], 1.0));
		}
			
		if(!b0 && i < 2) continue;
		if(!b1 && i >= 2) continue;
		computeNormals(c, neighbors_normal[i], neighbors_normal_g[i]);
	}
	
	vec3 n = normalize(cross((neighbors[1] - neighbors[3]), (neighbors[2] - neighbors[0])));
	vec3 n_g;
	if(draw_ground) n_g = normalize(cross((neighbors_g[1] - neighbors_g[3]), (neighbors_g[2] - neighbors_g[0])));
	
	if(b0)
	{
		gl_Position = VP * vec4(neighbors[0], 1.0);
		position = vec4(neighbors[0], 1.0);
		normal = neighbors_normal[0];
		texcoord = computeTexcoord(coord + o[0]);
		ground = 0.0;
		EmitVertex();
		
		gl_Position = VP * vec4(neighbors[1], 1.0);
		position = vec4(neighbors[1], 1.0);
		normal = neighbors_normal[1];
		texcoord = computeTexcoord(coord + o[1]);
		ground = 0.0;
		EmitVertex();
	}
	
	if(b0 || b1)
	{		
		gl_Position = VP * vec4(pos, 1.0);
		position = vec4(pos, 1.0);
		normal = n;
		texcoord = computeTexcoord(coord);
		ground = 0.0;
		EmitVertex();
	}
	
	if(b1)
	{		
		gl_Position = VP * vec4(neighbors[3], 1.0);
		position = vec4(neighbors[3], 1.0);
		normal = neighbors_normal[3];
		texcoord = computeTexcoord(coord + o[3]);
		ground = 0.0;
		EmitVertex();
		
		gl_Position = VP * vec4(neighbors[2], 1.0);
		position = vec4(neighbors[2], 1.0);
		normal = neighbors_normal[2];
		texcoord = computeTexcoord(coord + o[2]);
		ground = 0.0;
		EmitVertex();
		
		EndPrimitive();
	} else if(b0) {
		EndPrimitive();
	}
	
	// Ground
	if(draw_ground)
	{
		if(b0)
		{			
			gl_Position = VP * vec4(neighbors_g[0], 1.0);
			position = vec4(neighbors_g[0], 1.0);
			normal = neighbors_normal_g[0];
			texcoord = computeTexcoord(coord + o[0]);
			ground = 1.0;
			EmitVertex();
			
			gl_Position = VP * vec4(neighbors_g[1], 1.0);
			position = vec4(neighbors_g[1], 1.0);
			normal = neighbors_normal_g[1];
			texcoord = computeTexcoord(coord + o[1]);
			ground = 1.0;
			EmitVertex();
		}
		
		if(b0 || b1)
		{
			gl_Position = VP * vec4(pos_g, 1.0);
			position = vec4(pos_g, 1.0);
			normal = n_g;
			texcoord = computeTexcoord(coord);
			ground = 1.0;
			EmitVertex();
		}
		
		if(b1)
		{			
			gl_Position = VP * vec4(neighbors_g[3], 1.0);
			position = vec4(neighbors_g[3], 1.0);
			normal = neighbors_normal_g[3];
			texcoord = computeTexcoord(coord + o[3]);
			ground = 1.0;
			EmitVertex();
			
			gl_Position = VP * vec4(neighbors_g[2], 1.0);
			position = vec4(neighbors_g[2], 1.0);
			normal = neighbors_normal_g[2];
			texcoord = computeTexcoord(coord + o[2]);
			ground = 1.0;
			EmitVertex();
			
			EndPrimitive();
		} else if(b0) {
			EndPrimitive();
		}
	}
}

#version 430

uniform int iterations = 10;
uniform float time = 0.0;
uniform int size_x = 200;
uniform int size_y = 200;
uniform float cell_size;
uniform float moyheight = 2.0;
uniform vec3 acceleration = vec3(0.0, -9.0, 1.0);
uniform float damping = 0.1;

ivec2 coord;
	
struct WaterCell
{
	vec4 data; // Water Height, Ground Height, Speed(xy)
};

layout(std140, binding = 4) buffer InBuffer
{
	WaterCell	Ins[];
};

const ivec2 offsets[4] = {             ivec2(0, -1), 
						  ivec2(-1, 0),             ivec2(1, 0),
						               ivec2(0,  1)};

int to1D(ivec2 c)
{
	return c.y * size_x + c.x;
}

float interpolate(vec2 c, float v00, float v01, float v10, float v11)
{
	vec2 v = vec2(v00 * (1.0 - c.y) + v01 * c.y, v10 * (1.0 - c.y) + v11 * c.y);
	return v.x * (1.0 - c.x) + v.y * c.x;
}

vec3 interpolate(vec2 c, vec3 v00, vec3 v01, vec3 v10, vec3 v11)
{
	vec3 v[2] = {v00 * (1.0 - c.y) + v01 * c.y, v10 * (1.0 - c.y) + v11 * c.y};
	return v[0] * (1.0 - c.x) + v[1] * c.x;
}

bool valid(vec2 c)
{
	return (c.x >= 0 && c.y >= 0 && c.x < size_x && c.y < size_y);
}
/*
bool inWorkgroup(ivec2 c)
{
	return c / gl_WorkGroupSize.xy == coord / gl_WorkGroupSize.xy;
}

shared vec4 neighbors[16][16];

vec4 get(ivec2 c)
{
	if(inWorkgroup(c))
		return neighbors[c.x % gl_WorkGroupSize.x][c.y % gl_WorkGroupSize.y];
	else 
		return Ins[to1D(c)].data;
}
*/
layout (local_size_x = 16, local_size_y = 16) in;
void main()
{
	float t = time / iterations;
	coord = ivec2(gl_GlobalInvocationID.xy);
	int idx = to1D(coord);
	bool inbound = coord.x < size_x && coord.y < size_y && idx < size_x * size_y;

	//if(inbound)
	//	neighbors[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = Ins[idx].data;
	barrier();
	
	for(int it = 0; it < iterations; ++it)
	{
		vec4 local;
		if(inbound) // Advect water height (data.x) and velocity (data.zw)
		{
			// Ground
			// ... Ground is constant.
			
			vec2 mod_coord = coord - t * Ins[to1D(coord)].data.zw / cell_size;
			vec2 fract_mod_coord = fract(mod_coord);
			
			vec4 v00;
			vec4 v01;
			vec4 v10;
			vec4 v11;
			
			ivec2 trunc_coord;
			trunc_coord = ivec2(mod_coord);
			if(!valid(trunc_coord))
				v00 = vec4(moyheight, 0.0, 0.0, 0.0);
			else
				v00 = Ins[to1D(trunc_coord)].data;
				//v00 = get(trunc_coord);
				
			trunc_coord = ivec2(mod_coord + vec2(0.0, 1.0));
			if(!valid(trunc_coord))
				v01 = vec4(moyheight, 0.0, 0.0, 0.0);
			else
				v01 = Ins[to1D(trunc_coord)].data;
				
			trunc_coord = ivec2(mod_coord + vec2(1.0, 0.0));
			if(!valid(trunc_coord))
				v10 = vec4(moyheight, 0.0, 0.0, 0.0);
			else
				v10 = Ins[to1D(trunc_coord)].data;
				
			trunc_coord = ivec2(mod_coord + vec2(1.0, 1.0));
			if(!valid(trunc_coord))
				v11 = vec4(moyheight, 0.0, 0.0, 0.0);
			else
				v11 = Ins[to1D(trunc_coord)].data;
			
			vec3 local = interpolate(fract_mod_coord, v00.xzw, v01.xzw, v10.xzw, v11.xzw);
			
			Ins[idx].data.xzw = local;
			//neighbors[gl_LocalInvocationID.x][gl_LocalInvocationID.y].xzw = local;
		}
		
		barrier();
		
		if(inbound)
		{
			// Update Height
			vec2 grad;
			
			//local = neighbors[gl_LocalInvocationID.x][gl_LocalInvocationID.y];
			local = Ins[idx].data;
			if(coord.x == size_x - 1)
				grad.x = 0.0 - local.z;
			else
				grad.x = Ins[to1D(ivec2(coord.x + 1, coord.y))].data.z - local.z;
				
			if(coord.y == size_y - 1)
				grad.y = 0.0 - local.w;
			else
				grad.y = Ins[to1D(ivec2(coord.x, coord.y + 1))].data.w - local.w;
			
			grad = grad / cell_size;
				
			float div = grad.x + grad.y;
			local.x -= local.x * t * div;
			Ins[idx].data.x = local.x;
			//neighbors[gl_LocalInvocationID.x][gl_LocalInvocationID.y].x = local.x;
		}
		
		barrier();
		
		if(inbound)
		{
			// Update velocities, works on Water Height (.x) + Ground Height (.y)
			float h = local.x + local.y;
			float h2 = moyheight;
			if(coord.x > 0)
				h2 = Ins[to1D(ivec2(coord.x - 1, coord.y))].data.x + Ins[to1D(ivec2(coord.x - 1, coord.y))].data.y;
				
			float h3 = moyheight;
			if(coord.y > 0)
				h3 = Ins[to1D(ivec2(coord.x, coord.y - 1))].data.x + Ins[to1D(ivec2(coord.x, coord.y - 1))].data.y;
			
			local.zw *= (1.0 - damping*t);
			
			local.z += 9.81 * ( (h2 - h) / cell_size ) * t;
			local.w += 9.81 * ( (h3 - h) / cell_size ) * t;
			
			Ins[idx].data = local;
			//neighbors[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = local;
		}
	}
}

#version 430

uniform float time = 0.0;
uniform int size_x = 200;
uniform int size_y = 200;
uniform float cell_size;
uniform float moyheight = 2.0;
uniform vec3 acceleration = vec3(0.0, -9.0, 1.0);

struct WaterCell
{
	vec4 data; // Water Height, Ground Height, Speed(xy)
};

layout(std140, binding = 4) buffer InBuffer
{
	WaterCell	Ins[];
};

layout(std140, binding = 5) buffer OutBuffer
{
	WaterCell	Outs[];
};

const ivec2 offsets[4] = {             ivec2(0, -1), 
						  ivec2(-1, 0),             ivec2(1, 0),
						               ivec2(0,  1)};

int to1D(ivec2 coord)
{
	return coord.y * size_x + coord.x;
}

float interpolate(vec2 coord, float v00, float v01, float v10, float v11)
{
	vec2 v = vec2(v00 * (1.0 - coord.y) + v01 * coord.y, v10 * (1.0 - coord.y) + v11 * coord.y);
	return v.x * (1.0 - coord.x) + v.y * coord.x;
}

bool valid(vec2 coord)
{
	return (coord.x >= 0 && coord.y >= 0 && coord.x < size_x && coord.y < size_y);
}

layout (local_size_x = 16, local_size_y = 16) in;
void main()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	int idx = to1D(coord);
	bool inbound = coord.x < size_x && coord.y < size_y && idx < size_x * size_y;
	
	if(inbound)
	{
		// Ground
		Outs[idx].data.y = Ins[idx].data.y; // Ground is constant.
		
		// Advect water height (data.x)
		vec2 mod_coord = coord - time * Ins[idx].data.zw / cell_size;
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
		
		Outs[idx].data.x = interpolate(fract_mod_coord, v00.x, v01.x, v10.x, v11.x);
		Outs[idx].data.z = interpolate(fract_mod_coord, v00.z, v01.z, v10.z, v11.z);
		Outs[idx].data.w = interpolate(fract_mod_coord, v00.w, v01.w, v10.w, v11.w);
	}
	
	barrier();
	if(inbound)
	{
		// Update Height
		vec2 grad;
		
		if(coord.x == size_x - 1)
			grad.x = 0.0 - Outs[idx].data.z;
		else
			grad.x = Outs[to1D(ivec2(coord.x + 1, coord.y))].data.z - Outs[idx].data.z;
			
		if(coord.y == size_y - 1)
			grad.y = 0.0 - Outs[idx].data.w;
		else
			grad.y = Outs[to1D(ivec2(coord.x, coord.y + 1))].data.w - Outs[idx].data.w;
		
		grad = grad / cell_size;
			
		float div = grad.x + grad.y;
		Outs[idx].data.x -= Outs[idx].data.x * time * div;
	}
	
	barrier();
	if(inbound)
	{
		// Update velocities, works on Water Height (.x) + Ground Height (.y)
		float h = Outs[idx].data.x + Outs[idx].data.y;
		float h2 = moyheight + Outs[idx].data.y;
		if(coord.x > 0)
			h2 = Outs[to1D(ivec2(coord.x - 1, coord.y))].data.x + Outs[to1D(ivec2(coord.x - 1, coord.y))].data.y;
			
		float h3 = moyheight + Outs[idx].data.y;
		if(coord.y > 0)
			h3 = Outs[to1D(ivec2(coord.x, coord.y - 1))].data.x + Outs[to1D(ivec2(coord.x, coord.y - 1))].data.y;
		
		Outs[idx].data.z += 9.81 * ( (h2 - h) / cell_size ) * time;
		Outs[idx].data.w += 9.81 * ( (h3 - h) / cell_size ) * time;
	}
}

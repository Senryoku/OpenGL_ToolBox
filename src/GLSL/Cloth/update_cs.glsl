#version 430

uniform float time = 0.0;
uniform int width = 250;
uniform int height = 250;
uniform float cellsize = 0.1;
uniform int iterations = 50;
uniform vec3 acceleration = vec3(0.0, -9.0, 1.0);

struct ClothPoint
{
	vec4 position_fixed;
	vec4 speed_data1;
};

layout(std140, binding = 4) buffer InBuffer
{
	ClothPoint	Ins[];
};

layout(std140, binding = 5) buffer OutBuffer
{
	ClothPoint	Outs[];
};

const ivec2 offsets[4] = {             ivec2(0, -1), 
						  ivec2(-1, 0),             ivec2(1, 0),
						               ivec2(0,  1)};

int to1D(ivec2 coord)
{
	return coord.y * width + coord.x;
}
									   
layout (local_size_x = 16, local_size_y = 16) in;
void main()
{
	ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
	int idx = to1D(coord);
	bool inbound = coord.x < width && coord.y < height && idx < width * height;
	if(inbound)
	{
		// Apply acceleration and speed
		float fixed_pos = Ins[idx].position_fixed.w;
		Outs[idx].position_fixed.w = fixed_pos;
		Outs[idx].speed_data1.w = Ins[idx].speed_data1.w;
		if(fixed_pos > 0)
		{
			vec3 speed = 0.99 * Ins[idx].speed_data1.xyz + acceleration * time;
			Outs[idx].position_fixed.xyz = Ins[idx].position_fixed.xyz + speed * time;
			Outs[idx].speed_data1.xyz = speed;
		} else {
			Outs[idx].position_fixed.xyz = Ins[idx].position_fixed.xyz;
			Outs[idx].speed_data1.xyz = vec3(0.0);
		}
		
		if(Outs[idx].position_fixed.y < 0)
			Outs[idx].position_fixed.y = 0;
	}
	
	barrier();
		
	// Apply constraints
	for(int step = 0; step < iterations; ++step)
	{
		vec3 tmp;
		if(inbound)
		{
			tmp = Outs[idx].position_fixed.xyz;
			float fixed_pos = Outs[idx].position_fixed.w;
			if(fixed_pos > 0)
			{
				for(int i = 0; i < 4; ++i)
				{
					ivec2 other = ivec2(coord) + offsets[i];
					if(other.y < height && other.y >= 0 &&
					   other.x < width && other.x >= 0)
					{
						int otheridx = to1D(other);
						if(otheridx >= 0  && otheridx < width * height)
						{
							vec3 v = Outs[otheridx].position_fixed.xyz - Outs[idx].position_fixed.xyz;

							float l = length(v);
							float factor = (l - cellsize);
							if(Outs[otheridx].position_fixed.w > 0.0) factor *= 0.5;
							if(l != 0.0 && abs(factor) > 0.001)
								tmp += (factor / l) * v; 
						}
					}
				}	
			}
		}
		barrier();
		if(inbound)
			Outs[idx].position_fixed.xyz = tmp;
		barrier();
	}
}

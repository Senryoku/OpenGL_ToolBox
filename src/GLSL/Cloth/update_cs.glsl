#version 430

uniform float time = 0.0;
uniform int width = 250;
uniform int height = 250;
uniform float cellsize = 0.1;
uniform int iterations = 1;
uniform int constraints_iterations = 50;
uniform float damping = 0.00;
uniform vec3 acceleration = vec3(0.0, -9.81, 1.0);

struct ClothPoint
{
	vec4 position_fixed;
	vec4 oldpos_data1;
};

layout(std140, binding = 4) buffer InBuffer
{
	ClothPoint	Ins[];
};

const ivec2 offsets[4] = {             ivec2(0, -1), 
						  ivec2(-1, 0),             ivec2(1, 0),
						               ivec2(0,  1)};
									   
const ivec2 offsets_diag[4] = {ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1)};

int to1D(ivec2 coord)
{
	return coord.y * width + coord.x;
}
									   
layout (local_size_x = 16, local_size_y = 16) in;
void main()
{
	float t = time / float(iterations);
	ClothPoint In;
	for(int it = 0; it < iterations; ++it)
	{
		ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
		int idx = to1D(coord);
		bool inbound = coord.x < width && coord.y < height && idx < width * height;
		if(inbound)
		{
			In = Ins[idx];
			Ins[idx].oldpos_data1.xyz = In.position_fixed.xyz;
			Ins[idx].oldpos_data1.w = t;
			// Apply acceleration and speed
			float fixed_pos = In.position_fixed.w;
			if(fixed_pos > 0)
			{
				vec3 speed = (1.0 - damping * t) * (In.position_fixed.xyz - In.oldpos_data1.xyz) + acceleration * t;
				Ins[idx].position_fixed.xyz = In.position_fixed.xyz + speed * t;
			} // Does nothing if fixed
			
			if(In.position_fixed.y < 0)
				Ins[idx].position_fixed.y = 0;
		}
		
		barrier();
			
		// Apply constraints
		for(int step = 0; step < constraints_iterations; ++step)
		{
			vec3 tmp;
			if(inbound)
			{
				ClothPoint CurrVal = Ins[idx];
				tmp = CurrVal.position_fixed.xyz;
				float fixed_pos = CurrVal.position_fixed.w;
				if(fixed_pos > 0)
				{
					for(int i = 0; i < 4; ++i)
					{
						ivec2 other = ivec2(coord) + offsets[i];
						if(other.y < height && other.y >= 0 &&
						   other.x < width && other.x >= 0)
						{
							int otheridx = to1D(other);
							vec3 v = Ins[otheridx].position_fixed.xyz - CurrVal.position_fixed.xyz;

							float l = length(v);
							float factor = 1.0;
							if(Ins[otheridx].position_fixed.w > 0.0) factor = 0.5;
							if(l != 0.0)
								tmp += factor * (1.0 - cellsize / l) * v; 
						}
					}	
					//  Diagonales
					/*
					float diag_length = sqrt(cellsize*cellsize + cellsize*cellsize);
					for(int i = 0; i < 4; ++i)
					{
						ivec2 other = ivec2(coord) + offsets_diag[i];
						if(other.y < height && other.y >= 0 &&
						   other.x < width && other.x >= 0)
						{
							int otheridx = to1D(other);
							vec3 v = Ins[otheridx].position_fixed.xyz - CurrVal.position_fixed.xyz;

							float l = length(v);
							float factor = 1.0;
							if(Ins[otheridx].position_fixed.w > 0.0) factor = 0.5;
							if(l != 0.0)
								tmp += factor * (1.0 - diag_length / l) * v; 
						}
					}
					*/
				}
			}
			barrier();
			if(inbound)
			{
				if(tmp.y < 0.0f) tmp.y = 0.0f;
				Ins[idx].position_fixed.xyz = tmp;
			}
			barrier();
		}
	}
}

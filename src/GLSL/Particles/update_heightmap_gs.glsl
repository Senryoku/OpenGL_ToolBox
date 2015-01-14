#version 430

layout(points) in;
layout(points, max_vertices = 1) out;

struct HeightmapCell
{
	vec4 data; // x: Height
};

uniform int size_x = 200;
uniform int size_y = 200;
uniform float cell_size;
uniform mat4 HeightmapModelMatrix = mat4(1.0);
layout(std140, binding = 4) buffer InBuffer
{
	HeightmapCell	Ins[];
};

vec2 to2D(int idx)
{
	return vec2(idx / size_x, idx % size_y);
}

int to1D(vec2 c)
{
	return int(int(c.x) * size_y + c.y);
}

bool valid(vec2 c)
{
	return c.x >= 0 && c.y >= 0 &&
		c.x < size_x && c.y < size_y;
}

vec2 o[4] = {vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, -1.0)};
vec2 o8[8] = {vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, -1.0),
			  vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)};

vec3 computeNormalHeightmap(vec2 coord)
{
	vec3 neighbors[4];
	for(int i = 0; i < 4; ++i)
	{
		vec2 c = coord + o[i];
		if(!valid(c)) c = coord;
		neighbors[i].xz = c * cell_size;
		neighbors[i].y = Ins[to1D(c)].data.x;
	}
	
	return normalize(cross((neighbors[1] - neighbors[3])/(2.0 * cell_size), normalize(neighbors[2] - neighbors[0])/(2.0 * cell_size)));
}

vec3 computeNormalGroundHeightmap(vec2 coord)
{
	vec3 neighbors[4];
	for(int i = 0; i < 4; ++i)
	{
		vec2 c = coord + o[i];
		if(!valid(c)) c = coord;
		neighbors[i].xz = c * cell_size;
		neighbors[i].y = Ins[to1D(c)].data.y;
	}
	
	return normalize(cross((neighbors[1] - neighbors[3])/(2.0 * cell_size), normalize(neighbors[2] - neighbors[0])/(2.0 * cell_size)));
}

uniform float time = 0.0;
uniform float particle_size = 0.25;

uniform float buoyancy = 1.1;

const float SphereDragCoef = 0.47;

const float WaterDensity = 998.2071;
const float AirDensity = 1.2041;

layout(location = 0) in vec4 inter_position_type[];
layout(location = 1) in vec4 inter_speed_lifetime[];

layout(location = 0) out vec4 position_type;
layout(location = 1) out vec4 speed_lifetime;

float rand(vec2 co)
{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
	float lifetime = inter_speed_lifetime[0].w - time;
	
	vec3 old_pos = inter_position_type[0].xyz;
	
	if(lifetime <= 0.0)
	{
		position_type.xyz = vec3(0.0, 10.0, 0.0);
		position_type.w = inter_position_type[0].w;
		speed_lifetime.xyz = 2.0 * vec3(-2.5 + 5.0 * rand(vec2(lifetime, inter_position_type[0].z)), 
								  2.0 + 3.0 * rand(inter_position_type[0].yx),
								  -2.5 + 5.0 * rand(vec2(inter_position_type[0].x, lifetime)));
		speed_lifetime.w = 20.0 * rand(inter_position_type[0].xy);
	} else {
		vec3 speed = inter_speed_lifetime[0].xyz + vec3(0.0, -9.81, 0.0) * time;
		position_type.xyz = inter_position_type[0].xyz + speed * time;
		position_type.w = inter_position_type[0].w;
		speed_lifetime.xyz = speed;
		speed_lifetime.w = lifetime;
	
		float Density = AirDensity;
		const vec2 coord = (inverse(HeightmapModelMatrix) * vec4(position_type.xyz, 1.0)).xz / cell_size;
		if(valid(coord))
		{
			float alt = Ins[to1D(coord)].data.x;
			// Under Water
			if(position_type.y - particle_size * 0.5 < alt)
			{
				float displ = clamp((alt - (position_type.y - particle_size * 0.5)), 0.0, alt);
				speed_lifetime.xyz += vec3(0.0, buoyancy * clamp(displ / particle_size, 0.0, 1.0) * 9.81, 0.0) * time;
				Density = WaterDensity;
				// Just came under.
				if(old_pos.y + particle_size * 0.5 > alt)
				{
					Ins[to1D(coord)].data.x -= displ;
					for(int i = 0; i < 8; ++i)
					{
						vec2 c = coord + o8[i];
						if(!valid(c)) c = coord;
						Ins[to1D(c)].data.x += displ * 0.125;
					}
				}
			} else { // Above Water
				float displ = (position_type.y - particle_size * 0.5) - alt;
				// Just moved out!
				if(old_pos.y - particle_size * 0.5 < alt)
				{
					Ins[to1D(coord)].data.x += displ;
					for(int i = 0; i < 8; ++i)
					{
						vec2 c = coord + o8[i];
						if(!valid(c)) c = coord;
						Ins[to1D(c)].data.x -= displ * 0.125;
					}
				}
			}
		
			if(position_type.y - particle_size * 0.5 < Ins[to1D(coord)].data.y)
			{
				speed_lifetime.xyz = 0.9 * reflect(speed_lifetime.xyz, computeNormalGroundHeightmap(coord));
				position_type.y = Ins[to1D(coord)].data.y + particle_size * 0.5;
			}
		}
		
		if(position_type.y - particle_size * 0.5 < 0.0)
		{
			speed_lifetime.xyz = 0.9 * reflect(speed_lifetime.xyz, vec3(0.0, 1.0, 0.0));
			position_type.y = particle_size * 0.5;
		}
		
		// Drag Force
		const float CrossSectionArea = (3.1415926 * particle_size * particle_size * 0.25);
		float s = length(speed_lifetime.xyz);
		speed_lifetime.xyz -= (0.5 * Density * CrossSectionArea * SphereDragCoef * s * s * time) * normalize(speed_lifetime.xyz);
	}
	
	EmitVertex();
	EndPrimitive();
}

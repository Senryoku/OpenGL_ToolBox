#version 430

struct LightStruct
{
	vec4		position;
	vec4		color;
};

layout(std140) uniform LightBlock
{
	LightStruct	Lights[1000];
};
uniform unsigned int lightCount = 0;
uniform float lightRadius = 100.0;

uniform float	minDiffuse = 0.0;

uniform vec3	cameraPosition;

uniform layout(binding = 0) image2D ColorDepth;
uniform layout(binding = 1) sampler2D Position;
uniform layout(binding = 2) sampler2D Normal;

// Bounding Box
shared int min_depth = 1000;
shared int max_depth = 0;
shared vec3 min_pixel;
shared vec3 max_pixel;

// Lights
shared int local_lights_count = 0;
shared int local_lights[1000];

void add_light(int l)
{
	int idx = atomicAdd(local_lights_count, 1);
	local_lights[idx] = l;
}

layout (local_size_x = 32, local_size_y = 32) in;
void main(void)
{
	uvec2 pixel = gl_GlobalInvocationID.xy;
	uvec2 local_pixel = gl_LocalInvocationID.xy;

	// Compute Bounding Box
	int depth = int(imageLoad(ColorDepth, pixel).w);
	atomicMin(min_depth, depth);
	atomicMax(max_depth, depth);
	if(local_pixel == uvec2(0, 0))
		min_pixel = pixel;
	else if(local_pixel == uvec2(31, 1))
		max_pixel = pixel;
	barrier();
	// ... Construct BBox
	
	// Test lights
	// We have to test 1000 lights tops, and 32*32 = 1024 pixels per tile
	// so let's have each of them test only one light :)
	int l = local_pixel.x * 32 + local_pixel.y; 
	if(l < lightCount)
	{
		bool light_passed = false;
		// ...
		if(light_passed)
			add_light(l);
	}
	barrier();
	
	//Compute lights' contributions
}

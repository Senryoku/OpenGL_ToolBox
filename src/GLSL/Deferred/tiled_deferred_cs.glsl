#version 430

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

struct LightStruct
{
	vec4		position;
	vec4		color;
};

layout(std140) uniform LightBlock
{
	LightStruct	Lights[1024];
};
uniform unsigned int lightCount = 0;
uniform float lightRadius = 100.0;

uniform float	minDiffuse = 0.0;

uniform vec3	cameraPosition;

layout(binding = 0, rgba32f) uniform image2D ColorDepth;
layout(binding = 1, rgba32f) uniform readonly image2D Position;
layout(binding = 2, rgba32f) uniform readonly image2D Normal;

// Bounding Box
shared int min_depth; // = 1000;
shared int max_depth; // = 0;
shared vec2 min_pixel;
shared vec2 max_pixel;

// Lights
shared int local_lights_count; // = 0;
shared int local_lights[1024];

void add_light(int l)
{
	int idx = atomicAdd(local_lights_count, 1);
	local_lights[idx] = l;
}

bool sphereAABBIntersect(vec3 min, vec3 max, vec3 center, float radius)
{
    float r = radius * radius;
    if (center.x < min.x) r -= pow(center.x - min.x, 2.0);
    else if (center.x > max.x) r -= pow(center.x - max.x, 2.0);
    if (center.y < min.y) r -= pow(center.y - min.y, 2.0);
    else if (center.y > max.y) r -= pow(center.y - max.y, 2.0);
    if (center.z < min.z) r -= pow(center.z - min.y, 2.0);
    else if (center.z > max.z) r -= pow(center.z - max.z, 2.0);
    return r > 0;
}

vec3 phong(vec3 p, vec3 N, vec3 diffuse, vec3 lp, vec3 lc)
{
	vec3 L = normalize(lp - p);
	float dNL = dot(N, L);
	
	float diffuseFactor = max(dNL, minDiffuse);
	
	vec3 V = normalize(cameraPosition - p);
	vec3 R = normalize(reflect(-L, N));
	
	float specularFactor = pow(max(dot(R, V), 0.f), 8.0);
								
	return diffuseFactor * diffuse * lc + specularFactor * lc;
}

layout (local_size_x = 32, local_size_y = 32) in;
void main(void)
{
	uvec2 pixel = gl_GlobalInvocationID.xy;
	uvec2 local_pixel = gl_LocalInvocationID.xy;
	ivec2 image_size = imageSize(ColorDepth).xy;
	
	// DEBUG
	if(pixel.x > image_size.x || pixel.y > image_size.y)
		return;
	imageStore(ColorDepth, ivec2(pixel), vec4(1.0, 0.0, 0.0, 1.0));
	return;
	/*

	// Compute Bounding Box
	
	int depth = int(imageLoad(ColorDepth, ivec2(pixel)).w);
	
	atomicMin(min_depth, depth);
	atomicMax(max_depth, depth);
	
	if(local_pixel == uvec2(0, 0))
		min_pixel = 2.0 * (vec2(pixel)/image_size) - 1.0;
	else if(local_pixel == uvec2(31, 31))
		max_pixel = 2.0 * (vec2(pixel)/image_size) - 1.0;
	
	barrier();
	
	// ... Construct AABB
	mat4 inverseProjView = inverse(ProjectionMatrix * ViewMatrix); // Should be precomputed
	vec4 min = inverseProjView * vec4(min_pixel.x, min_pixel.y, min_depth, 1.0);
	vec4 max = inverseProjView * vec4(max_pixel.x, max_pixel.y, max_depth, 1.0);
	
	// Test lights
	// We have to test 1024 lights tops, and 32*32 = 1024 pixels per tile
	// so let's have each of them test only one light :)
	int l = int(local_pixel.x * 32 + local_pixel.y); 
	if(l < lightCount)
	{
		if(sphereAABBIntersect(min.xyz, max.xyz, Lights[l].position.xyz, lightRadius))
			add_light(l);
	}
	barrier();
	
	//Compute lights' contributions
	vec3 color = imageLoad(ColorDepth, ivec2(pixel)).xyz;
	vec3 position = imageLoad(Position, ivec2(pixel)).xyz;
	vec3 normal = normalize(imageLoad(Normal, ivec2(pixel)).xyz);
	
	vec4 ColorOut = vec4(0.0, 0.0, 0.0, 1.0);
	for(int l2 = 0; l2 < local_lights_count; ++l2)
	{
		float d = length(position - Lights[local_lights[l2]].position.xyz);
		if(d < lightRadius)
			ColorOut.rgb += (1.0 - d/lightRadius) * phong(position, normal, color, Lights[local_lights[l2]].position.xyz, Lights[local_lights[l2]].color.rgb);
	}
	imageStore(ColorDepth, ivec2(pixel), ColorOut);
	*/
}

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
uniform unsigned int lightCount = 75;
uniform float lightRadius = 100.0;

uniform float	minDiffuse = 0.0;

uniform vec3	cameraPosition;

layout(binding = 0, rgba32f) uniform image2D ColorDepth;
layout(binding = 1, rgba32f) uniform readonly image2D Position;
layout(binding = 2, rgba32f) uniform readonly image2D Normal;

// Bounding Box
shared int min_depth; // = 1000;
shared int max_depth; // = 0;
shared vec4 min_bbox;
shared vec4 max_bbox;

shared int min_x;
shared int min_y;
shared int min_z;
shared int max_x;
shared int max_y;
shared int max_z;

// Lights
shared int local_lights_count; // = 0;
shared int local_lights[1024];

void add_light(int l)
{
	int idx = atomicAdd(local_lights_count, 1);
	local_lights[idx] = l;
}

float square(float f)
{
	return f * f;
}

bool sphereAABBIntersect(vec3 min, vec3 max, vec3 center, float radius)
{
    float r = radius * radius;
    if(center.x < min.x) r -= square(center.x - min.x);
    else if(center.x > max.x) r -= square(center.x - max.x);
    if(center.y < min.y) r -= square(center.y - min.y);
    else if(center.y > max.y) r -= square(center.y - max.y);
    if(center.z < min.z) r -= square(center.z - min.z);
    else if(center.z > max.z) r -= square(center.z - max.z);
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
	
	bool isVisible = pixel.x >= 0 && pixel.y >= 0 && pixel.x < uint(image_size.x) && pixel.y < image_size.y;
	vec4 coldepth;
	
	if(local_pixel == uvec2(0, 0))
	{
		min_depth = 100000;
		max_depth = -100000;
		local_lights_count = 0;
		
		// TEST (WHAAAAAAAT ?)
		/*
		ivec3 position = ivec3(imageLoad(Position, ivec2(pixel)).xyz);
		min_x = position.x;
		max_x = position.x;
		min_y = position.y;
		max_y = position.y;
		min_z = position.z;
		max_z = position.z;
		*/
	}
	barrier();
		
	// Compute Bounding Box
	if(isVisible)
	{
		coldepth = imageLoad(ColorDepth, ivec2(pixel));
		/*
		int depth = int(coldepth.w);
		
		atomicMin(min_depth, depth);
		atomicMax(max_depth, depth);
		*/
		
		// TEST
		ivec3 position = ivec3(imageLoad(Position, ivec2(pixel)).xyz);
		atomicMin(min_x, position.x);
		atomicMax(max_x, position.x);
		atomicMin(min_y, position.y);
		atomicMax(max_y, position.y);
		atomicMin(min_z, position.z);
		atomicMax(max_z, position.z);
	}
	barrier();
	
	if(local_pixel == uvec2(0, 0))
	{
		// Construct AABB
		// Doesn't work T_T
		/*
		mat4 inverseProjView = inverse(ViewMatrix) * inverse(ProjectionMatrix); // Should be precomputed
		vec2 min_pixel = 2.0 * (vec2(pixel)/image_size) - 1.0;
		vec2 max_pixel = 2.0 * (vec2(pixel + uvec2(31, 31))/image_size) - 1.0;
		min_bbox = inverseProjView * vec4(min_pixel.x, min_pixel.y, min_depth * 0.001, 1.0);
		max_bbox = inverseProjView * vec4(max_pixel.x, max_pixel.y, max_depth * 0.001, 1.0);
		min_bbox /= min_bbox.w;
		max_bbox /= max_bbox.w;
		*/
		
		// TEST
		min_bbox = vec4(min_x, min_x, min_x, 1.0);
		max_bbox = vec4(max_x, max_x, max_x, 1.0);
	}
	barrier();
	
	if(isVisible)
	{
		// Test lights
		// We have to test 1024 lights tops, and 32*32 = 1024 pixels per tile
		// so let's have each of them test only one light :) 
		// (May cause problems for screen edges. I'll think it through later.)
		int l = int(local_pixel.x * 32 + local_pixel.y); 
		if(l < lightCount)
		{
			if(sphereAABBIntersect(min_bbox.xyz, max_bbox.xyz, Lights[l].position.xyz, lightRadius))
				add_light(l);
		}
	}
	barrier();
	
	//Compute lights' contributions
	if(isVisible)
	{
		vec3 color = coldepth.xyz;
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
		
		// DEBUG
		//imageStore(ColorDepth, ivec2(pixel), vec4(float(local_lights_count) * 0.1, 0.0, 0.0, 1.0));
	}
}

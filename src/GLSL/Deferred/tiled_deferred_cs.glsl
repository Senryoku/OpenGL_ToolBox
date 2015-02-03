#version 430

//#define SSAO

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

// SSAO
#ifdef SSAO
uniform float SamplingRadius = 20.0;
uniform float ScreenRadius = 20.0;
uniform vec2 Attenuation = vec2(1.0);

layout(binding = 3) uniform sampler2D PositionDepthSampler;
#endif

layout(binding = 0, rgba32f) uniform image2D ColorMaterial;
layout(binding = 1, rgba32f) uniform readonly image2D PositionDepth;
layout(binding = 2, rgba32f) uniform readonly image2D Normal;

// Bounding Box
shared int min_x;
shared int min_y;
shared int min_z;
shared int max_x;
shared int max_y;
shared int max_z;

// Lights
shared int local_lights_count; // = 0;
shared int local_lights[1024];

const float MATERIAL_UNLIT = 2.0;

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
	
	float specularFactor = pow(max(dot(R, V), 0.f), 64.0);
								
	return diffuseFactor * diffuse * lc + specularFactor * lc;
}

float rand(vec2 co)
{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

#ifdef SSAO
float ao_pixel(vec3 srcPosition, vec3 srcNormal, vec2 uv)
{
	vec4 dstPositionDepth = texture(PositionDepthSampler, uv);
	vec3 positionVec = dstPositionDepth.xyz - srcPosition;
	float bias = 0.3 + dstPositionDepth.w * 0.5;
	float intensity = max(dot(normalize(positionVec), srcNormal) - bias, 0.0);
	float dist = length(positionVec) / SamplingRadius;
	float attenuation = 1.0 / (Attenuation.x + (Attenuation.y * dist));
	return intensity * attenuation;
}

float ao(vec3 position, vec3 normal, vec2 uv, float depth)
{
	float ret = 0.0;
	const vec2 offset[4] = {vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(0.0, -1.0), vec2(-1.0, 0.0)};
	const float rot = sin(3.14159 / 4);
	const float offset_length = ScreenRadius * (1.0 - depth);
	for(int i = 0; i < 4; ++i)
	{
		vec2 k[2];
		k[0] = reflect(offset[i], normalize(vec2(rand(vec2(position.xy)), rand(vec2(position.zy)))));
		k[1] = vec2(k[0].x * rot - k[0].y * rot,
					k[0].x * rot + k[0].y * rot);
		k[0] /= textureSize(PositionDepthSampler, 0).xy;
		k[1] /= textureSize(PositionDepthSampler, 0).xy;
		for(int j = 0; j < 4; ++j)
		{
			vec2 tuv = uv + offset_length / (j + 1) * k[j % 2];
			if(all(lessThanEqual(tuv, vec2(1.0))) && all(lessThanEqual(vec2(0.0), tuv)))
				ret += ao_pixel(position, normal, tuv);
		}
	}	
	return clamp(ret / 16.0, 0.0, 1.0);
}
#endif

const int highValue = 1000000;

layout (local_size_x = 16, local_size_y = 16) in;
void main(void)
{
	uvec2 pixel = gl_GlobalInvocationID.xy;
	uvec2 local_pixel = gl_LocalInvocationID.xy;
	ivec2 image_size = imageSize(ColorMaterial).xy;
	
	bool isVisible = pixel.x >= 0 && pixel.y >= 0 && pixel.x < uint(image_size.x) && pixel.y < image_size.y;
	vec4 colmat;
	vec4 position;
	
	if(local_pixel == uvec2(0, 0))
	{
		local_lights_count = 0;
		
		min_x = highValue;
		max_x = -highValue;
		min_y = highValue;
		max_y = -highValue;
		min_z = highValue;
		max_z = -highValue;
	}
	barrier();
		
	// Compute Bounding Box
	if(isVisible)
	{
		colmat = imageLoad(ColorMaterial, ivec2(pixel));
		position = imageLoad(PositionDepth, ivec2(pixel));
		
		isVisible = isVisible && position.w > 0.0 && position.w < 1.0;
		
		if(isVisible && colmat.w != MATERIAL_UNLIT)
		{
			atomicMin(min_x, int(position.x));
			atomicMax(max_x, int(position.x + 1.0));
			atomicMin(min_y, int(position.y));
			atomicMax(max_y, int(position.y + 1.0));
			atomicMin(min_z, int(position.z));
			atomicMax(max_z, int(position.z + 1.0));
		}
	}
	barrier();
	
	// Construct boundingbox
	vec3 min_bbox = vec3(min_x, min_y, min_z);
	vec3 max_bbox = vec3(max_x, max_y, max_z);

	// Test lights
	for(int i = 0; i < int(lightCount); i += int(gl_WorkGroupSize.x) * int(gl_WorkGroupSize.y))
	{
		int l = int(gl_LocalInvocationIndex) + i;
		if(l < lightCount)
		{
			if(sphereAABBIntersect(min_bbox, max_bbox, Lights[l].position.xyz, lightRadius))
				add_light(l);
		}
	}
	barrier();
	
	//Compute lights' contributions
	if(isVisible && colmat.w != MATERIAL_UNLIT)
	{
		vec3 color = colmat.xyz;
		vec3 normal = normalize(imageLoad(Normal, ivec2(pixel)).xyz);
		
		vec4 ColorOut = vec4(0.0, 0.0, 0.0, position.w);
		for(int l2 = 0; l2 < local_lights_count; ++l2)
		{
			float d = length(position.xyz - Lights[local_lights[l2]].position.xyz);
			if(d < lightRadius)
				ColorOut.rgb += (1.0 - d/lightRadius) * phong(position.xyz, normal, color, Lights[local_lights[l2]].position.xyz, Lights[local_lights[l2]].color.rgb);
		}
		#ifdef SSAO
		ColorOut.rgb = pow(1.0 - clamp(ao(position.xyz, normal, pixel/vec2(image_size), colmat.w), 0.0, 1.0), 3.0) * clamp(ColorOut.rgb, 0.0, 1.0);
		#endif
		imageStore(ColorMaterial, ivec2(pixel), ColorOut);
		
		// DEBUG SSAO
		//imageStore(ColorMaterial, ivec2(pixel), 0.05 + vec4(vec3(ao(position.xyz, normal, pixel/vec2(image_size), colmat.w)), 1.0));
		
		// DEBUG Tiled Lightning
		//imageStore(ColorMaterial, ivec2(pixel), vec4(float(local_lights_count) / lightCount, 0.0, 0.0, 1.0));
	}
}

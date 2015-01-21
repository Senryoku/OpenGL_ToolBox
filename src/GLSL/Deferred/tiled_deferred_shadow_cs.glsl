#version 430

const float MATERIAL_UNLIT = 2.0;

struct LightStruct
{
	vec4		position;
	vec4		color;
};

layout(std140, binding = 1) uniform LightBlock
{
	LightStruct	Lights[1024];
};

layout(std140, binding = 2) uniform ShadowBlock
{
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
} Shadows[8];

uniform unsigned int lightCount = 75;
uniform float lightRadius = 100.0;

uniform unsigned int shadowCount = 0;

uniform float	minDiffuse = 0.0;
uniform float	bias = 0.00001f;

uniform vec3	cameraPosition;

layout(binding = 0, rgba32f) uniform image2D ColorMaterial;
layout(binding = 1, rgba32f) uniform readonly image2D PositionDepth;
layout(binding = 2, rgba32f) uniform readonly image2D Normal;

layout(binding = 3) uniform sampler2D ShadowMaps[8];

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
shared int lit;

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

vec3 phong(vec3 p, vec3 N, vec3 V, vec3 diffuse, vec3 lp, vec3 lc, bool t)
{
	vec3 L = normalize(lp - p);
	float dNL = dot(N, L);
	if(t && dNL < 0.0)
	{
		N = -N;
		dNL = -dNL;
	}
	
	float diffuseFactor = max(dNL, minDiffuse);
	
	vec3 R = normalize(reflect(-L, N));
	float specularFactor = pow(max(dot(R, V), 0.f), 64.0);
								
	return diffuseFactor * diffuse * lc + specularFactor * lc;
}

const int highValue = 2147483647;
const float boxfactor = 1000.0f; // Minimize the impact of the use of int for bounding boxes

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
		lit = 0;
	}
	barrier();
		
	// Compute Bounding Box
	if(isVisible)
	{
		position = imageLoad(PositionDepth, ivec2(pixel));
		isVisible = isVisible && position.w > 0.0 && position.w < 1.0;
		if(isVisible)
		{
			colmat = imageLoad(ColorMaterial, ivec2(pixel));
			
			if(isVisible && colmat.w != MATERIAL_UNLIT)
			{
				atomicMin(min_x, int(boxfactor * position.x - 1.0));
				atomicMax(max_x, int(boxfactor * position.x + 1.0));
				atomicMin(min_y, int(boxfactor * position.y - 1.0));
				atomicMax(max_y, int(boxfactor * position.y + 1.0));
				atomicMin(min_z, int(boxfactor * position.z - 1.0));
				atomicMax(max_z, int(boxfactor * position.z + 1.0));
				
				lit = 1;
			}
		}
	}
	barrier();
	
	// Construct boundingbox
	if(lit > 0)
	{
		vec3 min_bbox = vec3(min_x, min_y, min_z) / boxfactor;
		vec3 max_bbox = vec3(max_x, max_y, max_z) / boxfactor;

		// Test lights
		for(int i = 0; i < lightCount; i += gl_WorkGroupSize.x * gl_WorkGroupSize.y)
		{
			int l = int(gl_LocalInvocationIndex) + i;
			if(l < lightCount)
			{
				if(sphereAABBIntersect(min_bbox, max_bbox, Lights[l].position.xyz, lightRadius))
					add_light(l);
			}
		}
	}
	
	barrier();
	
	//Compute lights' contributions
	if(lit > 0 && isVisible && colmat.w != MATERIAL_UNLIT)
	{
		vec3 color = colmat.xyz;
		vec3 normal = normalize(imageLoad(Normal, ivec2(pixel)).xyz);
		
		vec4 ColorOut = vec4(0.0, 0.0, 0.0, 1.0);
	
		vec3 V = normalize(cameraPosition - position.xyz);
		
		bool transparent = colmat.a > 0.0 && colmat.a < 1.0;
		if(transparent)
			ColorOut.a = colmat.a;
		
		for(int l2 = 0; l2 < local_lights_count; ++l2)
		{
			float d = length(position.xyz - Lights[local_lights[l2]].position.xyz);
			if(d < lightRadius)
				ColorOut.rgb += (1.0 - square(d/lightRadius)) * phong(position.xyz, normal, V, color, Lights[local_lights[l2]].position.xyz, Lights[local_lights[l2]].color.rgb, transparent);
		}
		
		for(int shadow = 0; shadow < shadowCount; ++shadow)
		{
			vec4 sc = Shadows[shadow].depthMVP * vec4(position.xyz, 1.0);
			if((sc.x/sc.w >= 0 && sc.x/sc.w <= 1.f) &&
				(sc.y/sc.w >= 0 && sc.y/sc.w <= 1.f) && ((sc.x/sc.w * 2.0 - 1.0)*(sc.x/sc.w * 2.0 - 1.0) + (sc.y/sc.w * 2.0 - 1.0)*(sc.y/sc.w * 2.0 - 1.0) < 1.0))
			{				
				if(textureProj(ShadowMaps[shadow], sc.xyw).z + bias >= sc.z/sc.w)
				{
					// Ok, WHAT the actual FUCK ?! Inlining seem to fail here... compiler bug ? Oô
					//ColorOut.rgb += phong(position.xyz, normal, color, Shadows[shadow].position.xyz, Shadows[shadow].color.rgb);
					
					// So... this is just the call to phong(...) but inlined by hand...
					
					vec3 tmp_normal = normal;
					vec3 L = normalize(Shadows[shadow].position.xyz - position.xyz);
					float dNL = dot(tmp_normal, L);
					if(transparent && dNL < 0)
					{
						tmp_normal *= -1.0;
						dNL = -dNL;
					}
					
					float diffuseFactor = max(dNL, minDiffuse);
					
					vec3 R = normalize(reflect(-L, tmp_normal));

					float specularFactor = pow(max(dot(R, V), 0.f), 8.0);
												
					ColorOut.rgb += diffuseFactor * color * Shadows[shadow].color.rgb + specularFactor * Shadows[shadow].color.rgb;
				}
			}
		}
		
		imageStore(ColorMaterial, ivec2(pixel), ColorOut);
		
		// DEBUG
		//imageStore(ColorMaterial, ivec2(pixel), vec4(float(local_lights_count) / lightCount, 0.0, 0.0, 1.0));
	}
}

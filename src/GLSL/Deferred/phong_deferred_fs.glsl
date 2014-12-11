#version 430 core

uniform vec3	iResolution;	// viewport resolution (in pixels)
uniform float	minDiffuse = 0.1;

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform unsigned int lightCount = 0;

struct LightStruct
{
	vec4		position;
	vec4		color;
};

layout(std430) uniform LightBlock
{
	LightStruct	Lights[1000];
};

uniform layout(binding = 0) sampler2D ColorDepth;
uniform layout(binding = 1) sampler2D Position;
uniform layout(binding = 2) sampler2D Normal;

in vec2 texcoords;

vec3 phong(vec3 p, vec3 N, vec3 diffuse, vec3 L, vec3 lc)
{
	float dNL = dot(N, L);
	
	float diffuseFactor = max(dNL, minDiffuse);
	
	vec3 V = normalize(-p);
	vec3 R = normalize(-reflect(L, N));
	
	float specularFactor = pow(max(dot(R, V), 0.f), 8.0);
								
	return diffuseFactor * diffuse * lc + specularFactor * lc;
}

out vec4 colorOut;
void main(void)
{
	vec4 colDepth = texture(ColorDepth, texcoords);
	vec4 normal = texture(Normal, texcoords);
	vec4 position = texture(Position, texcoords);

	colorOut.rgb = colDepth.rgb; //vec3(0.0, 0.0, 0.0);
	for(int l = 0; l < lightCount; ++l)
	{
		colorOut.rgb += phong(position.xyz, normal.xyz, colDepth.rgb, Lights[l].position.xyz, Lights[l].color.rgb);
	}
	
	colorOut.rgb += phong(position.xyz, normal.xyz, colDepth.rgb, vec3(0.0, 50.0, 0.0), vec3(1.0));
	
	colorOut.a = 1.0;
}

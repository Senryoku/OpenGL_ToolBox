#version 430 core

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

uniform layout(binding = 0) sampler2D Color;
uniform layout(binding = 1) sampler2D Position;
uniform layout(binding = 2) sampler2D Normal;

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

in vec2 texcoords;
out vec4 colorOut;

void main(void)
{
	vec4 In0 = texture(Color, texcoords);
	vec4 In1 = texture(Position, texcoords);
	vec4 In2 = texture(Normal, texcoords);

	vec3 color = In0.xyz;
	vec3 position = In1.xyz;
	vec3 normal = normalize(In2.xyz);

	colorOut.rgb = vec3(0.0, 0.0, 0.0);
	for(int l = 0; l < lightCount; ++l)
	{
		colorOut.rgb += clamp(1.0 - length(position - Lights[l].position.xyz)/lightRadius, 0.0, 1.0) * phong(position, normal, color, Lights[l].position.xyz, Lights[l].color.rgb);
	}
	
	colorOut.a = 1.0;
}

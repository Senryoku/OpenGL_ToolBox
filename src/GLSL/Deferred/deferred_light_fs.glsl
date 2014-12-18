#version 430 core

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};
/*
struct LightStruct
{
	vec4		position;
	vec4		color;
};

layout(std140) uniform LightBlock
{
	LightStruct	Light;
};
*/
uniform vec4 LightPosition;
uniform vec4 LightColor;

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

out vec4 colorOut;

void main(void)
{
	vec2 texcoords = gl_FragCoord.xy/textureSize(Color, 0).xy;
	vec4 In1 = texture(Position, texcoords);
	vec3 position = In1.xyz;
	
	float d = length(position - LightPosition.xyz);
	if(d > lightRadius)
		discard;

	vec4 In0 = texture(Color, texcoords);
	vec4 In2 = texture(Normal, texcoords);

	vec3 color = In0.rgb;
	vec3 normal = normalize(In2.xyz);

	colorOut = vec4(0.0, 0.0, 0.0, 1.0);
	colorOut.rgb = (1.0 - d/lightRadius) * phong(position, normal, color, LightPosition.xyz, LightColor.rgb);
}

#version 430 core

layout(location = 0)
uniform mat4 ModelViewMatrix;

uniform float Ns = 8.f;
uniform vec3 lightPosition = vec3(25.f, 10.f, 25.f);
uniform vec4 Ka = vec4(0.2f, 0.2f, 0.2f, 1.f);
uniform vec4 Ks = vec4(0.6f, 0.6f, 0.6f, 1.f);
uniform vec4 diffuse = vec4(0.1, 0.1, 0.1, 1.0);

uniform float bias = 0.000005f;

in layout(location = 0) vec3 position;
in layout(location = 1) vec3 normal;
in layout(location = 2) vec2 texcoord;
in layout(location = 3) vec4 shadowcoord;
in layout(location = 4) vec3 reflectDir;

uniform layout(binding = 0) samplerCube EnvMap;
uniform layout(binding = 1) sampler2D ShadowMap;

uniform int poissonSamples = 4;
uniform float poissonDiskRadius = 2500.f;
uniform vec2 poissonDisk[16] = vec2[]
(
  vec2(0, 0),
  vec2(-0.94201624, -0.39906216),
  vec2(0.94558609, -0.76890725),
  vec2(-0.094184101, -0.92938870),
  vec2(0.34495938, 0.29387760),
  vec2(-0.7995101009,	0.8886048820),
  vec2(0.2934734351,	-0.2760867724),
  vec2(0.2341774684,	0.8865435198),
  vec2(0.8728931057,	-0.1874143791),
  vec2(-0.9215307223,	0.0450047820),
  vec2(-0.5043259108,	0.5018383689),
  vec2(0.3657662904,	-0.0115255959),
  vec2(0.0116447277,	0.7612487287),
  vec2(-0.0966746207,	0.4331707379),
  vec2(-0.9557312367,	0.2478491994),
  vec2(0.4434516553,	-0.7824516751)
);

float random(vec4 seed4)
{
	float dot_product = dot(seed4, vec4(12.9898,78.233,45.164,94.673));
    return fract(sin(dot_product) * 43758.5453)*2.f - 1.f;
}

out vec4 colorOut;
void main(void)
{
	vec3 N = normalize(normal);
	vec3 L = normalize((ModelViewMatrix * vec4(lightPosition, 1.0)).xyz - position);
	
	float dNL = dot(N, L);
	
	float bbias = bias * tan(acos(dNL));
	
	float visibility = 1.0f,
		   specular_visibility = 1.f;
		   
	// If we are in the light's fustrum...
	if((shadowcoord.x/shadowcoord.w  >= 0 && shadowcoord.x/shadowcoord.w  <= 1.f) &&
    (shadowcoord.y/shadowcoord.w  >= 0 && shadowcoord.y/shadowcoord.w  <= 1.f))
	{
		vec4 sc;
		for (int i = 0; i < poissonSamples; ++i)
		{
			sc = shadowcoord;
			sc.xy+= poissonDisk[i] * sc.w/poissonDiskRadius;
			if(textureProj(ShadowMap, sc.xyw).z + bbias < sc.z/sc.w)
			{
				visibility -= 0.6f / poissonSamples;
				specular_visibility = 0.f;
			}
		}
	} else {
		visibility = 0.4f;
		specular_visibility = 0.f;
	}
	
	float diffuseFactor = max(dNL, 0.f);
	
	vec3 V = normalize(-position);
	vec3 R = normalize(-reflect(L, N));
	
	float specularFactor = Ns > 0.f ?
								pow(max(dot(R, V), 0.f), Ns) :
								0.f;
		
	vec4 reflectColor = texture(EnvMap, reflectDir);
	colorOut = Ka*reflectColor + visibility*diffuseFactor*diffuse*reflectColor + specular_visibility*specularFactor*Ks;
	colorOut.w = diffuse.w;
}

#version 430 core

layout(location = 0)
uniform mat4 ModelViewMatrix;

uniform float minDiffuse = 0.0f;
uniform float Ns = 8.f;

uniform unsigned int lightCount = 0;

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
	//sampler2D	shadowmap;
} Lights[8];

uniform vec4 Ka = vec4(0.2f, 0.2f, 0.2f, 1.f);

uniform float bias = 0.000005f;

in layout(location = 0) vec3 position;
in layout(location = 1) vec3 normal;
in layout(location = 2) vec2 texcoord;
in layout(location = 3) vec4 shadowcoord[8];

uniform layout(binding = 0) sampler2D Texture;
uniform layout(binding = 1) sampler2D NormalMap;
uniform layout(binding = 2) sampler2D ShadowMap[8];

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

vec4 phong(vec3 p, vec3 n, vec4 diffuse, vec3 lp, vec4 lc)
{
	vec3 L = normalize((ModelViewMatrix * vec4(lp, 1.0)).xyz - p);
	float dNL = dot(n, L);
	
	float diffuseFactor = max(dNL, minDiffuse);
	
	vec3 V = normalize(-p);
	vec3 R = normalize(-reflect(L, n));
	
	float specularFactor = Ns > 0.f ?
								pow(max(dot(R, V), 0.f), Ns) :
								0.f;
								
	return diffuseFactor * diffuse * lc + specularFactor * lc;
}

out vec4 colorOut;
void main(void)
{	
	vec3 tang;
	vec3 tmp0 = cross(normal, vec3(0.f, 0.f, 1.f));
	vec3 tmp1 = cross(normal, vec3(0.f, 1.f, 0.f));
	if(length(tmp1) == 0.0)
		tang = normalize(tmp0);	
	else
		tang = normalize(tmp1);	
	vec3 bitang = normalize(cross(normal, tang));
	
	mat3 TangentToWorldSpace = inverse(transpose(mat3(tang, bitang, normal)));
	
	vec3 N2 = 2.0f * vec3(texture(NormalMap, texcoord)) - 1.0f;
	vec3 N = normalize(TangentToWorldSpace  * N2);
		
	vec4 diffuse = texture(Texture, texcoord);
	colorOut = diffuse * Ka;
	
	for(int l = 0; l < lightCount; ++l)
	{
		vec3 L = normalize((ModelViewMatrix * vec4(Lights[l].position.xyz, 1.0)).xyz - position);
		
		float dNL = dot(N, L);
		
		float bbias = bias * tan(acos(dNL));
		
		float visibility = 1.0f,
			  specular_visibility = 1.f;
		   
		vec4 sc = shadowcoord[l];
		if((sc.x/sc.w  >= 0 && sc.x/sc.w  <= 1.f) &&
		   (sc.y/sc.w  >= 0 && sc.y/sc.w  <= 1.f))
		{
			for (int i = 0; i < poissonSamples; ++i)
			{
				sc.xy += poissonDisk[i] * sc.w/poissonDiskRadius;
				if(textureProj(ShadowMap[l], sc.xyw).z + bbias < sc.z/sc.w)
				{
					visibility -= (1.0f - minDiffuse) / poissonSamples;
					specular_visibility = 0.f;
				}
			}
		} else {
			visibility = minDiffuse;
			specular_visibility = 0.f;
		}
		colorOut += visibility * phong(position, N, diffuse, Lights[l].position.xyz, Lights[l].color);
	}
	colorOut.w = diffuse.w;
}

#version 430 core

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform float minDiffuse = 0.0f;

uniform float Ns = 8.f;
uniform vec4 Ka = vec4(0.2f, 0.2f, 0.2f, 1.f);
uniform vec4 diffuse = vec4(1.0, 1.0, 1.0, 1.0);

uniform float bias = 0.000005f;

uniform unsigned int lightCount = 0;

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
} Lights[8];

in layout(location = 0) vec3 position;
in layout(location = 1) vec3 normal;
in layout(location = 2) vec2 texcoord;
in layout(location = 3) vec3 reflectDir;
in layout(location = 4) vec4 shadowcoord[8];

uniform layout(binding = 0) samplerCube EnvMap;
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

vec4 phong(vec3 p, vec3 N, vec4 diffuse, vec3 L, vec4 lc)
{
	float dNL = dot(N, L);
	
	float diffuseFactor = max(dNL, minDiffuse);
	
	vec3 V = normalize(-p);
	vec3 R = normalize(-reflect(L, N));
	
	float specularFactor = Ns > 0.f ?
								pow(max(dot(R, V), 0.f), Ns) :
								0.f;
								
	return diffuseFactor * diffuse * lc + specularFactor * lc;
}

// FROM http://ruh.li/GraphicsCookTorrance.html
vec3 cookTorrance(vec3 p, vec3 normal, vec3 lightDirection, vec3 lightColor)
{
    float roughnessValue = 0.3; // 0 : smooth, 1: rough
    float F0 = 0.4; // fresnel reflectance at normal incidence
    float k = 0.1; // fraction of diffuse reflection (specular reflection = 1 - k)
    
    float NdotL = max(dot(normal, lightDirection), 0.000001);
    
    float specular = 0.0;
    if(NdotL > 0.0)
    {
        vec3 eyeDir = normalize(-p);

        // calculate intermediary values
        vec3 halfVector = normalize(lightDirection + eyeDir);
        float NdotH = max(dot(normal, halfVector), 0.000001); 
        float NdotV = max(dot(normal, eyeDir), 0.000001); // note: this could also be NdotL, which is the same value
        float VdotH = max(dot(eyeDir, halfVector), 0.0);
        float mSquared = roughnessValue * roughnessValue;
        
        // geometric attenuation
        float NH2 = 2.0 * NdotH;
        float g1 = (NH2 * NdotV) / VdotH;
        float g2 = (NH2 * NdotL) / VdotH;
        float geoAtt = min(1.0, min(g1, g2));
     
        // roughness (or: microfacet distribution function)
        // beckmann distribution function
        float r1 = 1.0 / ( 4.0 * mSquared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (mSquared * NdotH * NdotH);
        float roughness = r1 * exp(r2);
        
        // fresnel
        // Schlick approximation
        float fresnel = pow(1.0 - VdotH, 5.0);
        fresnel *= (1.0 - F0);
        fresnel += F0;
        
        specular = (fresnel * geoAtt * roughness) / (NdotV * NdotL * 3.14);
    }
    
    return lightColor * NdotL * (k + specular * (1.0 - k));
}

out vec4 colorOut;
void main(void)
{
	vec3 N = normalize(normal);
	
	vec4 reflectColor = texture(EnvMap, reflectDir);
	colorOut = Ka * reflectColor;
	
	for(int l = 0; l < lightCount; ++l)
	{
		vec3 L = normalize((ViewMatrix * vec4(Lights[l].position.xyz, 1.0)).xyz - position);
		
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
		colorOut += visibility * cookTorrance(position, N, L, Lights[l].color);//phong(position, N, diffuse, L, Lights[l].color);
	}
	
	colorOut.w = diffuse.w;
}

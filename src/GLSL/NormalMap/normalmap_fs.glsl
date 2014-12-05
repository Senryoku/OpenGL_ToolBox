#version 430 core

#define Samples9

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform float minDiffuse = 0.0f;
uniform vec4 ambiant = vec4(0.2f, 0.2f, 0.2f, 1.f);

#ifdef Phong
uniform float Ns = 8.f;
#endif

uniform float roughness = 0.3; // 0 : smooth, 1: rough
uniform float F0 = 0.4; // fresnel reflectance at normal incidence
uniform float diffuseReflection = 0.8; // fraction of diffuse reflection (specular reflection = 1 - k)

uniform unsigned int lightCount = 0;

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
} Lights[8];

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

#ifdef Phong
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
#endif

// FROM http://ruh.li/GraphicsCookTorrance.html
vec3 cookTorrance(vec3 p, vec3 normal, vec3 diffuseColor, vec3 specularColor, vec3 eyeDir, vec3 lightDirection)
{    
    float NdotL = max(dot(normal, lightDirection), 0.000001);
    
    float specular = 0.0;
    if(NdotL > 0.0)
    {
        // calculate intermediary values
        vec3 halfVector = normalize(lightDirection + eyeDir);
        float NdotH = max(dot(normal, halfVector), 0.000001); 
        float NdotV = max(dot(normal, eyeDir), 0.000001);
        float VdotH = max(dot(eyeDir, halfVector), 0.0);
        float mSquared = roughness * roughness;
        
        // geometric attenuation
        float NH2 = 2.0 * NdotH;
        float g1 = (NH2 * NdotV) / VdotH;
        float g2 = (NH2 * NdotL) / VdotH;
        float geoAtt = min(1.0, min(g1, g2));
     
        // roughness (or: microfacet distribution function)
        // beckmann distribution function
        float r1 = 1.0 / ( 4.0 * mSquared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (mSquared * NdotH * NdotH);
        float r = r1 * exp(r2);
        
        // fresnel
        // Schlick approximation
        float fresnel = pow(1.0 - VdotH, 5.0);
        fresnel *= (1.0 - F0);
        fresnel += F0;
        
        specular = (fresnel * geoAtt * r) / (NdotV * NdotL * 3.1415926);
    }
    
    return NdotL * (diffuseColor + specularColor * (diffuseReflection + specular * (1.0 - diffuseReflection)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// C/C From http://www.geeks3d.com/20130122/normal-mapping-without-precomputed-tangent-space-vectors/

// http://www.thetenthplanet.de/archives/1180
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );
 
    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
 
    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

vec3 perturb_normal( vec3 N, vec3 V, vec2 texcoord )
{
    // assume N, the interpolated vertex normal and 
    // V, the view vector (vertex to eye)
   vec3 map = texture(NormalMap, texcoord ).xyz;
   map = map * 255./127. - 128./127.;
    mat3 TBN = cotangent_frame(N, -V, texcoord);
    return normalize(TBN * map);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// My glitchy version
vec3 normal_mapping()
{
	vec3 tang;
	vec3 tmp0 = cross(normal, vec3(0.f, 0.f, 1.f));
	vec3 tmp1 = cross(normal, vec3(0.f, 1.f, 0.f));
	if(length(tmp0) > length(tmp1))
		tang = normalize(tmp0);	
	else
		tang = normalize(tmp1);	
	vec3 bitang = normalize(cross(normal, tang));
	
	mat3 TangentToWorldSpace = inverse(transpose(mat3(tang, bitang, normal)));
	
	vec3 N2 = 2.0f * vec3(texture(NormalMap, texcoord)) - 1.0f;
	return normalize(TangentToWorldSpace  * N2);
}

out vec4 colorOut;
void main(void)
{	
	vec3 N = perturb_normal(normal, position, texcoord);
	
	vec4 diffuse = texture(Texture, texcoord);
	colorOut = ambiant * diffuse;
	
    vec3 eyeDir = normalize(-position);
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
			#ifdef PoissonSampling
			for (int i = 0; i < poissonSamples; ++i)
			{
				sc.xy += poissonDisk[i] * sc.w/poissonDiskRadius;
				if(textureProj(ShadowMap[l], sc.xyw).z + bbias < sc.z/sc.w)
				{
					visibility -= (1.0f - minDiffuse) / poissonSamples;
					specular_visibility = 0.f;
				}
			}
			#endif
			
			#ifdef Samples5
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(0, 0)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
				
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(-1, -1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(-1, 1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(1, -1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(1, 1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			#endif
			
			#ifdef Samples9
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(0, 0)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
				
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(-1, -1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(-1, 1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(1, -1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(1, 1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
				
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(1, 0)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(0, 1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(-1, 0)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			if(textureProjOffset(ShadowMap[l], sc.xyw, ivec2(0, -1)).z + bbias < sc.z/sc.w)
				visibility -= (1.0f - minDiffuse) / 9.0;
			#endif
		} else {
			visibility = minDiffuse;
			specular_visibility = 0.f;
		}
		#ifdef PHONG
		colorOut += visibility * phong(position, N, diffuse, L, Lights[l].color);
		#else
		colorOut += visibility * vec4(cookTorrance(position, N, diffuse.rgb, Lights[l].color.rgb, eyeDir, L), 1.0);
		#endif
	}
	colorOut.w = diffuse.w;
}

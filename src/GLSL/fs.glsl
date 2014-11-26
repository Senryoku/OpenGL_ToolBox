// ShaderToy inputs
uniform vec3     		iResolution;           // viewport resolution (in pixels)
uniform float     		iGlobalTime;           // shader playback time (in seconds)
uniform float     		iChannelTime[4];       // channel playback time (in seconds)
uniform vec3			iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4				iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform samplerCube	iChannel0;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel1;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel2;          // input channel. XX = 2D/Cube
uniform vec4      		iDate;                 // (year, month, day, time in seconds)
uniform float     		iSampleRate; 
    
uniform float ArmCount;
uniform int SphereCount;
uniform float SizeMult;
uniform float SpeedMult;

#define PI 3.14159

// You can play with these defines ! :D

#define SHOW_LIGHT
#define SHADOW
#define REFLEXION
#define COOKTORRANCE
#define NORMALMAPPING_TEST

// Yeap, that s really dirty
#ifdef COOKTORRANCE 
	#define phong cookTorrance
#endif

//#define MULTISAMPLING
#define SAMPLES 4

#define MOTIONBLUR_SAMPLES 1.0
//#define MOTIONBLUR_SAMPLES 4.0

#define SPHERE_LIGHT
const float LightDefaultRadius = 0.5;

#define LIGHTS_COUNT 3

//#define GAMMA
const vec3 Gamma = vec3(2.0, 2.0, 2.0);

struct Material
{
    int type;
    float reflectivity;
    
    vec3 diffuse;
    
    // Cook-Torrance
    float roughness;
    float fresnelReflectance;
    float diffuseReflection;
    
    // Blinn-Phong
    float specular;
    
    vec4 infos;
    vec2 uv;
};
    
struct Plane
{
    vec3 p;
    vec3 n;
};

struct Sphere
{
    vec3 c;
    float r;
};

struct Ray
{
    vec3 o;
    vec3 d;
    float m;
};
    
struct Output
{
    vec3 p;
    vec3 n;
    float d;
    Material m;
};
    
struct Light
{
    vec3 color;
    vec3 position;
    #ifdef SPHERE_LIGHT
    float radius;
    #endif
};

Light Lights[LIGHTS_COUNT];


mat2 rot2d(float a)
{
	float c = cos(a);
	float s = sin(a);
	return mat2(c,-s,
				s, c);
}

vec3 rotateX(vec3 p, float a)
{
	float sa = sin(a);
	float ca = cos(a);
	return vec3(p.x, ca*p.y - sa*p.z, sa*p.y + ca*p.z);
}

vec3 rotateY(vec3 p, float a)
{
	float sa = sin(a);
	float ca = cos(a);
	return vec3(ca*p.x + sa*p.z, p.y, -sa*p.x + ca*p.z);
}

mat4 rot(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

mat3 transpose(mat3 i)
{
    return mat3(i[0].x, i[1].x, i[2].x,
                i[0].y, i[1].y, i[2].y,
                i[0].z, i[1].z, i[2].z);
}

float det(mat2 m)
{
    return m[0].x * m[1].y - m[1].x * m[0].y;
}

float det(mat3 m)
{
    return m[0].x * det(mat2(m[1].yz, m[2].yz)) -
           m[1].x * det(mat2(m[0].yz, m[2].yz)) +
           m[2].x * det(mat2(m[0].yz, m[1].yz));
}

mat3 inverse(mat3 m)
{
    float d = det(m);
    if(d == 0.0)
        return mat3(1.0);
    return (1.0 / d) * mat3(
        vec3(det(mat2(m[1].yz, m[2].yz)), det(mat2(m[2].yz, m[0].yz)), det(mat2(m[0].yz, m[1].yz))),
        vec3(det(mat2(m[2].xz, m[1].xz)), det(mat2(m[0].xz, m[2].xz)), det(mat2(m[1].xz, m[0].xz))),
        vec3(det(mat2(m[1].xy, m[2].xy)), det(mat2(m[2].xy, m[0].xy)), det(mat2(m[0].xy, m[1].xy))));
}

vec3 tangentToWorldSpace(vec3 v, vec3 n)
{   
	vec3 tang = normalize(v);	
	vec3 bitang = normalize(cross(n, tang));
	tang = normalize(cross(bitang, n));
    
    return normalize((tang * v.x) + (bitang * v.y) + (n * v.z));
}

vec2 sphereUV(vec3 p, vec3 c)
{
    vec3 d = normalize(c - p);
    if(d.z == 0.0) return vec2(0.0, 0.0);
    return vec2(0.5 + atan(d.x, d.z)/(2.0 * PI),
                0.5 - asin(d.y)/PI);
}

//////////////////////////////////////////////////////////////
// FROM https://www.shadertoy.com/view/4ss3W7

float luminance(vec3 c)
{
	return dot(c, vec3(.2126, .7152, .0722));
}

vec3 normal(vec2 t, sampler2D tx, vec2 txsize, float depth)
{    
#define OFFSET_X 0.3
#define OFFSET_Y 0.3
    float R = abs(luminance(texture2D(tx, t + vec2( OFFSET_X,0) / txsize).xyz));
	float L = abs(luminance(texture2D(tx, t + vec2(-OFFSET_X,0) / txsize).xyz));
	float D = abs(luminance(texture2D(tx, t + vec2(0, OFFSET_Y) / txsize).xyz));
	float U = abs(luminance(texture2D(tx, t + vec2(0,-OFFSET_Y) / txsize).xyz));
  
	float X = (L-R) * .5;
	float Y = (U-D) * .5;

	return normalize(vec3(X, Y, 1. / depth));
}

//////////////////////////////////////////////////////////////

bool tracePlane(Plane p, Ray r, inout Output o)
{
    float d = dot(r.o, p.p);
    
    float l = dot(p.n, (p.p - r.o)) / dot(p.n, r.d);
    
    vec3 h = r.o + l * r.d;
        
   	if(l < 0.0 || l > o.d || l > r.m)
        return false;
     
    // Hit
    o.p = h;
    o.n = p.n;
    o.d = abs(l);
    o.m = Material(1, 0.1, vec3(0.0), 0.8, 0.4, 0.1, 0.0, vec4(0.0), vec2(0.0));
    
    return true;
}

bool traceSphere(Sphere s, Ray r, inout Output o)
{	
    vec3 d = r.o - s.c;
	
	float a = dot(r.d, r.d);
	float b = dot(r.d, d);
	float c = dot(d, d) - s.r * s.r;
	
	float g = b*b - a*c;
	
	if(g > 0.0)
    {
		float dis = (-sqrt(g) - b) / a;
		if(dis > 0.0 && dis < o.d)
        {
			o.p = r.o + r.d * dis;
			o.n = (o.p - s.c) / s.r;
            o.d = dis;
            o.m = Material(0, 0.15, vec3(0.0), 0.2, 0.9, 0.5, 64.0, vec4(s.c, 0.0), vec2(0.0));
            return true;
		}
	}
    return false;
}

bool traceScene(Ray r, inout Output o, float time)
{
    Sphere s;
    bool b = false;
    
    b = tracePlane(Plane(vec3(0.0, -5.0, 0.0), vec3(0.0, 1.0, 0.0)), r, o) || b;
    
    s = Sphere(vec3(0.0), 2.0);
    b = traceSphere(s, r, o) || b;
	
    float d = PI/(ArmCount/2.0);
    
    for(float i = 0.0; i < ArmCount; ++i)
    {
        float size = 0.5;
    	float speed = 0.5;
        for(float j = 0.0; j < float(SphereCount); ++j)
        {
        	s = Sphere((4.0 + j*1.0) * vec3(cos(time * speed + i*d), 
                                                0.0, 
                                                sin(time * speed + i*d)), size);
        	b = traceSphere(s, r, o) || b;
            size *= SizeMult;
            speed *= SpeedMult;
        }
    }
    
    return b;
}

bool traceScene(Ray r, inout Output o)
{
    return traceScene(r, o, iGlobalTime);
}

#ifndef COOKTORRANCE

vec3 phong(Light li, vec3 p, vec3 rd, vec3 n, in Material mat)
{   
    vec3 color = vec3(0.0);
    vec3 l = li.position - p;
    
    #ifdef SPHERE_LIGHT
    vec3 r = reflect(rd, n);
    vec3 centerToRay = dot(l,n) * r - l;
    vec3 closestPoint = l + centerToRay * clamp(li.radius / length(centerToRay), 0.0, 1.0);
    l = normalize(closestPoint);
    #else
    l = normalize(l);
    #endif
    
    float lambertTerm = dot(n,l);
    if(lambertTerm > 0.0)
    {		
        // Diffuse Term
        color += li.color * mat.diffuse * lambertTerm;	

        // Specular Term
        if(mat.specular > 0.0)
        {
            vec3 e = normalize(rd);
            vec3 r = reflect(l, n);
            float specular = pow( max(dot(r, e), 0.0), mat.specular);
            color += li.color * specular;
        }
    }
    
    return color;
}

vec3 phong(vec3 p, vec3 rd, vec3 n, in Material mat)
{
    vec3 color = mat.diffuse; // "Ambiant" Term
    
    for(int i = 0; i < LIGHTS_COUNT; ++i)
    {
    	color += phong(Lights[i], p, rd, n, mat);
    }
    return color;
}

#else

// FROM http://ruh.li/GraphicsCookTorrance.html
vec3 cookTorrance(Light li, vec3 p, vec3 rd, vec3 n, in Material m)
{
    // set important material values
    float roughnessValue = m.roughness; // 0 : smooth, 1: rough
    float F0 = m.fresnelReflectance; // fresnel reflectance at normal incidence
    float k = m.diffuseReflection; // fraction of diffuse reflection (specular reflection = 1 - k)
    vec3 lightColor = li.color;
    
    // interpolating normals will change the length of the normal, so renormalize the normal.
    vec3 normal = normalize(n);
    
    vec3 lightDirection = li.position - p;
       
    #ifdef SPHERE_LIGHT
    vec3 r = reflect(rd, n);
    vec3 centerToRay = dot(lightDirection, n) * r - lightDirection;
    vec3 closestPoint = lightDirection + centerToRay * clamp(li.radius / length(centerToRay), 0.0, 1.0);
    lightDirection = normalize(closestPoint);
    #else
    lightDirection = normalize(l);
    #endif
    
    // do the lighting calculation for each fragment.
    
    float NdotL = max(dot(normal, lightDirection), 0.000001);
    
    float specular = 0.0;
    if(NdotL > 0.0)
    {
        vec3 eyeDir = normalize(-rd);

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

vec3 cookTorrance(vec3 p, vec3 rd, vec3 n, in Material m)
{
    vec3 color = m.diffuse;
    
    for(int i = 0; i < LIGHTS_COUNT; ++i)
    {
    	color += cookTorrance(Lights[i], p, rd, n, m);
    }
    
    return color;
}

#endif

vec3 background(in vec3 d)
{
    return textureCube(iChannel0, d).rgb;
}

float falloff( float r, float R)
{
	float x = clamp(r / R, 0.0, 1.0);
	float y = (1.0 - x * x);
	return y * y * y;
}

float rand(vec2 co)
{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void sparkle(out vec3 c, vec2 center, vec2 tmp, float offset)
{
    vec2 v;
    vec2 v1 = (tmp - center) * vec2(6.0/PI, 1.0);
    vec2 v2 = (1.0 - tmp - center) * vec2(6.0/PI, 1.0);

    float l = length(v1); 
    if(length(v2) < l)
    {
        l = length(v2);
        v = v2;
    } else {
        v = v1;
    }

    if(l < 0.06 && (l/0.1 - mod(iGlobalTime + offset, 1.0)) > 0.0 &&
       (l/0.1 - mod(iGlobalTime + offset, 1.0)) < 0.15)
        c = vec3(1.0 - 10.0 * l);
}

vec3 diffuse(inout Output o)
{
    if(o.m.type == 0)
    {
        const float str = 0.5;
        vec3 c = str * texture2D(iChannel1, o.m.uv).xyz;
        
        #ifdef BLUE_BACK
        vec2 tmp = sphereUV(o.p, o.m.infos.xyz);
        if(length(tmp - vec2(0.32, 0.45)) > 0.1 && dot(c, c) > (str * str * 0.8) * 3.0)
        {
            c = vec3(0.11, 0.22, 0.48) * 0.8 + 0.2 * texture2D(iChannel3, tmp).xxx;
            
            #ifdef HORRIBLE_SPARKLES
            for(float i = 0.0; i < 3.0; ++i)
            	for(float j = 0.0; j < 3.0; ++j)
            		sparkle(c, vec2(rand(vec2(i + 64.0, j)), rand(vec2(j + 53.0, i))), tmp, rand(vec2(i, j)));
       		#endif
        }
        #endif
        
        return c;
        
        //return 0.3 * texture2D(iChannel1, sphereUV(o.p, o.infos.xyz)).xyz;
    } else if(o.m.type == 1) {
        return 0.5 * texture2D(iChannel2, o.m.uv).xyz;
    } else {
        return vec3(1.0, 0.0, 0.0);
    }
}

vec3 shadow(in Ray r, in Output o)
{
    vec3 add = o.m.diffuse;
    for(int i = 0; i < LIGHTS_COUNT; ++i)
    {
        Ray shadowray;
        Output shadowoutput;
        vec3 l = Lights[i].position - o.p;
        
        #ifdef SPHERE_LIGHT
        vec3 re = reflect(r.d, o.n);
        vec3 centerToRay = dot(l, o.n) * re - l;
        vec3 closestPoint = l + centerToRay * clamp(Lights[i].radius / length(centerToRay), 0.0, 1.0);
    	l = normalize(closestPoint);
        #else
        l = normalize(l);
        #endif
        
        shadowray.o = o.p + 0.01 * l;
        shadowray.d = l;
        shadowray.m = 100.0;
        shadowoutput.d = length(Lights[i].position - o.p);
        if(!traceScene(shadowray, shadowoutput))
        {
            add += phong(Lights[i], o.p, r.d, o.n, o.m);
        }
    }
    return add;
}

vec3 getColor(in Ray r, inout Output o)
{
    if(o.m.type == 0)
    {
        o.m.uv = sphereUV(o.p, o.m.infos.xyz);
		
		o.m.uv.x *= 6.0 / PI;

        #ifdef NORMALMAPPING_TEST
        // Normal Mapping! (Or Not.)
        o.n = tangentToWorldSpace(normal(o.m.uv, iChannel1, vec2(256.0, 32.0), 2.0), o.n);
        #endif
    } else if(o.m.type == 1) {
        o.m.uv = vec2(0.5) + o.p.xz / 50.0;

        #ifdef NORMALMAPPING_TEST
        o.n = tangentToWorldSpace(normal(o.m.uv, iChannel2, vec2(512.0), 5.0).xyz, o.n);
        #endif
    }
    
    o.m.diffuse = diffuse(o);
    
    #ifdef SHADOW
    return shadow(r, o);
    #else
    return phong(o.p, r.d, o.n, o.m);
    #endif
}

vec3 trace(Ray r, float time)
{
    vec3 rgb = vec3(0.0);
    
    #ifdef SHOW_LIGHT
    Output outlight;
    float minl = 100.0;
    for(int i = 0; i < LIGHTS_COUNT; ++i)
    {
    	outlight.d = 100.0;
        #ifdef SPHERE_LIGHT
        float radius = Lights[i].radius;
        #else
        const float radius = 0.3;
        #endif
        
        Sphere Light = Sphere(Lights[i].position, radius);
        if(traceSphere(Light, r, outlight))
        {
            float d = abs(dot(r.d, outlight.n));
            rgb += Lights[i].color * d * d * d;
        }
        
        minl = min(minl, outlight.d);
    }
	#endif
    
    Output o;
    o.d = 100.0;
    if(traceScene(r, o, time))
    {
        #ifdef SHOW_LIGHT
        if(o.d < minl) rgb = vec3(0.0);
        #endif 

        rgb += getColor(r, o);

        #ifdef REFLEXION
        // Could use Cook-Torrance parameters
        // and some kind of blur based on them...
        if(o.m.reflectivity > 0.0)
        {
            Ray refray;
            refray.d = reflect(r.d, o.n);
            refray.o = o.p + 0.01 * refray.d;
            refray.m = 100.0;
            Output refout;
            refout.d = 100.0;
            if(traceScene(refray, refout, time))
            {
                rgb += o.m.reflectivity * getColor(refray, refout);
            } else {
                rgb += o.m.reflectivity * background(refray.d);
            }
        }
        #endif
    } else {
        rgb += background(r.d);
    } 
	
    return rgb;
}

vec3 trace(Ray r)
{
    return trace(r, iGlobalTime);
}

void main(void)
{
	vec2 pixel = (gl_FragCoord.xy / iResolution.xy) * 2.0 - 1.0;
    
    #if LIGHTS_COUNT > 0
    Lights[0].color = vec3(1.0, 0.5, 0.5);
    Lights[0].position = 4.5 * vec3(cos(iGlobalTime), cos(1.2 * iGlobalTime), sin(iGlobalTime));
    #ifdef SPHERE_LIGHT
    Lights[0].radius = LightDefaultRadius;
    #endif
    #endif
    
    #if LIGHTS_COUNT > 1
    Lights[1].color = vec3(0.8, 1.0, 0.8);
    Lights[1].position = 4.5 * vec3(cos(0.5*iGlobalTime), sin(2.0 * iGlobalTime), sin(0.5*iGlobalTime));
    Lights[2].position.xz *= 1.5;
    #ifdef SPHERE_LIGHT
    Lights[1].radius = LightDefaultRadius;
    #endif
    #endif
    
    #if LIGHTS_COUNT > 2
    Lights[2].color = vec3(0.8, 0.8, 1.0);
    Lights[2].position = 4.5 * vec3(cos(0.8*iGlobalTime), 0.8 * sin(3.0 * iGlobalTime), sin(0.8*iGlobalTime));
    Lights[2].position.xz *= 2.0;
    #ifdef SPHERE_LIGHT
    Lights[2].radius = LightDefaultRadius;
    #endif
    #endif
    
	float asp = iResolution.x / iResolution.y;
	Ray r;
    r.m = 100.0;
	r.d = normalize(vec3(asp * pixel.x, pixel.y, -5.0));
	r.o = vec3(0.0, 0.0, 30.0);
    
    vec2 um = (iMouse.xy / iResolution.xy-.5);
    um.x *= 8.0;
    um.y = clamp(um.y, -0.5, 0.1);
	if(iMouse.z > 0.0)
	{
		r.o = rotateX(r.o, um.y);
		r.d = rotateX(r.d, um.y);
		r.o = rotateY(r.o, um.x);
		r.d = rotateY(r.d, um.x);
    }
	
    vec3 rgb = vec3(0.0);
    
    #ifdef MULTISAMPLING
    //TODO
    vec3 rd = r.d;
    r.d += dFdx(r.d)/2.0;
    r.d = normalize(r.d);
	r.m = 100.0;
    for(int i = 0; i < SAMPLES; i++)
    {
        for(float j = 0.0; j < MOTIONBLUR_SAMPLES; ++j)
        {
        	rgb += trace(r, iGlobalTime - 0.005 * j) / MOTIONBLUR_SAMPLES / float(SAMPLES);
        }
        r.d *= mat3(rot(rd, 2.0 * PI / float(SAMPLES)));
    }
    #else
    for(float j = 0.0; j < MOTIONBLUR_SAMPLES; ++j)
    {
        rgb += trace(r, iGlobalTime - 0.005 * j) / MOTIONBLUR_SAMPLES;
    }
    #endif
    
    // Nice transitions <3
    #ifdef GAMMA
    float Duration = 2.0;
    float x = falloff(mod(iGlobalTime, Duration), 1.0);
    float phase = mod(iGlobalTime, 4.0 * Duration);
    if(phase < Duration)
        x = x - 1.0;
    else if(phase < 2.0 * Duration)
        x = - x;
    else if(phase < 3.0 * Duration)
        x = 1.0 - x;
    else if(phase > 3.0 * Duration)
        x = x;

    if(pixel.x > x)
        rgb = pow(rgb, 1.0 / Gamma);
    #endif
	
   	gl_FragColor.rgb = rgb;
    gl_FragColor.a = 1.0;
}


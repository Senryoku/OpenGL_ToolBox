#version 440

// ShaderToy inputs
uniform vec3     		iResolution;           // viewport resolution (in pixels)
uniform float     		iGlobalTime;           // shader playback time (in seconds)
uniform float     		iChannelTime[4];       // channel playback time (in seconds)
uniform vec3			iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4			iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform samplerCube	iChannel0;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel1;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel2;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel3;          // input channel. XX = 2D/Cube
uniform vec4      		iDate;                 // (year, month, day, time in seconds)
uniform float     		iSampleRate; 

// First try at RayMarching / Use of implicit surfaces
// Senryoku - 09/2014
// Based on "Blobs" by Eric Galin

// Updates:
// 22/11/14 - Small cleanup and optimization
// (...Many modifications...)

// Configuration

// Comment this to enable mouse control
#define AUTO_ROTATE 
#define SELF_REFLEXION
#define SELF_SHADOWING
#define SOFT_SHADOW
#define ENABLE_POINT_LIGHT_FLARE

const int Steps = 150; // Max. ray steps before bailing out
const float Epsilon = 0.0025; // Marching epsilon

const float RayMaxLength = 6.5;
const float Near = 16.5; // Screen rays starting point

// Point Light
vec3 LightPos = vec3(5.0 , 2.0, -5.0);
const vec3 LightColor = vec3(1.0, 1.0, 1.0);

float Time = 0.0;

// Pre-declarations

// Rotations around an axis
vec3 rotateX(vec3 p, float a);
vec3 rotateY(vec3 p, float a);
vec3 rotateZ(vec3 p, float a);

struct Sphere
{
    vec3 c;
    float r;
};

struct Ray
{
    vec3 o;
    vec3 d;
};

bool traceSphere(Sphere s, Ray r, out float o)
{	
    vec3 d = r.o - s.c;
	
	float a = dot(r.d, r.d);
	float b = dot(r.d, d);
	float c = dot(d, d) - s.r * s.r;
	
	float g = b*b - a*c;
	
	if(g > 0.0)
    {
		float dis = (-sqrt(g) - b) / a;
		if(dis > 0.0)
        {
			o = dis;
            return true;
		}
	}
    return false;
}

///////////////////////////////////////////////////////////////////////////
// Distance => Energy function

// Smooth falloff function
// Returns (1 - r²/R²)^3
// r : small radius
// R : Large radius
float falloff( float r, float R)
{
	float x = clamp(r / R, 0.0, 1.0);
	float y = (1.0 - x * x);
	return y * y * y;
}

// Returns the maximum of the derivative of the falloff function
// (Lipschitz constant)
float falloff_lipschitz(float R)
{
	return 1.71730020672 // ~= 96.0 / (25.0 * sqrt(5.0))
				/ (R * R);
}

///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Primitive functions

// Point skeleton
// p : point
// c : center of skeleton
// e : energy associated to skeleton
// R : large radius
float point(vec3 p, vec3 c, float e, float R)
{
	return e * falloff(length(p - c), R);
}

// Segment skeleton
// p : point
// a : First point of skeleton
// b : Second point of skeleton
// e : energy associated to skeleton
// R : large radius
float segment(vec3 p, vec3 a, vec3 b, float e, float R)
{
	vec3 ab = b - a;
	float t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
	return e * falloff(length(p - (a + t * ab)), R);
}

// Circle skeleton
// p : point
// c : circle center
// n : circle normal
// r : circle radius
// e : energy associated to skeleton
// R : large radius
float circle(vec3 p, vec3 c, vec3 n, float r, float e, float R)
{
	vec3 rad = c + r * normalize(p - dot(p, n) * n);
	return e * falloff(distance(p, rad), R);
}

// Disc skeleton
// p : point
// c : disc center
// n : disc normal
// r : disc radius
// e : energy associated to skeleton
// R : large radius
float disc(vec3 p, vec3 c, vec3 n, float r, float e, float R)
{
    vec3 proj = p - dot(p, n) * n;
    if(length(proj - c) < r)
        return e * falloff(distance(p, proj), R);
	vec3 rad = c + r * normalize(proj);
	return e * falloff(distance(p, rad), R);
}

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
// Object

// Potential Field properties
const float AmbiantEnergy = -5.0; // Base value of the potential field

// Primitive properties
const float Energy = 6.0; // Energy of each primitive (reached on skeleton)
const float Radius = 1.1; // Influence radius of each primitive

float SphereTracedObject(vec3 p)
{
	p.z=-p.z;
	float v = AmbiantEnergy;

	v += circle(p, vec3(0.0, 0.0, 0.0), normalize(vec3(sin(Time), tan(Time), cos(Time))), 3.0, Energy, Radius);
	v += point(p, vec3( 3.0 * cos(2.0 * Time), 3.0 * sin(2.0 * Time), 0.0), Energy, Radius);
	v += point(p, vec3( 3.0 * cos(3.0 * Time), 3.0 * sin(3.0 * Time), 1.0), Energy, Radius);
	v += point(p, vec3( 3.0 * cos(6.0 * Time), 3.0 * sin(6.0 * Time), 1.0), Energy, Radius);
	v += point(p, vec3( 0.0, 0.0, 0.0), Energy, 2.0);
	v += point(p, vec3( cos(2.0 * Time), 2.0 * sin(Time), 1.0), Energy, Radius);
	v += point(p, vec3( 3.0 * sin(3.0*Time), 1.0, 2.0 * cos(3.0*Time)), Energy, Radius);
	v += point(p, vec3( 3.0 * sin(2.0*Time + 0.5), 1.0, 3.0 * cos(2.0*Time)), Energy, Radius);

	return v;
}

// Returns lipschtz constant of the object's function
// /!\ Have to be updated with the object function...
float object_lipschitz()
{
    return 11.0; // Good enough :] More permissive constant => Speeds up the whole thing. (But can cause artefacts)
	//return 7.0 * Energy * falloff_lipschitz(Radius) + Energy * falloff_lipschitz(2.0);
}

// Normal of the SphereTracedObject at point p
// p : point
vec3 SphereTracedObjectNormal(in vec3 p)
{
	float eps = 0.0001;
	vec3 n;
	float v = SphereTracedObject(p);
	n.x = SphereTracedObject( vec3(p.x+eps, p.y, p.z) ) - v;
	n.y = SphereTracedObject( vec3(p.x, p.y+eps, p.z) ) - v;
	n.z = SphereTracedObject( vec3(p.x, p.y, p.z+eps) ) - v;
	return -normalize(n);
}

///////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Tracing

// Trace ray using sphere tracing
// a : ray origin
// u : ray direction
vec3 SphereTrace(vec3 a, vec3 u, out bool hit)
{
	hit = false;
	vec3 p = a;
	float lambda = object_lipschitz();
	float depth = 0.0;
	float step = 0.0;
	for(int i = 0; i < Steps; i++)
	{
		float v = SphereTracedObject(p);
		if (v > 0.0)
		{
			hit = true; 
			return p;
		}
		
		step = max(abs(v) / lambda, Epsilon);
		depth += step;
		
		if(depth > RayMaxLength)
			return p;
		
		p += step * u;
	}
	return p;
}

float softShadow(vec3 a, vec3 u, float k)
{
    float r = 1.0;
	vec3 p = a;
	float lambda = object_lipschitz();
	float depth = 0.0;
	float step = 0.0;
	for(int i = 0; i < Steps; i++)
	{
		float v = SphereTracedObject(p);
		if (v > 0.0)
			return 0.0;
        r = min(r, k * (v / AmbiantEnergy) / depth);
		
		step = max(abs(v) / lambda, Epsilon);
		depth += step;
		
		if(depth > RayMaxLength)
			return r;
		
		p += step * u;
	}
	return r;
}

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Shading

// Background color
vec3 background(vec3 rd, float bias)
{
	return texture(iChannel0, rd, bias).xyz;
}

vec3 background(vec3 rd)
{
	return texture(iChannel0, rd).xyz;
}

vec3 shade_reflect(vec3 p, vec3 rd, vec3 n)
{
	vec3 diffuse = background(reflect(rd, n));
	vec3 l = normalize(LightPos - p);

	// Phong shading
	vec3 color = 0.4 * diffuse; // "Ambiant" Term
	
    float penumbra = 1.0;
	float lambertTerm = dot(n,l);
	if(lambertTerm > 0.0)
	{
		#ifdef SELF_SHADOWING
		bool gotout = false;
		bool hit = false;
		vec3 p2 = p + Epsilon * l;
		for(int i = 0; i < Steps; i++)
		{
			float v = SphereTracedObject(p2);
			if (v < 0.0)
			{
				gotout = true;
				break;
			}

			p2 += Epsilon * l;
		}
        #ifdef SOFT_SHADOW
        if(gotout)
            penumbra = clamp(softShadow(p2, l, 16.0), 0.0, 1.0);
        #else
		if(gotout)
			SphereTrace(p2, l, hit);
		if(hit) // We're in shadows, skip diffuse and specular terms 
			return color;
        #endif
		#endif
		
        vec3 shaded = penumbra * LightColor;
        
		// Diffuse Term
		color += lambertTerm * shaded * diffuse;	

		// Specular Term
		vec3 r = reflect(l, n);
		float specular = pow( max(dot(r, rd), 0.0), 8.0);
		color += specular * shaded;	
	}

	return color;
}

//////////////////////////////////////////////////////////////////////////

out vec4 outColor;

void main(void)
{
	vec2 pixel = (gl_FragCoord.xy / iResolution.xy) * 2.0 - 1.0;
	
	Time = 0.2 * iGlobalTime;
	
	LightPos = vec3(5.0 * sin(Time * 2.0), 2.0, 5.0 * cos(Time * 2.0));

	// Compute ray origin and direction
	float asp = iResolution.x / iResolution.y;
	vec3 rd = normalize(vec3(asp * pixel.x, pixel.y, -5.0));
	vec3 ro = vec3(0.0, 0.0, 20.0);

#ifdef AUTO_ROTATE
	ro = rotateY(ro, Time * 0.5);
	rd = rotateY(rd, Time * 0.5);
#else
	vec2 um = 5.0 * (iMouse.xy / iResolution.xy-.5);
	ro = rotateX(ro, um.y);
	rd = rotateX(rd, um.y);
	ro = rotateY(ro, um.x);
	rd = rotateY(rd, um.x);
#endif
	
	// Trace ray
	bool hit = false;
	vec3 pos = vec3(0.0);
	
	vec3 rgb = vec3(0.0);
	
	// Skiping useless pixels (hackish way :D)
	if(pixel.x > -0.6 && pixel.x < 0.6)
		pos = SphereTrace(ro + Near * rd, rd, hit);

	if (hit)
	{
		// Compute normal
		vec3 n = SphereTracedObjectNormal(pos);

		// Shade
#ifdef SELF_REFLEXION
		bool hit2 = false;
		vec3 ref = SphereTrace(pos + 0.1 * n, n, hit2);
		vec3 p2 = pos;
		vec3 rd2 = rd;
        if(hit2)
        {
            rd2 = normalize(ref - pos);
            p2 = ref;
			n = SphereTracedObjectNormal(ref);
        }
        rgb += shade_reflect(p2, rd2, n);
#else
		rgb += shade_reflect(pos, rd, n);
#endif
	} else {
		rgb += background(rd);
	}
	
    #ifdef ENABLE_POINT_LIGHT_FLARE
	float dist;
	const float rad = 0.2;
	if(traceSphere(Sphere(LightPos, rad), Ray(ro, rd), dist) && (!hit || dist < length(ro - pos)))
	{
		float d = 1.0 - sin(acos((length(ro - LightPos) - dist)/rad)); 
		rgb += LightColor * d * d;
	}
	#endif 
	
	outColor = vec4(rgb, 1.0);
}

//////////////////////////////////////////////////////////////////////////
// Transformations

vec3 rotateZ(vec3 p, float a)
{
	float sa = sin(a);
	float ca = cos(a);
	return vec3(ca*p.x + sa*p.y, -sa*p.x + ca*p.y, p.z);
}

vec3 rotateY(vec3 p, float a)
{
	float sa = sin(a);
	float ca = cos(a);
	return vec3(ca*p.x + sa*p.z, p.y, -sa*p.x + ca*p.z);
}

vec3 rotateX(vec3 p, float a)
{
	float sa = sin(a);
	float ca = cos(a);
	return vec3(p.x, ca*p.y - sa*p.z, sa*p.y + ca*p.z);
}

//////////////////////////////////////////////////////////////////////////

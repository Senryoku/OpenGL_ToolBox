#version 440

// ShaderToy inputs
uniform vec3     		iResolution;           // viewport resolution (in pixels)
uniform float     	iGlobalTime;           // shader playback time (in seconds)
uniform float     	iChannelTime[4];       // channel playback time (in seconds)
uniform vec3			iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4			iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform sampler3D	iChannel0;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel1;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel2;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel3;          // input channel. XX = 2D/Cube
uniform vec4      		iDate;                 // (year, month, day, time in seconds)
uniform float     	iSampleRate; 

// First try at RayMarching / Use of implicit surfaces
// Senryoku - 09/2014
// Based on "Blobs" by Eric Galin

// Updates:
// 22/11/14 - Small cleanup and optimization
// (...Many modifications...)

// Configuration

// Comment this to enable mouse control
// #define AUTO_ROTATE 

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

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
// Object

float SphereTracedObject(vec3 p)
{
	p.z = -p.z;
	float v = -0.5;

	v += texture(iChannel0, p).x;

	return v;
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
		
		depth += Epsilon;
		
		if(depth > RayMaxLength)
			return p;
		
		p += step * u;
	}
	return p;
}

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Shading

// ...

//////////////////////////////////////////////////////////////////////////

void main(void)
{
	vec2 pixel = (gl_FragCoord.xy / iResolution.xy) * 2.0 - 1.0;
	
	Time = 0.2 * iGlobalTime;

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
		rgb = vec3(1.0);
	}
	
	gl_FragColor = vec4(rgb, 1.0);
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

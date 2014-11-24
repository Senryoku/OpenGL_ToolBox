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

const int Steps = 300; // Max. ray steps before bailing out
const float Epsilon = 4.0 / Steps; // Marching epsilon

// Point Light
vec3 LightPos = vec3(5.0 , 2.0, -5.0);
const vec3 LightColor = vec3(1.0, 1.0, 1.0);

float Time = 0.0;

// Pre-declarations

mat4 rotationMatrix(vec3 axis, float angle)
{
	axis = normalize(axis);
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0 - c;
	return mat4(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.0,
	oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, 0.0,
	oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, 0.0,
	0.0, 0.0, 0.0, 1.0);
} 

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
// Object

float SphereTracedObject(vec3 p)
{
	p.z = -p.z;
	float v = -0.5;

	v += texture(iChannel0, p + vec3(0.5, 0.5, 0.5)).x;

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
	float step = 0.0;
	for(int i = 0; i < Steps; i++)
	{
		float v = SphereTracedObject(p);
		if (v > 0.0)
		{
			hit = true; 
			return p;
		}
		
		p += Epsilon * u;
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
	
	vec3 rd = normalize(vec3(asp * pixel.x, pixel.y, 1.0));
	vec3 position = vec3(0.0, 0.0, 2.0);

	const vec3 up = vec3(0.0, 1.0, 0.0);
	
	if(iMouse.z > 0)
	{
		vec2 um = 5.0 * (iMouse.xy / iResolution.xy-.5);
		position = vec3(rotationMatrix(up, um.x) * vec4(position, 1.0));
		position = vec3(rotationMatrix(normalize(-position), um.y) * vec4(position, 1.0));
	}
	
	const vec3 forward = normalize(-position);
	mat4 viewMatrix = mat4(vec4(cross(forward, up), 0), vec4(up, 0), vec4(forward, 0), vec4(vec3(0.0), 1));
	
	rd = vec3(inverse(viewMatrix) * vec4(rd, 1));
	
	vec3 ro = position;
	
	// Trace ray
	bool hit = false;
	vec3 pos = vec3(0.0);
	
	vec3 rgb = vec3(0.0);
	
	pos = SphereTrace(ro + 1.0 * rd, rd, hit);

	if (hit)
	{
		rgb = vec3(1.0);
	}
	
	gl_FragColor = vec4(rgb, 1.0);
	//gl_FragColor = vec4(texture(iChannel0, vec3(pixel, Time)).x, 0.0, 0.0, 1.0);
}

//////////////////////////////////////////////////////////////////////////

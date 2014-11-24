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

// Configuration

const float Tex3DRes = 512.0;
const int Steps = int(sqrt(2.0) * Tex3DRes); // Max. ray steps before bailing out
const float Epsilon = 1.0 / Tex3DRes; // Marching epsilon

uniform float maxLoD = log2(Tex3DRes) - 1;

// Point Light
vec3 LightPos = vec3(5.0 , 2.0, -5.0);
const vec3 LightColor = vec3(1.0, 1.0, 1.0);

float Time = 0.0;

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

float object(vec3 p, float bias)
{
	return textureLod(iChannel0, p + vec3(0.5, 0.5, 0.5), bias).x;
}

float object(vec3 p)
{
	return texture(iChannel0, p + vec3(0.5, 0.5, 0.5)).x;
}

///////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Tracing

vec3 lodTrace(vec3 a, vec3 u, out bool hit, out int i)
{
	hit = false;
	vec3 p = a;
	float step = 0.0;
	float LoD = maxLoD;
	float depth = 0.0;
	for(i = 0; i < Steps; i++)
	{
		float v = object(p, LoD);
		if (v > 0.0)
		{
			if(LoD < 0.5)
			{
				hit = true; 
				return p;
			}
			LoD = clamp(LoD - 1.0, 0.0, maxLoD);
		} else {
			depth += max(1.0, LoD) * Epsilon;
			if(depth >= sqrt(2.0))
				return p;
			p += (1.0 + LoD) * Epsilon * u;
			LoD = clamp(LoD + 1.0, 0.0, maxLoD);
		}
	}
	return p;
}

vec3 trace(vec3 a, vec3 u, out bool hit)
{
	hit = false;
	vec3 p = a;
	for(int i = 0; i < Steps; i++)
	{
		float v = object(p);
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

bool traceBox(vec3 ro, vec3 rd, vec3 lb, vec3 rt, out float t)
{
	vec3 dirfrac;
	dirfrac.x = 1.0f / rd.x;
	dirfrac.y = 1.0f / rd.y;
	dirfrac.z = 1.0f / rd.z;
	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray
	float t1 = (lb.x - ro.x) * dirfrac.x;
	float t2 = (rt.x - ro.x) * dirfrac.x;
	float t3 = (lb.y - ro.y) * dirfrac.y;
	float t4 = (rt.y - ro.y) * dirfrac.y;
	float t5 = (lb.z - ro.z) * dirfrac.z;
	float t6 = (rt.z - ro.z) * dirfrac.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
	if (tmax < 0)
	{
		t = tmax;
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax)
	{
		t = tmax;
		return false;
	}

	t = tmin;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Shading

// Phong
vec3 phong(vec3 p, vec3 rd, vec3 n, vec3 diffuse, vec3 lightPos, vec3 lightColor)
{
	vec3 l = normalize(lightPos - p);

	// Phong shading
	vec3 color = 0.4 * diffuse; // "Ambiant" Term
	
	float lambertTerm = dot(n,l);
	if(lambertTerm > 0.0)
	{
		// Diffuse Term
		color += lambertTerm * lightColor * diffuse;	

		// Specular Term
		vec3 r = reflect(l, n);
		float specular = pow( max(dot(r, rd), 0.0), 64.0);
		color += specular * lightColor;	
	}

	return color;
}

//////////////////////////////////////////////////////////////////////////

void main(void)
{
	vec2 pixel = (gl_FragCoord.xy / iResolution.xy) * 2.0 - 1.0;
	
	Time = 0.2 * iGlobalTime;

	// Compute ray origin and direction
	float asp = iResolution.x / iResolution.y;
	
	vec3 rd = normalize(vec3(asp * pixel.x, pixel.y, 2.0));
	vec3 position = vec3(0.0, 0.5, 2.0);

	const vec3 up = vec3(0.0, -1.0, 0.0);
	
	vec2 um = vec2(0.0);
	if(iMouse.w > 0)
	{
		position *= (0.5 + iMouse.y / iResolution.y);
		if(iMouse.z > 0)
		{
			um.x = 5.0 * (iMouse.x / iResolution.x - 0.5);
		}
	} else if(iMouse.z > 0) {
		um = 5.0 * (iMouse.xy / iResolution.xy - 0.5);
	}
	position = vec3(rotationMatrix(up, um.x) * vec4(position, 1.0));
	position = vec3(rotationMatrix(cross(up, normalize(-position)), um.y) * vec4(position, 1.0));
	
	const vec3 forward = normalize(-position);
	const vec3 right = cross(forward, up);
	const mat4 viewMatrix = mat4(vec4(right, 0), vec4(cross(forward, right), 0), vec4(forward, 0), vec4(vec3(0.0), 1));
	
	rd = vec3(viewMatrix * vec4(rd, 1));
	
	vec3 ro = position;
	
	vec3 rgb = vec3(0.0); // Background
	float t = 0.0;
	int s = 0;
	if(traceBox(ro, rd, vec3(-0.5), vec3(0.5), t))
	{
		// Trace ray
		bool hit = false;
		vec3 pos = vec3(0.0);
		
		pos = lodTrace(ro + t * rd, rd, hit, s);
		
		// Compute normal
		vec3 n = vec3(0.0);
		// Using Fragment derivatives
		//n = cross(dFdx(pos), dFdy(pos));

		// Treating each voxel as a "sphere". (Really weird)
		n = normalize(ro + (vec3(ivec3(pos * Tex3DRes) + 0.5) / Tex3DRes));
		
		if (hit)
		{
			rgb = phong(pos, rd, n, vec3(1.0), vec3(1.0, 2.0, 1.0), LightColor);
		} else {
			rgb = vec3(0.0,  s / float(Steps), s / float(Steps));
		}
	} else { // No intersection, early bail
		rgb = vec3(0.0, max(0.1, pixel.y + 0.5), max(0.1, pixel.y + 0.5)); 
	}
	
	gl_FragColor = vec4(rgb, 1.0);
	//gl_FragColor = vec4(texture(iChannel0, vec3(pixel, Time)).x, 0.0, 0.0, 1.0);
}

//////////////////////////////////////////////////////////////////////////

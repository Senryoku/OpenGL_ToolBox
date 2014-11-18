// ShaderToy inputs
uniform vec3     		iResolution;           // viewport resolution (in pixels)
uniform float     		iGlobalTime;           // shader playback time (in seconds)
uniform float     		iChannelTime[4];       // channel playback time (in seconds)
uniform vec3			iChannelResolution[4]; // channel resolution (in pixels)
uniform vec4				iMouse;                // mouse pixel coords. xy: current (if MLB down), zw: click
uniform sampler2D	iChannel0;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel1;          // input channel. XX = 2D/Cube
uniform sampler2D	iChannel2;          // input channel. XX = 2D/Cube
uniform vec4      		iDate;                 // (year, month, day, time in seconds)
uniform float     		iSampleRate; 

const float PI = 3.14159265359;

float module(vec2 z)
{
	return length(z);
}

float arg(vec2 z)
{
	return atan(z.x/z.y) / PI;
}

void main(void)
{
	vec2 pixel = gl_FragCoord.xy / iResolution.xy;
	pixel.y *= -1.0;
   
    vec2 z = texture(iChannel0, pixel).xy;
	
	float col = module(z);
	
	gl_FragColor = vec4(col, 0.0, 0.0, 1.0);
}


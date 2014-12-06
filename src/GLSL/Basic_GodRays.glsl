#version 430

uniform vec3     	iResolution;       // viewport resolution (in pixels)
uniform sampler2D	iChannel0;          // input channel. XX = 2D/Cube

uniform int Samples = 128;
uniform float Intensity = 0.125;
uniform float Decay = 0.96875;
	
void main(void)
{
	vec2 pixel = gl_FragCoord.xy / iResolution.xy;

    vec2 Direction = vec2(0.5) - pixel;
    Direction /= Samples;
    vec3 Color = texture(iChannel0, pixel).rgb;
	
	float _intensity = Intensity;

    for(int Sample = 0; Sample < Samples; Sample++)
    {
        Color += texture(iChannel0, pixel).rgb * _intensity;
        _intensity *= Decay;
        pixel += Direction;
    }
    
    gl_FragColor = vec4(Color, 1.0);
}


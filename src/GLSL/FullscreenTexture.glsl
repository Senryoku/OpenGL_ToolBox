// ShaderToy inputs
uniform vec3     	iResolution;       // viewport resolution (in pixels)
uniform sampler2D	iChannel0;          // input channel. XX = 2D/Cube

//in vec2 texcoords;

void main(void)
{
	vec2 pixel = gl_FragCoord.xy / iResolution.xy;
	//vec2 pixel = texcoords;
	//pixel.y *= -1.0;
   
    gl_FragColor = texture(iChannel0, pixel);
}


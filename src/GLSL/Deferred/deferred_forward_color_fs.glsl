#version 430 core

#define Samples9

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform vec4 Color;

uniform layout(binding = 0) sampler2D ColorDepth;

out layout(location = 0) vec4 colorOut;

void main(void)
{		
	vec4 In = texture(ColorDepth, gl_FragCoord.xy/textureSize(ColorDepth, 0).xy);
	if(gl_FragCoord.z < In.w)
		colorOut.rgb = Color.rgb;
	else
		colorOut.rgb = vec3(0.0);
	colorOut.a = 1.0;
}

#version 430 core

#define Samples9

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform mat4 ModelMatrix = mat4(1.0);

uniform vec4 Color;

uniform layout(binding = 0) sampler2D ColorMaterial;

out layout(location = 0) vec4 colorOut;

void main(void)
{		
	vec4 In = texture(ColorMaterial, gl_FragCoord.xy/textureSize(ColorMaterial, 0).xy);
	if(gl_FragCoord.z < In.w)
		colorOut.rgb = Color.rgb;
	else
		discard;
	//colorOut.rgb = vec3(0.0);
	colorOut.a = 1.0;
}

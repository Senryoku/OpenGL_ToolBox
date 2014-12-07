#version 430

layout(std140) uniform LightBlock {
	vec4		position;
	vec4		color;
	mat4 		depthMVP;
} Lights[8];

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform float			Intensity = 1.0f;
uniform float			Radius = 1.0f;
uniform vec3     		iResolution; // viewport resolution (in pixels)   
uniform unsigned int	lightCount = 0;

in vec2 texcoords;

out vec4 outColour;

void main()
{
	outColour = vec4(0.0, 0.0, 0.0, 1.0);

	for(unsigned int i = 0; i < lightCount; ++i)
	{
		vec4 lp = Lights[i].position/Lights[i].position.w;
		vec4 hLP = ProjectionMatrix * ViewMatrix * lp;
		if(hLP.z < 0.0) continue;
		vec2 lightPos = (hLP.xy/hLP.w + 1.0) * 0.5;
		vec2 tmp = vec2(texcoords - lightPos);
		tmp.y /= iResolution.x/iResolution.y;
		float dist = length(tmp) / (hLP.z/hLP.w) ;
		outColour.rgb += Intensity * (Radius - dist * dist) * Lights[i].color.rgb;
	}
}

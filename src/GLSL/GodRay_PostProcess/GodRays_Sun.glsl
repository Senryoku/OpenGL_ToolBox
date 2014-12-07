#version 430

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform vec3 SunPosition;

uniform sampler2D	Scene;
uniform sampler2D	GodRays;      
uniform vec3     	iResolution; // viewport resolution (in pixels)   

uniform int Samples = 128; // determines how many samples are to be drawn
uniform float Intensity = 0.5; // determines the intensity of each sample
uniform float Density = 0.5; // determines how far apart each sample is from the last
uniform float Exposure = 0.1; // determines the intensity of the Ray as a whole
uniform float Decay = 0.96875;

in vec2 texcoords;

out vec4 outColour;

void main()
{
	vec2 pixel = texcoords;
	vec4 Colour = texture(GodRays, pixel);

	vec4 hLP = ProjectionMatrix * ViewMatrix * vec4(SunPosition, 1.0);
	vec2 lightPos = (hLP.xy / hLP.w + 1.0) * 0.5;
	vec2 deltaTextCoord = vec2(pixel - lightPos);

	/* deltaTexCoord is then altered by the amount of samples and density which pushes  deltaTexCoord
	away from the light source*/
	deltaTextCoord *= 1.0 /  float(Samples) * Density;
	float illuminationDecay = 1.0;

	vec2 l_UV = pixel;

	for(int i = 0; i < Samples; i++)
	{
		l_UV -= deltaTextCoord;
		vec4 l_Ray = texture(GodRays, l_UV);
		l_Ray *= illuminationDecay * Intensity;
		Colour += l_Ray;
		illuminationDecay *= Decay;
	}

	Colour *= Exposure;
	outColour = Colour + texture(Scene, pixel);
}

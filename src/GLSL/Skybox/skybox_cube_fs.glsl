#version 430 core
 
uniform samplerCube SkyBox;

in layout(location = 0) vec2 o;
in layout(location = 1) mat4 m;
 
out vec4 colorOut;
 
void main()
{
	vec2 pixel = o - 0.5;
	vec3 rd = mat3(m) * normalize(vec3(pixel.x, pixel.y, 0.5)).xyz;
	
	colorOut = vec4(texture(SkyBox, rd).rgb, 1.0);
	//colorOut = vec4(1.0, 0.0, 0.0, 1.0);
}

#version 440
const float PI = 3.14159265359;
layout(binding = 0, rgba8) uniform readonly image2D In;
layout(binding = 1, rgba8) uniform writeonly image2D Out;

vec2 complex_mult(in vec2 lhs, in vec2 rhs)
{
	return vec2(lhs.x * rhs.x - lhs.y * rhs.y, lhs.x * rhs.y + lhs.y * rhs.x);
}

layout (local_size_x = 32, local_size_y = 32) in;
void main()
{
	ivec2 cur = ivec2(gl_GlobalInvocationID.xy);
	vec2 res = vec2(0.0);
	
	float tmp = - 2.0 * PI * (cur.y - imageSize(In).y/2) / imageSize(In).x;
	for(int i = 0; i < imageSize(In).x; ++i)
	{
		float arg = tmp * i;
		res += complex_mult(vec2(cos(arg), sin(arg)), vec2(imageLoad(In, ivec2(i, cur.y)).x, 0.0));
	}
	res /= imageSize(In).x; 
		
	imageStore(Out, cur, vec4(res, 0.0, 1.0));
}

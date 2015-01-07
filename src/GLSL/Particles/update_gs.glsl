#version 430

layout(points) in;
layout(points, max_vertices = 1) out;

uniform float time = 0.0;

layout(location = 0) in vec4 inter_position_type[];
layout(location = 1) in vec4 inter_speed_lifetime[];

layout(location = 0) out vec4 position_type;
layout(location = 1) out vec4 speed_lifetime;

float rand(vec2 co)
{
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main()
{
	float lifetime = inter_speed_lifetime[0].w - time;
	
	if(lifetime <= 0.0)
	{
		position_type.xyz = vec3(0.0, 10.0, 0.0);
		position_type.w = inter_position_type[0].w;
		speed_lifetime.xyz = vec3(-2.5 + 5.0 * rand(vec2(lifetime, inter_position_type[0].y)), 
								  5.0,
								  -2.5 + 5.0 * rand(vec2(inter_position_type[0].y, lifetime)));
		speed_lifetime.w = 20.0 * rand(inter_position_type[0].xy);
	} else {
		vec3 speed = inter_speed_lifetime[0].xyz + vec3(0.0, -9.0, 0.0) * time;
		position_type.xyz = inter_position_type[0].xyz + speed * time;
		position_type.w = inter_position_type[0].w;
		speed_lifetime.xyz = speed;
		speed_lifetime.w = lifetime;
	}
	
	if(position_type.y < 0.0)
		speed_lifetime.xyz = 0.75 * reflect(speed_lifetime.xyz, vec3(0.0, 1.0, 0.0));
	
	if(position_type.y < 0.05)
		position_type.y = 0.05;
	
	EmitVertex();
	EndPrimitive();
}

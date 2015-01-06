#version 430

layout(points) in;
layout(points) out;
layout(max_vertices = 1) out;

uniform float time = 0.0;

in layout(location = 0) vec4 in_position_type[];
in layout(location = 1) vec4 in_speed_lifetime[];

out layout(location = 0) vec4 position_type;
out layout(location = 1) vec4 speed_lifetime;

void main()
{
	vec3 speed = in_speed_lifetime[0].xyz + vec3(0.0, -1.0, 0.0) * time;
	position_type.xyz = in_position_type[0].xyz + speed * time;
	position_type.w = in_position_type[0].w;
	speed_lifetime.xyz = speed;
	speed_lifetime.w = in_speed_lifetime[0].w - time;
	
	EmitVertex();
	EndPrimitive();
}

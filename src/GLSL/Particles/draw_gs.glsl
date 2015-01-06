#version 430

layout(points) in;
layout(triangle_strip) out;
layout(max_vertices = 4) out;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform float time = 0.0;
uniform float particle_size = 1.0;

in layout(location = 0) vec4 in_position_type[];
in layout(location = 1) vec4 in_speed_lifetime[];

out layout(location = 0) vec4 position_type;
out layout(location = 1) vec4 speed_lifetime;

void main()
{
	mat4 VP = ProjectionMatrix * ViewMatrix;
	gl_Position = VP * vec4(in_position_type[0].xyz + vec3(-particle_size, -particle_size, 0.0), 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	EmitVertex();
	
	gl_Position = VP * vec4(in_position_type[0].xyz + vec3(particle_size, -particle_size, 0.0), 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	EmitVertex();
	
	gl_Position = VP * vec4(in_position_type[0].xyz + vec3(-particle_size, particle_size, 0.0), 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	EmitVertex();
	
	gl_Position = VP * vec4(in_position_type[0].xyz + vec3(particle_size, particle_size, 0.0), 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	EmitVertex();
	
	EndPrimitive();
}

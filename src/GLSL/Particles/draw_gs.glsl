#version 430

layout(points) in;
layout(triangle_strip) out;
layout(max_vertices = 4) out;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform float particle_size = 0.25;
uniform vec3 cameraRight = vec3(0.0, 0.0, 0.0);
uniform vec3 cameraPosition;

in layout(location = 0) vec4 in_position_type[1];
in layout(location = 1) vec4 in_speed_lifetime[1];

flat out layout(location = 0) vec4 position_type;
flat out layout(location = 1) vec4 speed_lifetime;
out layout(location = 2) vec3 normal;
out layout(location = 3) vec2 texcoord;
out layout(location = 4) vec3 world_pos;

void main()
{
	mat4 VP = ProjectionMatrix * ViewMatrix;
	vec3 n = normalize(cameraPosition - in_position_type[0].xyz);
	vec3 up, right;
	if(cameraRight == vec3(0.0))
	{
		up = vec3(0.0, 1.0, 0.0);
		right = 0.5 * particle_size * normalize(cross(up, n));
		up = 0.5 * particle_size * normalize(cross(right, n));
	} else {
		right = cameraRight;
		up = 0.5 * particle_size * normalize(cross(right, n));
		right *= 0.5 * particle_size;
	}
	
	// Stupidly expending the billboard to ease raytracing, must simpler than figuring out the real exact size.
	right *= 1.5;
	up *= 1.5;
	
	world_pos = in_position_type[0].xyz - up - right;
	gl_Position = VP * vec4(world_pos, 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	normal = n;
	texcoord = vec2(0.0, 0.0);
	EmitVertex();
	
	world_pos = in_position_type[0].xyz - up + right;
	gl_Position = VP * vec4(world_pos, 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	normal = n;
	texcoord = vec2(1.0, 0.0);
	EmitVertex();
	
	world_pos = in_position_type[0].xyz + up - right;
	gl_Position = VP * vec4(world_pos, 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	normal = n;
	texcoord = vec2(0.0, 1.0);
	EmitVertex();
	
	world_pos = in_position_type[0].xyz + up + right;
	gl_Position = VP * vec4(world_pos, 1.0);
	position_type = in_position_type[0];
	speed_lifetime = in_speed_lifetime[0];
	normal = n;
	texcoord = vec2(1.0, 1.0);
	EmitVertex();
	
	EndPrimitive();
}

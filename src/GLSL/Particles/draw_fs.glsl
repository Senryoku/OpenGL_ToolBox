#version 430

uniform float particle_size = 0.25;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform vec3 cameraPosition;

in layout(location = 0) vec4 in_position_type;
in layout(location = 1) vec4 in_speed_lifetime;
in layout(location = 2) vec3 normal;
in layout(location = 3) vec2 texcoord;
in layout(location = 4) vec3 center;

out layout(location = 0) vec4 colorMaterialOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

const float MATERIAL_UNLIT = 2.0;

void main()
{
	//vec3 color = (1.0 - 2.0 * clamp(length(texcoord - 0.5), 0.0, 0.5)) * normalize(in_speed_lifetime.rgb);
	vec3 color = vec3(0.5 + 0.5 * (int(in_position_type.w) % 2), 0.5 + 0.5 * (int(in_position_type.w) % 3), 0.5 + 0.5 * (int(in_position_type.w) % 5)); // TEMP! : vec3(1.0);
	float l = length(texcoord - 0.5);
	if(l > 0.5) color = vec3(0.0);
	if(color == vec3(0.0))
		discard;
	
	float cos_alpha = sqrt(1.0 - 4.0 * l * l);
	
	vec3 n = normalize(normal);
	vec3 true_pos = in_position_type.xyz + 0.5 * particle_size * cos_alpha * n;
	
	// Compute Normal
	n = normalize(true_pos - center);
	color *= max(dot(n, normalize(vec3(1.0, 1.0, 0.0))), 0.1);
	
	// Adjusting depth to add some volume to out billboard sphere!
	vec4 tmp = ProjectionMatrix * ViewMatrix * vec4(true_pos, 1.0);
	gl_FragDepth = ((gl_DepthRange.diff * tmp.z / tmp.w) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
	
	colorMaterialOut = vec4(color, MATERIAL_UNLIT);
	worldPositionOut = vec4(true_pos, gl_FragDepth);
	worldNormalOut = vec4(n, 1.0);
}

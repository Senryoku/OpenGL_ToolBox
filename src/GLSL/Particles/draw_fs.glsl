#version 430

uniform float particle_size = 0.25;

layout(std140) uniform Camera
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform vec3 cameraPosition;

flat in layout(location = 0) vec4 in_position_type;
flat in layout(location = 1) vec4 in_speed_lifetime;
in layout(location = 2) vec3 normal;
in layout(location = 3) vec2 texcoord;
in layout(location = 4) vec3 world_pos;

out layout(location = 0) vec4 colorMaterialOut;
out layout(location = 1) vec4 worldPositionOut;
out layout(location = 2) vec4 worldNormalOut;

const float MATERIAL_UNLIT = 2.0;

bool traceSphere(vec3 ro, vec3 rd, vec3 ce, float r, out vec3 p, out vec3 n)
{	
    vec3 d = ro - ce;
	
	float a = dot(rd, rd);
	float b = dot(rd, d);
	float c = dot(d, d) - r * r;
	
	float g = b*b - a*c;
	
	if(g > 0.0)
    {
		float dis = (-sqrt(g) - b) / a;
		if(dis > 0.0 && dis < 10000)
        {
			p = ro + rd * dis;
			n = (p - ce) / r;
            return true;
		}
	}
    return false;
}

void main()
{
	vec3 color = vec3(0.5 + 0.5 * (int(in_position_type.w) % 2), 0.5 + 0.5 * (int(in_position_type.w) % 3), 0.5 + 0.5 * (int(in_position_type.w) % 5)); // TEMP! : vec3(1.0);
	vec3 p, n;
	if(traceSphere(cameraPosition, normalize(world_pos - cameraPosition), in_position_type.xyz, particle_size * 0.5, p, n))
	{
		color *= max(dot(n, normalize(vec3(1.0, 1.0, 0.0))), 0.1);
	
		vec4 tmp = ProjectionMatrix * ViewMatrix * vec4(p, 1.0);
		gl_FragDepth = ((gl_DepthRange.diff * tmp.z / tmp.w) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
	} else {
		discard;
	}
	
	colorMaterialOut = vec4(color, MATERIAL_UNLIT);
	worldPositionOut = vec4(p, gl_FragDepth);
	worldNormalOut = vec4(n, 1.0);
}

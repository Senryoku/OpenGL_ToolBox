#version 430

layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

layout(std140) uniform Camera {
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
};

uniform unsigned int lightCount = 0;

const mat4 CubeFaceMatrix[6] = mat4[6](
	mat4(
		 0.0,  0.0, -1.0,  0.0,
		 0.0, -1.0,  0.0,  0.0,
		-1.0,  0.0,  0.0,  0.0,
		 0.0,  0.0,  0.0,  1.0 
	), mat4(
		 0.0,  0.0,  1.0,  0.0,
		 0.0, -1.0,  0.0,  0.0,
		 1.0,  0.0,  0.0,  0.0,
		 0.0,  0.0,  0.0,  1.0 
	), mat4(
		 1.0,  0.0,  0.0,  0.0,
		 0.0,  0.0, -1.0,  0.0,
		 0.0,  1.0,  0.0,  0.0,
		 0.0,  0.0,  0.0,  1.0 
	), mat4(
		 1.0,  0.0,  0.0,  0.0,
		 0.0,  0.0,  1.0,  0.0,
		 0.0, -1.0,  0.0,  0.0,
		 0.0,  0.0,  0.0,  1.0 
	), mat4(
		 1.0,  0.0,  0.0,  0.0,
		 0.0, -1.0,  0.0,  0.0,
		 0.0,  0.0, -1.0,  0.0,
		 0.0,  0.0,  0.0,  1.0 
	), mat4(
		-1.0,  0.0,  0.0,  0.0,
		 0.0, -1.0,  0.0,  0.0,
		 0.0,  0.0,  1.0,  0.0,
		 0.0,  0.0,  0.0,  1.0 
	)
);


in layout(location = 0) vec3 in_position[3];
in layout(location = 1) vec3 in_normal[3];
in layout(location = 2) vec2 in_texcoord[3];
in layout(location = 3) vec4 in_shadowcoord[3][8];

out layout(location = 0) vec3 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;
out layout(location = 3) vec4 shadowcoord[8];

void main(void)
{
	for(gl_Layer = 0; gl_Layer != 6; ++gl_Layer)
	{
		for(int i = 0; i != 3; ++i)
		{
			position = vec3(CubeFaceMatrix[gl_Layer] * vec4(in_position[i], 1.0));
			gl_Position = ProjectionMatrix * vec4(position, 1.0);
			gl_PrimitiveID = gl_PrimitiveIDIn;
			normal = in_normal[i];
			texcoord = in_texcoord[i];
			for(int j = 0; j < lightCount; ++j)
				shadowcoord[j] = in_shadowcoord[i][j];
			EmitVertex();
		}
		EndPrimitive();
	}
}

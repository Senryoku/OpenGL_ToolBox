#version 430 core

uniform mat4 MVP = mat4(1.0);

in layout(location = 0) vec3 in_position;
in layout(location = 1) vec3 in_normal;
in layout(location = 2) vec2 in_texcoord;

void main(void)
{
    gl_Position = MVP * vec4(in_position, 1.f);
}

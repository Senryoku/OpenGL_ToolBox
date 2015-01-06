#version 430

layout (location = 0) in vec4 position_type;
layout (location = 1) in vec4 speed_lifetime;

out vec4 out_position_type;
out vec4 out_speed_lifetime;

void main()
{
    out_position_type = position_type;
    out_speed_lifetime = speed_lifetime;
}

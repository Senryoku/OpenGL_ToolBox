#version 430

layout (location = 0) in vec4 position_type;
layout (location = 1) in vec4 speed_lifetime;

layout (location = 0) out vec4 inter_position_type;
layout (location = 1) out vec4 inter_speed_lifetime;
/*
out vec4 inter_position_type;
out vec4 inter_speed_lifetime;
*/
void main()
{
    inter_position_type = position_type;
    inter_speed_lifetime = speed_lifetime;
}

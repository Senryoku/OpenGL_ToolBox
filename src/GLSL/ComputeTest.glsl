#version 440

layout(binding = 0, rgba8) uniform image2D Out;

layout (local_size_x = 32, local_size_y = 32) in;
void main()
{
	imageStore(Out, ivec2(gl_GlobalInvocationID.xy), imageLoad(Out, ivec2(gl_GlobalInvocationID.xy)) + vec4(gl_GlobalInvocationID.xy / 512.0, 0.0, 1.0));
}

uniform sampler2D	Texture;

in vec2 texcoords;

void main(void)
{
    gl_FragColor = texture(Texture, texcoords);
}


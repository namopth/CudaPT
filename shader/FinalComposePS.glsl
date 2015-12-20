#version 330 core

in vec2 outTexCoord;

out vec4 color;

uniform sampler2D hdrBuffer;
uniform float exposure;

void main()
{
	const float gamma = 2.2f;
	vec3 hdrColor = texture(hdrBuffer, outTexCoord).rgb;
	//vec3 result = hdrColor / (hdrColor + vec3(1.0));
	vec3 result = vec3(1.0f) - exp(-hdrColor * exposure);
	result = pow(result, vec3(1.0f / gamma));
	color = vec4(result, 1.0f);
	//color = vec4(hdrColor, 1.0f);
}
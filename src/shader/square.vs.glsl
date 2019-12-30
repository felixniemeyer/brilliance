#version 450

layout(location = 0) in vec2 position; 

layout(location = 0) out vec2 tex_coords; 

layout(push_constant) uniform PushConstantData {
	int time; 
	int dtime; 
} pc;

const float tScale = 0.001; 
const float rotTimeScale = tScale * 3.14 * 0.05; 
const float rOffset = 3.14 * 0.01; 
const float width = 0.2; 
const float wRatio = (0.5 - width) / 0.5; 

mat2 get2dRotMat(float angle) { 
	return mat2(
		cos(angle), -sin(angle), 
		sin(angle), cos(angle)
	); 
}

void main() { 
	

	if (position.y < 0.0) {
		gl_Position = vec4(position * 1.9, 0.0, 1.0); 
	} else {
		gl_Position = vec4(position * wRatio * 1.9, 0.0, wRatio);
	}

	float o = rOffset * (position.x < 0.0 ? -1.0 : 0.0);
	mat2 rot = get2dRotMat(float(pc.time) * rotTimeScale + o);

	vec2 cut = vec2(0, ( position.y - 0.5 ) * width - (0.5 - width));
	cut = rot * cut; 

	// gl_Position = vec4(cut, 0.0, 1.0);

	cut = cut + vec2(0.5);


	tex_coords = cut; 
}


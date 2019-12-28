#version 450

layout(location = 0) in vec2 tex_coords; 

layout(location = 0) out vec4 f_color; 

layout(push_constant) uniform PushConstantData {
	int time; 
} pc;

layout(set = 0, binding = 0) uniform sampler2D tex; 

void main() {
	
	f_color = texture(tex, tex_coords + vec2(sin(pc.time*0.001),0)); 
}

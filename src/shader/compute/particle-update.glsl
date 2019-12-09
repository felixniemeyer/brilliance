#version 450

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in; 

struct Particle {
	vec2 pos; 
	vec2 speed; 
};

layout(set = 0, binding = 0) buffer Particles {
	Particle particle[];
} particles; 

void main() {
	uint idx = gl_GlobalInvocationID.x; 
	float dtime = 1.0 / 60.0; //provide as uniform later on 
	particles.particle[idx].pos += particles.particle[idx].speed * dtime; 
}



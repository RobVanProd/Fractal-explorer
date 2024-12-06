import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import math

# Shader source code
vertex_shader = """
#version 330
in vec3 position;
uniform mat4 transform;
void main() {
    gl_Position = transform * vec4(position, 1.0);
}
"""

fragment_shader = """
#version 330
uniform vec2 resolution;
uniform float time;
out vec4 fragColor;

vec2 complex_square(vec2 z) {
    return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / min(resolution.x, resolution.y);
    
    // Animate the view
    float zoom = 1.5 + sin(time * 0.1) * 0.5;
    vec2 offset = vec2(sin(time * 0.3) * 0.5, cos(time * 0.2) * 0.5);
    
    vec2 c = uv * zoom + offset;
    vec2 z = c;
    
    float iter = 0.0;
    const float MAX_ITER = 100.0;
    
    for(float i = 0.0; i < MAX_ITER; i++) {
        z = complex_square(z) + c;
        if(length(z) > 2.0) {
            iter = i;
            break;
        }
        iter = MAX_ITER;
    }
    
    vec3 color = vec3(0.0);
    if(iter < MAX_ITER) {
        float hue = iter / MAX_ITER;
        // Create a rainbow color effect
        color = 0.5 + 0.5 * cos(6.28318 * (hue + vec3(0.0, 0.33, 0.67)));
    }
    
    fragColor = vec4(color, 1.0);
}
"""

class FractalRenderer:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Create window
        self.window = glfw.create_window(800, 600, "3D Fractal Renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        
        # Create shaders
        vertex_shader_id = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        fragment_shader_id = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vertex_shader_id, fragment_shader_id)
        
        # Create a full-screen quad
        vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
             1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)
        
        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Set vertex attributes
        position_location = glGetAttribLocation(self.shader_program, "position")
        glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(position_location)
        
        # Get uniform locations
        self.transform_loc = glGetUniformLocation(self.shader_program, "transform")
        self.resolution_loc = glGetUniformLocation(self.shader_program, "resolution")
        self.time_loc = glGetUniformLocation(self.shader_program, "time")
        
        # Initialize time
        self.start_time = glfw.get_time()
    
    def run(self):
        while not glfw.window_should_close(self.window):
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        
        self.cleanup()
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(self.shader_program)
        
        # Update uniforms
        width, height = glfw.get_window_size(self.window)
        glUniform2f(self.resolution_loc, width, height)
        current_time = glfw.get_time() - self.start_time
        glUniform1f(self.time_loc, current_time)
        
        # Create transformation matrix (identity for now)
        transform = np.identity(4, dtype=np.float32)
        glUniformMatrix4fv(self.transform_loc, 1, GL_FALSE, transform)
        
        # Draw
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    
    def cleanup(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteProgram(self.shader_program)
        glfw.terminate()

if __name__ == "__main__":
    try:
        renderer = FractalRenderer()
        renderer.run()
    except Exception as e:
        print(f"Error: {e}")
        glfw.terminate()

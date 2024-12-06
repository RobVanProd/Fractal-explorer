import pygame
import numpy as np
from numba import jit
import math
import colorsys
from pygame.locals import *

@jit(nopython=True)
def compute_fractal(width, height, max_iter, center_x, center_y, zoom, pixel_size, rotation):
    result = np.zeros((height, width), dtype=np.float64)
    orbit_x = np.zeros((height, width), dtype=np.float64)
    orbit_y = np.zeros((height, width), dtype=np.float64)
    edge_x = center_x
    edge_y = center_y
    max_diff = 0
    
    # Precompute rotation
    cos_rot = math.cos(rotation)
    sin_rot = math.sin(rotation)
    
    for y in range(height):
        for x in range(width):
            # Apply rotation to coordinates
            dx = (x - width/2) * pixel_size / zoom
            dy = (y - height/2) * pixel_size / zoom
            real = center_x + (dx * cos_rot - dy * sin_rot)
            imag = center_y + (dx * sin_rot + dy * cos_rot)
            
            c = real + imag * 1j
            z = 0j
            
            # Track last orbit position for trails
            last_x = 0
            last_y = 0
            
            for i in range(max_iter):
                z = z*z + c
                if abs(z) > 2.0:
                    # Smooth coloring formula
                    log_zn = np.log(abs(z))
                    nu = np.log(log_zn/np.log(2)) / np.log(2)
                    result[y, x] = i + 1 - nu
                    
                    # Store last orbit position
                    orbit_x[y, x] = last_x
                    orbit_y[y, x] = last_y
                    
                    # Check if this is a more interesting edge point
                    if x > 0 and abs(result[y, x] - result[y, x-1]) > max_diff:
                        max_diff = abs(result[y, x] - result[y, x-1])
                        edge_x = real
                        edge_y = imag
                    break
                
                # Store last position for orbit trails
                last_x = z.real
                last_y = z.imag
            else:
                result[y, x] = max_iter
                orbit_x[y, x] = last_x
                orbit_y[y, x] = last_y
    
    return result, edge_x, edge_y, orbit_x, orbit_y

class FractalRenderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Infinite Fractal Zoom")
        
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        
        # Fractal parameters
        self.max_iter = 100
        self.zoom = 1.0
        self.center_x = -0.7453
        self.center_y = 0.1127
        self.rotation = 0.0
        self.rotation_speed = 0.001
        
        # Smooth movement parameters
        self.target_x = self.center_x
        self.target_y = self.center_y
        self.velocity_x = 0
        self.velocity_y = 0
        self.edge_points = []
        self.max_edge_points = 30
        
        # Trail effect
        self.trail_surface = pygame.Surface((width, height))
        self.trail_surface.set_alpha(128)
        
        # Dynamic color parameters
        self.color_offset = 0
        self.color_speed = 0.5
        
        # Improved dynamic resolution control
        self.scale_factor = 4
        self.min_scale_factor = 2
        self.max_scale_factor = 8
        self.target_fps = 30
        self.fps_history = []
        self.resolution_cooldown = 0
        
        # Create enhanced color table
        self.update_color_table()
        
        # Warm up JIT
        _ = compute_fractal(10, 10, 50, 0, 0, 1.0, 4.0/10, 0.0)

    def update_color_table(self):
        self.color_table = np.zeros((self.max_iter + 1, 3), dtype=np.uint8)
        
        # Create a psychedelic color palette
        palette = [
            (1.0, 0.0, 0.5),  # Hot pink
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.7, 0.0),  # Gold
            (0.5, 0.0, 1.0),  # Purple
            (0.0, 1.0, 0.5),  # Emerald
            (1.0, 0.0, 0.5),  # Back to pink
        ]
        
        for i in range(self.max_iter + 1):
            if i < self.max_iter:
                # Create smooth transitions between colors
                phase = ((i % 64) / 64.0 + self.color_offset) % 1.0
                palette_index = phase * (len(palette) - 1)
                base_index = int(palette_index)
                fract = palette_index - base_index
                
                # Interpolate between colors
                r = palette[base_index][0] * (1 - fract) + palette[base_index + 1][0] * fract
                g = palette[base_index][1] * (1 - fract) + palette[base_index + 1][1] * fract
                b = palette[base_index][2] * (1 - fract) + palette[base_index + 1][2] * fract
                
                # Add wave effects
                t = i * 0.1
                wave = math.sin(phase * 6.28318 + self.color_offset * 10)
                r = r * (0.7 + 0.3 * wave)
                g = g * (0.7 + 0.3 * math.sin(wave + 2.09439))
                b = b * (0.7 + 0.3 * math.sin(wave + 4.18879))
                
                # Ensure colors stay in valid range
                r = max(0, min(1, r))
                g = max(0, min(1, g))
                b = max(0, min(1, b))
            else:
                r, g, b = 0, 0, 0
            
            self.color_table[i] = np.array([int(r * 255), int(g * 255), int(b * 255)])

    def update(self):
        current_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        
        # Update rotation
        self.rotation += self.rotation_speed
        
        # Update color cycling
        self.color_offset += self.color_speed * 0.01
        self.update_color_table()
        
        # Slower zoom for more stability
        self.zoom = math.exp(current_time * 0.2)
        
        # More gradual iteration increase
        new_max_iter = int(100 + math.log(self.zoom) * 15)
        if new_max_iter > self.max_iter:
            self.max_iter = new_max_iter
            self.update_color_table()
        
        # Calculate dimensions
        w = max(1, int(self.width / self.scale_factor))
        h = max(1, int(self.height / self.scale_factor))
        
        # Compute fractal with rotation and get orbit data
        iterations, edge_x, edge_y, orbit_x, orbit_y = compute_fractal(
            w, h, self.max_iter, self.center_x, self.center_y,
            self.zoom, 4.0/w, self.rotation
        )
        
        # Add new edge point to history
        self.edge_points.append((edge_x, edge_y))
        if len(self.edge_points) > self.max_edge_points:
            self.edge_points.pop(0)
        
        # Calculate average edge point
        if self.edge_points:
            avg_x = sum(x for x, _ in self.edge_points) / len(self.edge_points)
            avg_y = sum(y for _, y in self.edge_points) / len(self.edge_points)
            
            # Update target with smoothed edge point
            self.target_x = avg_x
            self.target_y = avg_y
        
        # Apply momentum-based movement
        dx = (self.target_x - self.center_x)
        dy = (self.target_y - self.center_y)
        
        # Update velocities with damping
        self.velocity_x = self.velocity_x * 0.9 + dx * 0.02
        self.velocity_y = self.velocity_y * 0.9 + dy * 0.02
        
        # Apply velocities to position
        self.center_x += self.velocity_x
        self.center_y += self.velocity_y
        
        # Get current FPS and update history
        fps = self.clock.get_fps()
        self.fps_history.append(fps)
        if len(self.fps_history) > 10:  # Keep last 10 FPS readings
            self.fps_history.pop(0)
        
        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else fps
        
        # Update resolution with cooldown and smoothing
        if self.resolution_cooldown > 0:
            self.resolution_cooldown -= 1
        elif avg_fps > 0:  # Avoid division by zero
            if avg_fps > self.target_fps + 5 and self.scale_factor > self.min_scale_factor:
                self.scale_factor = max(self.min_scale_factor, self.scale_factor - 0.5)
                self.resolution_cooldown = 30  # Wait 30 frames before next change
            elif avg_fps < self.target_fps - 5 and self.scale_factor < self.max_scale_factor:
                self.scale_factor = min(self.max_scale_factor, self.scale_factor + 0.5)
                self.resolution_cooldown = 30
        
        # Update window title with more info
        pygame.display.set_caption(f"Fractal Zoom - FPS: {int(avg_fps)} - Scale: 1/{int(self.scale_factor)} - Zoom: {self.zoom:.1f}x")
        
        # Convert iterations to colors using clipped indices
        iterations = np.clip(iterations, 0, self.max_iter).astype(np.int32)
        pixels = self.color_table[iterations]
        
        # Create and scale surface
        surf = pygame.surfarray.make_surface(pixels)
        scaled = pygame.transform.scale(surf, (self.width, self.height))
        
        # Draw trail effect
        self.trail_surface.blit(scaled, (0, 0))
        self.screen.blit(self.trail_surface, (0, 0))
    
    def run(self):
        running = True
        frames = 0
        last_time = pygame.time.get_ticks()
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
            
            self.update()
            pygame.display.flip()
            
            # Update FPS counter every second
            frames += 1
            current_time = pygame.time.get_ticks()
            if current_time - last_time > 1000:
                fps = frames * 1000 / (current_time - last_time)
                frames = 0
                last_time = current_time
            
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    renderer = FractalRenderer(800, 600)
    renderer.run()

import pygame
import numpy as np
from pygame.locals import *
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from numba import jit, prange

@jit(nopython=True, parallel=True)
def compute_fractal_numba(width, height, max_iter, center_x, center_y, zoom):
    x = np.linspace(center_x - 2/zoom, center_x + 2/zoom, width)
    y = np.linspace(center_y - 2/zoom * height/width, center_y + 2/zoom * height/width, height)
    
    divtime = np.zeros((height, width), dtype=np.float32)
    
    for i in prange(height):
        for j in range(width):
            c = x[j] + 1j * y[i]
            z = 0.0 + 0.0j
            for k in range(max_iter):
                if abs(z) > 2.0:
                    divtime[i, j] = k + 1 - np.log(np.log(abs(z))) / np.log(2)
                    break
                z = z*z + c
    return divtime

class FractalRenderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fast Fractal Explorer")
        
        # Create surfaces for transition blending
        self.main_surface = pygame.Surface((width, height))
        self.prev_surface = pygame.Surface((width, height))
        self.transition_surface = pygame.Surface((width, height))
        
        # Initialize scale factor
        self.scale_factor = 2
        self.min_scale_factor = 1
        self.max_scale_factor = 4
        self.scaled_width = width // self.scale_factor
        self.scaled_height = height // self.scale_factor
        
        # Create scaled surfaces
        self.scaled_surface = pygame.Surface((self.scaled_width, self.scaled_height))
        self.back_buffer = pygame.Surface((width, height))
        
        # Edge following parameters
        self.edge_points = []
        self.max_edge_points = 7
        self.movement_smoothing = 0.8
        self.velocity_x = 0
        self.velocity_y = 0
        
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        
        # Fractal parameters
        self.max_iter = 50
        self.zoom = 1.0
        self.center_x = -0.5
        self.center_y = 0.0
        self.zoom_speed = 1.016
        self.min_zoom_speed = 1.008
        self.max_zoom_speed = 1.08
        
        # Zoom reset parameters
        self.max_zoom = 1e3
        self.min_edge_strength = 0.5
        self.reset_countdown = 0
        self.min_edge_count = 6
        
        # Zoom reset parameters
        self.interesting_points = [
            (-0.5, 0.0),     # Main cardioid
            (-0.75, 0.0),    # Period-2 bulb
            (-0.12, 0.75),   # Upper valley
            (-1.25, 0.0),    # Left antenna
            (0.25, 0.0),     # Right edge
            (-0.75, 0.1),    # Upper bulb
            (-0.75, -0.1),   # Lower bulb
            (-1.0, 0.25),    # Spiral
            (-0.16, 1.04),   # Upper filament
            (-0.4, 0.6),     # Mini spiral
        ]
        self.current_point_index = 0
        
        # Transition parameters
        self.is_transitioning = False
        self.transition_progress = 0.0
        self.transition_speed = 0.05
        self.target_x = 0.0
        self.target_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.start_zoom = 1.0
        
        # Auto navigation
        self.auto_navigate = True
        
        # Color parameters
        self.color_offset = 0.0
        self.color_pulse_speed = 0.02
        self.color_pulse_enabled = False
        
        # Performance optimization
        self.fps_history = []
        self.frame_count = 0
        
        # Create color table with pre-computed values
        self.color_table = np.zeros((self.max_iter + 1, 3), dtype=np.uint8)
        self.update_color_table()
        
        # Threading
        self.num_threads = max(1, multiprocessing.cpu_count() - 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
    
    def update_color_table(self):
        palettes = [
            # Electric
            [(0,0,0), (32,0,255), (64,0,255), (255,128,0), (255,255,255)],
            # Fire
            [(0,0,0), (128,0,0), (255,64,0), (255,255,0), (255,255,255)],
            # Ocean
            [(0,0,32), (0,64,128), (0,128,255), (128,255,255), (255,255,255)]
        ]
        
        palette = palettes[0 % len(palettes)]
        num_colors = len(palette)
        
        for i in range(self.max_iter + 1):
            if i < self.max_iter:
                phase = ((i % 32) / 32.0 + self.color_offset) % 1.0
                idx = phase * (num_colors - 1)
                base_idx = int(idx)
                fract = idx - base_idx
                
                if base_idx < num_colors - 1:
                    r = int(palette[base_idx][0] * (1 - fract) + palette[base_idx + 1][0] * fract)
                    g = int(palette[base_idx][1] * (1 - fract) + palette[base_idx + 1][1] * fract)
                    b = int(palette[base_idx][2] * (1 - fract) + palette[base_idx + 1][2] * fract)
                else:
                    r, g, b = palette[base_idx]
            else:
                r, g, b = 0, 0, 0
            
            self.color_table[i] = np.array([r, g, b])
    
    def update_scaled_surface(self):
        self.scaled_width = self.width // self.scale_factor
        self.scaled_height = self.height // self.scale_factor
        self.scaled_surface = pygame.Surface((self.scaled_width, self.scaled_height))
    
    def should_reset(self, gradient_mag):
        if self.zoom > self.max_zoom:
            return True
            
        # Check if we've lost detail
        max_gradient = np.max(gradient_mag)
        if max_gradient < self.min_edge_strength:
            self.reset_countdown += 1
            return self.reset_countdown > 2
        
        # Count strong edges
        strong_edges = np.sum(gradient_mag > self.min_edge_strength * max_gradient)
        if strong_edges < self.min_edge_count:
            self.reset_countdown += 1
            return self.reset_countdown > 2
        
        self.reset_countdown = 0
        return False
    
    def reset_to_new_location(self):
        # Store current view before transitioning
        self.prev_surface.blit(self.main_surface, (0, 0))
        
        # Save current color and zoom settings
        old_zoom_speed = self.zoom_speed
        old_color_offset = self.color_offset
        
        # Pick next interesting point
        self.current_point_index = (self.current_point_index + 1) % len(self.interesting_points)
        new_x, new_y = self.interesting_points[self.current_point_index]
        
        # Start transition
        self.is_transitioning = True
        self.transition_progress = 0.0
        self.start_x = self.center_x
        self.start_y = self.center_y
        self.target_x = new_x
        self.target_y = new_y
        self.start_zoom = self.zoom
        self.zoom = 1.0  # Will be interpolated
        
        # Restore settings
        self.zoom_speed = old_zoom_speed
        self.color_offset = old_color_offset
    
    def update(self):
        if self.is_transitioning:
            # Smooth transition using easing function
            self.transition_progress = min(1.0, self.transition_progress + self.transition_speed)
            t = 1 - (1 - self.transition_progress) * (1 - self.transition_progress)  # Ease out quad
            
            self.center_x = self.start_x + (self.target_x - self.start_x) * t
            self.center_y = self.start_y + (self.target_y - self.start_y) * t
            self.zoom = self.start_zoom * (1 - t) + 2.0 * t  # Transition to zoom level 2.0
            
            # Compute current view
            iterations = compute_fractal_numba(self.scaled_width, self.scaled_height, 
                                            self.max_iter, self.center_x, self.center_y, self.zoom)
            
            # Draw current view to scaled surface
            pixels = self.color_table[iterations.astype(np.int32) % len(self.color_table)]
            scaled_pixels = pygame.surfarray.pixels3d(self.scaled_surface)
            scaled_pixels[:] = pixels.swapaxes(0, 1)
            del scaled_pixels  # Release the surface lock
            
            # Scale up to main surface
            pygame.transform.scale(self.scaled_surface, (self.width, self.height), self.main_surface)
            
            # Blend between previous and current view
            self.transition_surface.blit(self.prev_surface, (0, 0))
            self.main_surface.set_alpha(int(255 * t))
            self.transition_surface.blit(self.main_surface, (0, 0))
            self.screen.blit(self.transition_surface, (0, 0))
            
            if self.transition_progress >= 1.0:
                self.is_transitioning = False
                self.reset_countdown = 0
                self.main_surface.set_alpha(255)
                return
            
            pygame.display.flip()
            return
            
        if self.auto_navigate and not self.is_transitioning:
            self.zoom *= self.zoom_speed
        
        if self.color_pulse_enabled:
            self.color_offset += self.color_pulse_speed
            self.update_color_table()
        
        # Adaptive resolution based on frame rate
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            avg_frame_time = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 33
            current_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 30
            
            old_scale = self.scale_factor
            if current_fps < 25:
                self.scale_factor = min(self.scale_factor + 1, self.max_scale_factor)
            elif current_fps > 35 and self.scale_factor > self.min_scale_factor:
                self.scale_factor = max(self.scale_factor - 1, self.min_scale_factor)
            
            if old_scale != self.scale_factor:
                self.update_scaled_surface()
        
        iterations = compute_fractal_numba(self.scaled_width, self.scaled_height, 
                                         self.max_iter, self.center_x, self.center_y, self.zoom)
        
        if self.auto_navigate and not self.is_transitioning:
            gradient_x = np.abs(np.diff(iterations, axis=1))
            gradient_y = np.abs(np.diff(iterations, axis=0))
            
            gradient_mag = np.zeros_like(iterations)
            gradient_mag[:-1, :] += gradient_y * 1.5
            gradient_mag[:, :-1] += gradient_x * 1.2
            
            y_coords, x_coords = np.indices(gradient_mag.shape)
            center_y, center_x = gradient_mag.shape[0] // 2, gradient_mag.shape[1] // 2
            distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            gradient_mag *= 1 / (1 + distance * 0.1)
            
            # Check if we should reset
            if self.should_reset(gradient_mag):
                self.reset_to_new_location()
                return
            
            # Add threshold to avoid following weak edges
            threshold = np.max(gradient_mag) * 0.3
            gradient_mag[gradient_mag < threshold] = 0
            
            # Find strongest edges
            flat_indices = np.argpartition(gradient_mag.ravel(), -7)[-7:]
            y_coords, x_coords = np.unravel_index(flat_indices, gradient_mag.shape)
            
            # Convert to complex plane coordinates
            x_complex = (x_coords - self.scaled_width/2) / (self.scaled_width/4) / self.zoom + self.center_x
            y_complex = (y_coords - self.scaled_height/2) / (self.scaled_height/4) / self.zoom + self.center_y
            
            # Filter points outside main set
            new_points = [(x, y) for x, y in zip(x_complex, y_complex) 
                         if abs(x) < 2 and abs(y) < 2]
            
            if new_points:
                self.edge_points.extend(new_points)
                self.edge_points = self.edge_points[-self.max_edge_points:]
                
                # Weight recent points more heavily
                weights = np.exp(np.linspace(0, 2.5, len(self.edge_points)))
                self.target_x = np.average([x for x, _ in self.edge_points], weights=weights)
                self.target_y = np.average([y for _, y in self.edge_points], weights=weights)
                
                # Smooth movement
                dx = (self.target_x - self.center_x)
                dy = (self.target_y - self.center_y)
                
                self.velocity_x = self.velocity_x * self.movement_smoothing + dx * (1 - self.movement_smoothing)
                self.velocity_y = self.velocity_y * self.movement_smoothing + dy * (1 - self.movement_smoothing)
                
                self.center_x += self.velocity_x * 0.08
                self.center_y += self.velocity_y * 0.08
        
        # Draw the fractal
        pixels = self.color_table[iterations.astype(np.int32) % len(self.color_table)]
        scaled_pixels = pygame.surfarray.pixels3d(self.scaled_surface)
        scaled_pixels[:] = pixels.swapaxes(0, 1)
        del scaled_pixels  # Release the surface lock
        
        # Scale up to main surface
        pygame.transform.scale(self.scaled_surface, (self.width, self.height), self.main_surface)
        self.screen.blit(self.main_surface, (0, 0))
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_TAB:
                        self.auto_navigate = not self.auto_navigate
                        if not self.auto_navigate:
                            self.velocity_x = 0
                            self.velocity_y = 0
                    elif event.key == K_SPACE:
                        pass
                    elif event.key == K_s:
                        self.color_pulse_enabled = not self.color_pulse_enabled
                    elif event.key == K_UP:
                        self.zoom_speed = min(self.zoom_speed * 1.08, self.max_zoom_speed)
                    elif event.key == K_DOWN:
                        self.zoom_speed = max(self.zoom_speed / 1.08, self.min_zoom_speed)
            
            self.update()
        
        pygame.quit()

if __name__ == "__main__":
    renderer = FractalRenderer(800, 600)
    renderer.run()

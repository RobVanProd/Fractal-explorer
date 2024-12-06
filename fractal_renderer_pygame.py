import pygame
import numpy as np
from pygame.locals import *
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def compute_fractal_chunk(start_y, end_y, width, height, max_iter, center_x, center_y, zoom):
    chunk_height = end_y - start_y
    x = np.linspace(center_x - 2/zoom, center_x + 2/zoom, width)
    y = np.linspace(center_y - 2/zoom * height/width + (4/zoom * height/width * start_y)/height, 
                    center_y - 2/zoom * height/width + (4/zoom * height/width * end_y)/height, 
                    chunk_height)
    
    c = x[None, :] + 1j * y[:, None]
    z = np.zeros_like(c)
    divtime = np.zeros_like(z, dtype=np.float32)
    
    for i in range(max_iter):
        mask = np.abs(z) <= 2
        if not np.any(mask):
            break
        z[mask] = z[mask]**2 + c[mask]
        divtime[mask & (np.abs(z) > 2)] = i + 1 - np.log(np.log(np.abs(z[mask & (np.abs(z) > 2)]))) / np.log(2)
    
    return divtime

class FractalRenderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fast Fractal Explorer")
        
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        
        # Fractal parameters
        self.max_iter = 50  # Reduced max iterations
        self.zoom = 1.0
        self.center_x = -0.5
        self.center_y = 0.0
        self.zoom_speed = 1.1
        
        # Edge following
        self.target_x = self.center_x
        self.target_y = self.center_y
        self.velocity_x = 0
        self.velocity_y = 0
        self.edge_points = []
        self.max_edge_points = 10
        self.auto_navigate = True
        
        # Performance optimization
        self.scale_factor = 2
        self.min_scale_factor = 2
        self.max_scale_factor = 4
        self.fps_history = []
        
        # Threading
        self.num_threads = max(1, multiprocessing.cpu_count() - 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
        
        # Pre-allocate surfaces
        self.scaled_width = width // self.scale_factor
        self.scaled_height = height // self.scale_factor
        self.scaled_surface = pygame.Surface((self.scaled_width, self.scaled_height))
        self.full_surface = pygame.Surface((width, height))
        
        # Create color table
        self.color_table = np.zeros((self.max_iter + 1, 3), dtype=np.uint8)
        self.update_color_table()
    
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
                phase = ((i % 32) / 32.0 + 0) % 1.0
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
    
    def compute_fractal_parallel(self, w, h):
        chunk_size = h // self.num_threads
        futures = []
        
        for i in range(self.num_threads):
            start_y = i * chunk_size
            end_y = start_y + chunk_size if i < self.num_threads - 1 else h
            
            future = self.thread_pool.submit(
                compute_fractal_chunk,
                start_y, end_y, w, h, self.max_iter,
                self.center_x, self.center_y, self.zoom
            )
            futures.append(future)
        
        iterations = np.zeros((h, w), dtype=np.float32)
        
        for i, future in enumerate(futures):
            chunk_result = future.result()
            start_y = i * chunk_size
            end_y = start_y + chunk_size if i < self.num_threads - 1 else h
            iterations[start_y:end_y] = chunk_result
        
        return iterations
    
    def find_interesting_points(self, iterations):
        # Simplified gradient calculation
        gradient_x = np.abs(np.diff(iterations[::2, ::2], axis=1))
        gradient_y = np.abs(np.diff(iterations[::2, ::2], axis=0))
        gradient_mag = gradient_x[:-1, :] + gradient_y[:, :-1]
        
        # Find top gradient points
        flat_indices = np.argpartition(gradient_mag.ravel(), -5)[-5:]
        y_coords, x_coords = np.unravel_index(flat_indices, gradient_mag.shape)
        
        # Convert to complex plane coordinates
        x_complex = (x_coords * 2 - self.width/2) / (self.width/4) / self.zoom + self.center_x
        y_complex = (y_coords * 2 - self.height/2) / (self.height/4) / self.zoom + self.center_y
        
        return list(zip(x_complex, y_complex))
    
    def update(self):
        # Update parameters
        if self.auto_navigate:
            self.zoom *= self.zoom_speed
        #self.color_offset += 0.01
        
        # Compute fractal at reduced resolution
        w = self.scaled_width
        h = self.scaled_height
        
        iterations = self.compute_fractal_parallel(w, h)
        
        # Edge following with reduced frequency
        if self.auto_navigate and len(self.fps_history) % 2 == 0:
            new_points = self.find_interesting_points(iterations)
            if new_points:
                self.edge_points.extend(new_points)
                self.edge_points = self.edge_points[-self.max_edge_points:]
                
                weights = np.linspace(0.5, 1.0, len(self.edge_points))
                self.target_x = np.average([x for x, _ in self.edge_points], weights=weights)
                self.target_y = np.average([y for _, y in self.edge_points], weights=weights)
                
                dx = (self.target_x - self.center_x)
                dy = (self.target_y - self.center_y)
                
                self.velocity_x = self.velocity_x * 0.9 + dx * 0.1
                self.velocity_y = self.velocity_y * 0.9 + dy * 0.1
                
                self.center_x += self.velocity_x * 0.1
                self.center_y += self.velocity_y * 0.1
        
        # Apply colors and scale
        pixels = self.color_table[np.clip(iterations, 0, self.max_iter).astype(int)]
        scaled_array = pygame.surfarray.pixels3d(self.scaled_surface)
        scaled_array[:] = pixels.swapaxes(0, 1)
        del scaled_array  # Release the surface lock
        
        if self.scale_factor > 1:
            pygame.transform.scale(self.scaled_surface, (self.width, self.height), self.full_surface)
            self.screen.blit(self.full_surface, (0, 0))
        else:
            self.screen.blit(self.scaled_surface, (0, 0))
        
        pygame.display.flip()
        
        # Performance monitoring
        frame_time = self.clock.tick(60)
        self.fps_history.append(frame_time)
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)
            avg_frame_time = sum(self.fps_history) / len(self.fps_history)
            
            if avg_frame_time > 33:  # Below 30 FPS
                self.scale_factor = min(self.scale_factor + 1, self.max_scale_factor)
            elif avg_frame_time < 16 and self.scale_factor > self.min_scale_factor:  # Above 60 FPS
                self.scale_factor = max(self.scale_factor - 1, self.min_scale_factor)
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_SPACE:
                        self.update_color_table()
                    elif event.key == K_UP:
                        self.zoom_speed *= 1.1
                    elif event.key == K_DOWN:
                        self.zoom_speed /= 1.1
                    elif event.key == K_TAB:
                        self.auto_navigate = not self.auto_navigate
                        if not self.auto_navigate:
                            self.zoom_speed = 1.0
            
            self.update()
        
        pygame.quit()

if __name__ == "__main__":
    renderer = FractalRenderer(800, 600)
    renderer.run()

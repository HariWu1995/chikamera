from tqdm import tqdm

import math
import random as rd

from src.camera.coverage import evaluate_coverage


def simulated_annealing(grid_points, camera_coords, camera_range: int,
                        iterations: int = 1000, temp: int = 100):
    """
    Refines camera placement using simulated annealing.
    """
    best_solution = camera_coords[:]
    best_coverage = evaluate_coverage(best_solution, camera_range)
    
    pgbar = tqdm(range(iterations))
    for _ in pgbar:
        pgbar.set_description(f'temperature = {temp:.5f} - coverage = {best_coverage:.3f}')

        new_solution = best_solution[:]
        index = rd.randint(0, len(new_solution) - 1)
        new_solution[index] = rd.choice(grid_points)
        
        new_coverage = evaluate_coverage(new_solution, camera_range)
        delta = new_coverage - best_coverage
        
        if delta > 0 or math.exp(delta / temp) > rd.random():
            best_solution = new_solution[:]
            best_coverage = new_coverage
        
        # Reduce temperature
        temp *= 0.99
    
    return best_solution


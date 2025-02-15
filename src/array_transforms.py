from typing import List, Callable, Tuple
import numpy as np
import argparse
import json

# Basic Array Transformations (all preserve size)
def shift_right(arr: np.ndarray) -> np.ndarray:
    """Shift array right by 1, wrapping around"""
    return np.roll(arr, 1)

def shift_left(arr: np.ndarray) -> np.ndarray:
    """Shift array left by 1, wrapping around"""
    return np.roll(arr, -1)

def local_max(arr: np.ndarray) -> np.ndarray:
    """Replace each element with max of it and its neighbors"""
    return np.maximum(arr, np.maximum(shift_left(arr), shift_right(arr)))

def local_min(arr: np.ndarray) -> np.ndarray:
    """Replace each element with min of it and its neighbors"""
    return np.minimum(arr, np.minimum(shift_left(arr), shift_right(arr)))

def scale_up(arr: np.ndarray) -> np.ndarray:
    """Multiply each element by 2"""
    return arr * 2

def scale_down(arr: np.ndarray) -> np.ndarray:
    """Divide each element by 2"""
    return arr // 2

def threshold(arr: np.ndarray) -> np.ndarray:
    """Set values > mean to 1, others to 0"""
    return (arr > arr.mean()).astype(int)

# Function Composition Generator
class CompositionGenerator:
    def __init__(self, base_functions: List[Callable]):
        self.functions = base_functions
        
    def generate_composition(self, depth: int) -> Tuple[Callable, List[str]]:
        """Generate a random composition of given depth"""
        funcs = []
        names = []
        
        for _ in range(depth):
            f = np.random.choice(self.functions)
            funcs.append(f)
            names.append(f.__name__)
            
        def composed(arr: np.ndarray) -> np.ndarray:
            result = arr.copy()
            for f in funcs:
                result = f(result)
            return result
            
        return composed, names

# Dataset Generator
def generate_dataset(
    size: int,
    array_length: int,
    max_depth: int,
    generator: CompositionGenerator
) -> List[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Generate dataset of (input, output, composition_path) tuples"""
    dataset = []
    
    for _ in range(size):
        # Generate random input array
        input_arr = np.random.randint(0, 10, size=array_length)
        
        # Generate random composition
        depth = np.random.randint(1, max_depth + 1)
        composed_func, composition_path = generator.generate_composition(depth)
        
        # Generate output
        output_arr = composed_func(input_arr)
        
        dataset.append((input_arr, output_arr, composition_path))
    
    return dataset

def save_dataset(dataset: List[Tuple[np.ndarray, np.ndarray, List[str]]], filename: str):
    """Save dataset to JSON file."""
    serializable_dataset = [
        (input_arr.tolist(), output_arr.tolist(), path)
        for input_arr, output_arr, path in dataset
    ]
    with open(filename, 'w') as f:
        json.dump(serializable_dataset, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate array transformation datasets')
    parser.add_argument('--size', type=int, default=20,
                      help='Number of examples in dataset')
    parser.add_argument('--length', type=int, default=8,
                      help='Length of input/output arrays')
    parser.add_argument('--max-depth', type=int, default=6,
                      help='Maximum depth of function compositions')
    parser.add_argument('--output', type=str, default='dataset.json',
                      help='Output JSON file')
    parser.add_argument('--verbose', action='store_true',
                      help='Print example transformations')
    
    args = parser.parse_args()
    
    # Create base function list
    base_functions = [
        shift_right, shift_left,
        local_max, local_min,
        scale_up, scale_down,
        threshold
    ]
    
    # Create generator
    generator = CompositionGenerator(base_functions)
    
    # Generate dataset
    dataset = generate_dataset(
        size=args.size,
        array_length=args.length,
        max_depth=args.max_depth,
        generator=generator
    )
    
    # Save dataset
    save_dataset(dataset, args.output)
    print(f"Dataset saved to {args.output}")
    
    # Print examples if verbose
    if args.verbose:
        for i, (input_arr, output_arr, path) in enumerate(dataset[:3]):
            print(f"\nExample {i+1}:")
            print(f"Input:  {input_arr}")
            print(f"Output: {output_arr}")
            print(f"Composition: {' -> '.join(path)}")

if __name__ == "__main__":
    main() 
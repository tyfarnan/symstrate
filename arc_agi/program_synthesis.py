import json
from pathlib import Path
import numpy as np
from arc_dsl import *

def load_function_sequences(json_path):
    """Load and parse function sequences from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_program(function_sequence):
    """
    Convert a function sequence into an executable program using arc_dsl functions.
    
    Args:
        function_sequence (list): List of function names and parameters
    
    Returns:
        callable: A function that takes a grid and returns transformed grid
    """
    def program(grid):
        result = np.array(grid)
        for func_name in function_sequence:
            if func_name == "fill_pattern":
                result = fill_pattern(result)
            elif func_name == "extend_pattern":
                result = extend_pattern(result)
            # Add more function mappings as needed
        return result
    return program

def test_program(program, examples):
    """
    Test a synthesized program against a set of examples.
    
    Args:
        program (callable): The synthesized program
        examples (list): List of input/output example pairs
    
    Returns:
        dict: Results including success rate and failed cases
    """
    successes = 0
    failed_cases = []
    
    for i, example in enumerate(examples):
        input_grid = np.array(example['input'])
        expected_output = np.array(example['output'])
        
        try:
            actual_output = program(input_grid)
            if np.array_equal(actual_output, expected_output):
                successes += 1
            else:
                failed_cases.append({
                    'example_index': i,
                    'input': input_grid.tolist(),
                    'expected': expected_output.tolist(),
                    'actual': actual_output.tolist()
                })
        except Exception as e:
            failed_cases.append({
                'example_index': i,
                'input': input_grid.tolist(),
                'error': str(e)
            })
    
    return {
        'success_rate': successes / len(examples),
        'total_examples': len(examples),
        'successful_examples': successes,
        'failed_cases': failed_cases
    }

def synthesize_and_test_programs(function_seq_path, training_dir):
    """
    Main function to synthesize programs from sequences and test them.
    
    Args:
        function_seq_path (str): Path to function sequences JSON
        training_dir (str): Directory containing training examples
    """
    # Load function sequences
    sequences = load_function_sequences(function_seq_path)
    
    # Load training examples
    training_path = Path(training_dir)
    results = {}
    
    for json_path in training_path.glob('*.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        task_name = json_path.stem
        print(f"\nProcessing task: {task_name}")
        
        # Try each function sequence
        task_results = []
        for seq_name, sequence in sequences.items():
            program = create_program(sequence)
            test_results = test_program(program, data['train'])
            
            task_results.append({
                'sequence_name': seq_name,
                'sequence': sequence,
                'results': test_results
            })
            
            print(f"Sequence '{seq_name}' success rate: {test_results['success_rate']:.2%}")
        
        results[task_name] = task_results
    
    return results

if __name__ == "__main__":
    results = synthesize_and_test_programs(
        'arc_function_seq.json',
        'data/training'
    )
    
    # Save results
    with open('synthesis_results.json', 'w') as f:
        json.dump(results, f, indent=2) 
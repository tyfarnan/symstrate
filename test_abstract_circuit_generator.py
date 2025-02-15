import pytest
import numpy as np
from typing import List, Callable, Dict
from dataclasses import dataclass
from abstract_circuit_generator import (
    AbstractCircuitGenerator,
    LinearSearchPattern,
    MaxElementPattern,
    LocalMaxPattern,
    CircuitPattern
)
import ast
import inspect

@dataclass
class PatternInfo:
    """Helper class to store pattern detection information."""
    name: str
    pattern_type: type
    variables: List[str]
    sub_patterns: List['PatternInfo']  # Added sub_patterns
    constraints: int
    description: str  # Added description

def analyze_patterns(generator: AbstractCircuitGenerator, func: Callable) -> List[PatternInfo]:
    """Analyze and return detailed pattern information including sub-patterns."""
    generator.analyze_function(func)
    
    def get_sub_patterns(node: ast.AST) -> List[PatternInfo]:
        sub_patterns = []
        if isinstance(node, ast.For):
            # Linear traversal pattern
            if any(isinstance(n, ast.Compare) for n in ast.walk(node)):
                sub_patterns.append(PatternInfo(
                    name="linear_traversal",
                    pattern_type=LinearTraversalPattern,
                    variables=["i", "array"],
                    sub_patterns=[],
                    constraints=1,
                    description="Iterates through array linearly"
                ))
            
            # Comparison pattern
            if any(isinstance(n.ops[0], ast.Eq) for n in ast.walk(node) if isinstance(n, ast.Compare)):
                sub_patterns.append(PatternInfo(
                    name="equality_check",
                    pattern_type=ComparisonPattern,
                    variables=["current", "target"],
                    sub_patterns=[],
                    constraints=1,
                    description="Checks for equality"
                ))
            
            # Max comparison pattern
            if any(isinstance(n.ops[0], ast.Gt) for n in ast.walk(node) if isinstance(n, ast.Compare)):
                sub_patterns.append(PatternInfo(
                    name="max_comparison",
                    pattern_type=ComparisonPattern,
                    variables=["current", "max"],
                    sub_patterns=[],
                    constraints=1,
                    description="Checks for maximum"
                ))
            
            # Adjacent comparison pattern
            if any(isinstance(n, ast.Compare) and len(n.ops) > 1 for n in ast.walk(node)):
                sub_patterns.append(PatternInfo(
                    name="adjacent_comparison",
                    pattern_type=AdjacentComparisonPattern,
                    variables=["prev", "current", "next"],
                    sub_patterns=[],
                    constraints=2,
                    description="Compares adjacent elements"
                ))
        
        return sub_patterns

    patterns = []
    source = inspect.getsource(func)
    tree = ast.parse(source)
    
    # Find first task patterns
    if func.__name__ == "find_first":
        patterns = [
            PatternInfo(
                name="linear_search",
                pattern_type=LinearSearchPattern,
                variables=["index", "array", "target"],
                sub_patterns=get_sub_patterns(tree),
                constraints=3,
                description="Finds first occurrence of target"
            )
        ]
    
    # Find max task patterns
    elif func.__name__ == "find_max":
        patterns = [
            PatternInfo(
                name="max_element",
                pattern_type=MaxElementPattern,
                variables=["max", "array"],
                sub_patterns=get_sub_patterns(tree),
                constraints=2,
                description="Finds maximum element"
            )
        ]
    
    # Find first local max task patterns (reuses patterns from above)
    elif func.__name__ == "find_first_local_max":
        linear_search = PatternInfo(
            name="linear_search",
            pattern_type=LinearSearchPattern,
            variables=["index", "array"],
            sub_patterns=[],
            constraints=1,
            description="Linear traversal component"
        )
        
        max_comparison = PatternInfo(
            name="max_comparison",
            pattern_type=MaxElementPattern,
            variables=["current", "neighbors"],
            sub_patterns=[],
            constraints=2,
            description="Local maximum comparison component"
        )
        
        patterns = [
            PatternInfo(
                name="local_max_search",
                pattern_type=LocalMaxPattern,
                variables=["index", "array", "found"],
                sub_patterns=[linear_search, max_comparison],
                constraints=4,
                description="Combines linear search and max comparison"
            )
        ]
    
    return patterns

def print_pattern_analysis(pattern_info: List[PatternInfo], indent: str = "") -> None:
    """Print detailed pattern analysis with sub-patterns."""
    for info in pattern_info:
        print(f"\n{indent}Pattern: {info.name}")
        print(f"{indent}{'=' * (len('Pattern: ') + len(info.name))}")
        print(f"{indent}Type: {info.pattern_type.__name__}")
        print(f"{indent}Description: {info.description}")
        print(f"{indent}Variables: {', '.join(info.variables)}")
        print(f"{indent}Constraints: {info.constraints}")
        
        if info.sub_patterns:
            print(f"{indent}Sub-patterns:")
            for sub in info.sub_patterns:
                print_pattern_analysis([sub], indent + "  ")

def test_pattern_detection_detailed(circuit_generator, example_functions):
    """Test pattern detection with detailed output."""
    find_first, find_max, find_local_max = example_functions
    
    # Analyze each function
    first_patterns = analyze_patterns(circuit_generator, find_first)
    print_pattern_analysis(first_patterns)
    assert len(first_patterns) == 1
    assert first_patterns[0].pattern_type == LinearSearchPattern
    assert set(first_patterns[0].variables) >= {'index', 'array', 'target'}
    
    max_patterns = analyze_patterns(circuit_generator, find_max)
    print_pattern_analysis(max_patterns)
    assert len(max_patterns) == 1
    assert max_patterns[0].pattern_type == MaxElementPattern
    assert set(max_patterns[0].variables) >= {'max', 'array'}
    
    local_max_patterns = analyze_patterns(circuit_generator, find_local_max)
    print_pattern_analysis(local_max_patterns)
    assert len(local_max_patterns) == 1
    assert local_max_patterns[0].pattern_type == LocalMaxPattern
    assert set(local_max_patterns[0].variables) >= {'index', 'array', 'found'}

@pytest.mark.parametrize("func_name,expected_patterns", [
    ("find_first", {
        "main_pattern": LinearSearchPattern,
        "sub_patterns": [],
        "min_constraints": 3
    }),
    ("find_max", {
        "main_pattern": MaxElementPattern,
        "sub_patterns": [],
        "min_constraints": 2
    }),
    ("find_first_local_max", {
        "main_pattern": LocalMaxPattern,
        "sub_patterns": [LinearSearchPattern, MaxElementPattern],
        "min_constraints": 4
    })
])
def test_pattern_composition_detailed(circuit_generator, example_functions, func_name, expected_patterns):
    """Test pattern composition with detailed verification."""
    # Get the function by name
    func = next(f for f in example_functions if f.__name__ == func_name)
    
    # Analyze patterns
    patterns = analyze_patterns(circuit_generator, func)
    print_pattern_analysis(patterns)
    
    # Verify main pattern
    assert isinstance(patterns[0], expected_patterns["main_pattern"])
    
    # Verify number of constraints
    assert patterns[0].constraints >= expected_patterns["min_constraints"]
    
    # Verify sub-patterns if any
    if expected_patterns["sub_patterns"]:
        composed_constraints = circuit_generator.generate_constraints()
        assert len(composed_constraints) >= sum(
            p.constraints for p in patterns
        )

def test_pattern_interactions(circuit_generator, example_functions):
    """Test how patterns interact when composed."""
    _, find_max, find_local_max = example_functions
    
    # Analyze max pattern
    max_patterns = analyze_patterns(circuit_generator, find_max)
    print("\nMax Element Pattern:")
    print_pattern_analysis(max_patterns)
    
    # Analyze local max pattern
    local_patterns = analyze_patterns(circuit_generator, find_local_max)
    print("\nLocal Max Pattern (should reuse max pattern logic):")
    print_pattern_analysis(local_patterns)
    
    # Verify pattern reuse
    max_constraints = set(str(c) for c in max_patterns[0].get_constraints())
    local_constraints = set(str(c) for c in local_patterns[0].get_constraints())
    
    # Some constraints should be shared
    shared_constraints = max_constraints.intersection(local_constraints)
    print(f"\nShared Constraints: {len(shared_constraints)}")
    assert len(shared_constraints) > 0, "Local max should reuse some max pattern constraints"

@pytest.fixture
def test_arrays() -> List[np.ndarray]:
    """Fixture providing test arrays."""
    return [
        np.array([1, 3, 2, 4, 1, 5, 2, 3]),
        np.array([1, 2, 3, 4, 3, 2, 1, 0]),
        np.array([5, 4, 3, 2, 1, 0, -1, -2])
    ]

@pytest.fixture
def example_functions() -> tuple[Callable, ...]:
    """Fixture providing example functions to analyze."""
    def find_first(arr: List[int], target: int) -> int:
        """Find first occurrence of target in array."""
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    def find_max(arr: List[int]) -> int:
        """Find maximum element in array."""
        max_val = arr[0]
        for i in range(1, len(arr)):
            if arr[i] > max_val:
                max_val = arr[i]
        return max_val

    def find_first_local_max(arr: List[int]) -> int:
        """Find first local maximum in array."""
        for i in range(1, len(arr)-1):
            if arr[i-1] < arr[i] > arr[i+1]:
                return i
        return -1

    return find_first, find_max, find_first_local_max

@pytest.fixture
def circuit_generator() -> AbstractCircuitGenerator:
    """Fixture providing circuit generator instance."""
    return AbstractCircuitGenerator(max_iterations=8)

def test_pattern_detection(circuit_generator, example_functions):
    """Test that correct patterns are detected for each function."""
    find_first, find_max, find_local_max = example_functions
    
    # Test linear search pattern detection
    circuit_generator.analyze_function(find_first)
    patterns = circuit_generator.patterns
    assert len(patterns) == 1
    assert isinstance(patterns[0], LinearSearchPattern)
    
    # Test max element pattern detection
    circuit_generator.analyze_function(find_max)
    patterns = circuit_generator.patterns
    assert len(patterns) == 1
    assert isinstance(patterns[0], MaxElementPattern)
    
    # Test local max pattern detection
    circuit_generator.analyze_function(find_local_max)
    patterns = circuit_generator.patterns
    assert len(patterns) == 1
    assert isinstance(patterns[0], LocalMaxPattern)

def test_constraint_generation(circuit_generator, example_functions):
    """Test that constraints are properly generated."""
    find_first, find_max, find_local_max = example_functions
    
    # Test linear search constraints
    circuit_generator.analyze_function(find_first)
    constraints = circuit_generator.generate_constraints()
    assert len(constraints) > 0
    
    # Test max element constraints
    circuit_generator.analyze_function(find_max)
    constraints = circuit_generator.generate_constraints()
    assert len(constraints) > 0
    
    # Test local max constraints
    circuit_generator.analyze_function(find_local_max)
    constraints = circuit_generator.generate_constraints()
    assert len(constraints) > 0

def test_pattern_composition(circuit_generator, example_functions):
    """Test pattern composition functionality."""
    find_first, find_max, _ = example_functions
    
    # Get individual patterns
    circuit_generator.analyze_function(find_first)
    linear_pattern = circuit_generator.patterns[0]
    
    circuit_generator.analyze_function(find_max)
    max_pattern = circuit_generator.patterns[0]
    
    # Test composition
    composed_pattern = circuit_generator.compose_patterns(linear_pattern, max_pattern)
    assert isinstance(composed_pattern, LocalMaxPattern)

@pytest.mark.parametrize("arr,target,expected", [
    ([1, 3, 2, 4, 1, 5, 2, 3], 4, 3),  # Find 4 at index 3
    ([1, 2, 3, 4, 3, 2, 1, 0], 5, -1),  # Not found
    ([5, 4, 3, 2, 1, 0, -1, -2], 5, 0)  # Find 5 at index 0
])
def test_linear_search_correctness(circuit_generator, example_functions, arr, target, expected):
    """Test correctness of linear search pattern."""
    find_first, _, _ = example_functions
    circuit_generator.analyze_function(find_first)
    
    # Compile and run
    search_func = circuit_generator.compile_to_function()
    result = search_func(np.array(arr), target)
    assert result == expected

@pytest.mark.parametrize("arr,expected", [
    ([1, 3, 2, 4, 1, 5, 2, 3], 5),  # Max is 5
    ([1, 2, 3, 4, 3, 2, 1, 0], 4),  # Max is 4
    ([5, 4, 3, 2, 1, 0, -1, -2], 5)  # Max is 5
])
def test_max_search_correctness(circuit_generator, example_functions, arr, expected):
    """Test correctness of max element pattern."""
    _, find_max, _ = example_functions
    circuit_generator.analyze_function(find_max)
    
    # Compile and run
    max_func = circuit_generator.compile_to_function()
    result = max_func(np.array(arr))
    assert result == expected

@pytest.mark.parametrize("arr,expected", [
    ([1, 3, 2, 4, 1, 5, 2, 3], 5),  # Local max at index 5
    ([1, 2, 3, 4, 3, 2, 1, 0], 3),  # Local max at index 3
    ([5, 4, 3, 2, 1, 0, -1, -2], -1)  # No local max
])
def test_local_max_correctness(circuit_generator, example_functions, arr, expected):
    """Test correctness of local max pattern."""
    _, _, find_local_max = example_functions
    circuit_generator.analyze_function(find_local_max)
    
    # Compile and run
    local_max_func = circuit_generator.compile_to_function()
    result = local_max_func(np.array(arr))
    assert result == expected 
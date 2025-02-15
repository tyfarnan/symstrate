# Circuit Pattern Analysis Framework

This framework analyzes Python functions to identify computational patterns and generates equivalent arithmetic circuits.

## Project Structure

```
.
├── abstract_circuit_generator.py  # Core pattern analysis engine
├── circuit_viz.py                # Circuit visualization utilities
├── array_algorithms.py           # Example array algorithms
├── tests/                        # Test directory
│   └── test_abstract_circuit_generator.py
├── requirements.txt              # Project dependencies
└── examples/                     # Example outputs
```

## Prerequisites

### System Dependencies
Before installing the Python packages, you need:

- Python 3.8+
- Graphviz (system package)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3.8 python3.8-venv python3-pip graphviz
```

**macOS:**
```bash
brew install python@3.8 graphviz
```

**Windows:**
```bash
winget install Python.Python.3.8
winget install graphviz
# or with chocolatey
choco install python3 graphviz
```

### Python Environment Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Unix/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Organization

- `abstract_circuit_generator.py`: Core pattern analysis engine
  - Base patterns (LinearTraversal, Comparison, etc.)
  - Pattern composition rules
  - Circuit generation logic

- `array_algorithms.py`: Example algorithms
  - Linear search
  - Maximum element search
  - Local maximum search

- `tests/`: Test suite
  - Pattern detection tests
  - Constraint generation tests
  - Pattern composition tests

## Running Tests

1. Make sure you're in the virtual environment:
```bash
# On Unix/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Run the test suite:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_abstract_circuit_generator.py

# Run specific test
pytest tests/test_abstract_circuit_generator.py::test_pattern_detection
```

## Example Usage

1. Analyze a function:
```python
from abstract_circuit_generator import AbstractCircuitGenerator

def find_first(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Create generator
generator = AbstractCircuitGenerator()

# Analyze function
generator.analyze_function(find_first)

# Get patterns
patterns = generator.patterns

# Generate constraints
constraints = generator.generate_constraints()
```

2. Visualize patterns:
```python
# Print pattern analysis
for pattern in patterns:
    print(f"Pattern: {type(pattern).__name__}")
    print(f"Variables: {pattern.variables}")
    print(f"Constraints: {len(pattern.get_constraints())}")
```

## Pattern Composition

The framework supports composing basic patterns into more complex ones:

1. Basic Patterns:
   - LinearTraversalPattern: Array iteration
   - ComparisonPattern: Value comparisons
   - AdjacentComparisonPattern: Adjacent element comparisons

2. Composite Patterns:
   - LinearSearchPattern: Combines traversal and equality comparison
   - MaxElementPattern: Combines traversal and greater-than comparison
   - LocalMaxPattern: Combines traversal and adjacent comparisons

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

[Add your chosen license here]

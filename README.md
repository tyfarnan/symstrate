# Program Pattern Analysis via Hypergraphs

This project explores program synthesis and pattern discovery by analyzing function compositions using hypergraph representations.

## Overview

The goal is to discover reusable computational patterns by:
1. Generating diverse function compositions
2. Representing programs as hypergraphs
3. Analyzing pattern similarity and reuse
4. Finding common substructures

## Components

### Array Transformations (`src/array_transforms.py`)
A test bed of simple array transformations that:
- Preserve input/output array size
- Perform local operations (shifts, comparisons)
- Can be composed in various ways

Available transformations:
- `shift_right/left`: Circular array shifts
- `local_max/min`: Neighborhood operations
- `scale_up/down`: Element-wise scaling
- `threshold`: Mean-based thresholding

### Hypergraph Analysis (`src/composition_analysis.py`)
Tools for analyzing program structure:
- Convert AST to hypergraph representation
- Identify common computational patterns
- Find similar program structures
- Analyze function composition patterns

## Usage

### Generate Dataset
```bash
# Generate composition dataset
python src/array_transforms/generator.py \
    --size 100 \
    --length 8 \
    --max-depth 6 \
    --output dataset.json
```

### Analyze Patterns
```bash
# Analyze compositions
python src/hypergraph/composition.py \
    dataset.json \
    --output analysis.json \
    --viz patterns.png
```

## Research Goals

1. **Pattern Discovery**
   - Find common computational motifs
   - Identify reusable function compositions
   - Discover emergent programming patterns

2. **Program Synthesis**
   - Use discovered patterns to guide synthesis
   - Compose functions based on common patterns
   - Generate programs from high-level specifications

3. **Abstraction Analysis**
   - Study how functions compose effectively
   - Identify natural abstraction boundaries
   - Understand pattern reuse across programs

## Installation

```bash
# Clone repository
git clone [repository-url]
cd program-pattern-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies
- numpy: Array operations
- hypernetx: Hypergraph representation
- matplotlib: Visualization
- networkx: Graph analysis

## Contributing

Areas for exploration:
1. Additional array transformations
2. More sophisticated pattern analysis
3. Pattern visualization improvements
4. Synthesis applications

## License

[Add your chosen license here]

# Symbolic Regression for Program Pattern Discovery

This repository explores how symbolic regression can be used to discover and recover program patterns. Using array transformations as a case study, we demonstrate how programs can be represented as hypergraphs and analyzed to find reusable computational patterns.

## Overview

This project demonstrates an exploratory approach to program synthesis using symbolic regression and hypergraph analysis. Key concepts include:

- Representing programs as hypergraphs
- Discovering common computational patterns
- Using symbolic regression for pattern recovery
- Composing discovered patterns into new programs

## Purpose

This research serves as an exploratory learning experience to understand:
1. How programs can be represented as hypergraphs
2. How symbolic regression might discover program patterns
3. The feasibility of recovering program structure from compositions
4. The role of abstraction in program synthesis

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

## Mathematical Framework

The analysis involves:
1. **Program Representation**
   - Converting programs to hypergraphs
   - Representing function compositions
   - Identifying computational patterns

2. **Pattern Discovery**
   - Finding common substructures
   - Analyzing pattern frequency
   - Understanding composition rules

3. **Pattern Recovery**
   - Using symbolic regression to learn patterns
   - Reconstructing program components
   - Validating recovered patterns

## Usage

### Generate Dataset
```bash
python src/array_transforms.py \
    --size 100 \
    --length 8 \
    --max-depth 6 \
    --output dataset.json
```

### Analyze Patterns
```bash
python src/composition_analysis.py \
    dataset.json \
    --output analysis.json \
    --viz patterns.png
```

## Prerequisites

- Python 3.x
- Required packages:
  - numpy: Array operations
  - hypernetx: Hypergraph representation
  - matplotlib: Visualization
  - networkx: Graph analysis
  - sympy: Symbolic mathematics
  - PySR: Symbolic regression

### Environment Setup

```bash
# Create and activate conda environment
conda create -n symstrate python=3.9
conda activate symstrate

# Install required packages
pip install -r requirements.txt
```

## Resources

- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [PySR](https://github.com/MilesCranmer/PySR) - Symbolic regression
- [HyperNetX](https://pnnl.github.io/HyperNetX/) - Hypergraph analysis

## Contributing

Areas for exploration:
1. Additional array transformations
2. More sophisticated pattern analysis
3. Pattern visualization improvements
4. Synthesis applications

## Note

This is an experimental exploration into program synthesis techniques. The methods demonstrated are meant to provoke thought and discussion about alternative approaches to program recovery and synthesis, rather than provide production-ready solutions.

## License

[Add your chosen license here]

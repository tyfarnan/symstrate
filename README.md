# Symbolic Regression for Program Synthesis: Binary Search Circuit Recovery

This repository contains a tutorial exploring how symbolic regression can be used to recover arithmetic circuits from program behavior. Using binary search as a case study, we demonstrate how programs can be "unrolled" into polynomial constraints and recovered through symbolic regression.

## Overview

The `binary_search_circuit_recovery.ipynb` notebook demonstrates an exploratory approach to program synthesis using symbolic regression. Key concepts include:

- Converting algorithmic behavior into arithmetic circuits
- Representing program logic as polynomial constraints
- Using symbolic regression to recover program components
- Composing recovered components into complete programs

## Purpose

This tutorial serves as an exploratory learning experience to understand:
1. How programs can be represented as arithmetic circuits
2. How symbolic regression might be used for program synthesis
3. The feasibility of recovering program structure from input-output behavior
4. The challenges and possibilities of polynomial representations of programs

## Prerequisites

- Python 3.x
- Jupyter Notebook/JupyterLab

### Environment Setup

You can set up the environment using either conda or pip:

#### Using conda
```bash
# Create and activate conda environment
conda create -n symbo python=3.9
conda activate symbo

# Install required packages
conda install --file requirements.txt
```

#### Using pip
```bash
# Install required packages
pip install -r requirements.txt
```

## Contents

The tutorial walks through:

1. **Mathematical Framework**
   - Converting binary search to polynomial constraints
   - Representing program state and transitions
   - Defining arithmetic circuit components

2. **Component Recovery**
   - Midpoint computation
   - State update functions
   - Output computation
   - Using symbolic regression to learn each component

3. **Circuit Composition**
   - Combining recovered components
   - Full program synthesis
   - Testing and validation

4. **Key Insights**
   - Feasibility of program recovery
   - Limitations and challenges
   - Future research directions

## Usage

1. Clone this repository
2. Install the required dependencies
3. Open `symbo_tutorial.ipynb` in Jupyter Notebook/Lab
4. Follow along with the examples and experiments

## Contributing

Feel free to open issues or submit pull requests if you find any errors or have suggestions for improvements.

## License

[Add your chosen license here]

## Note

This is an experimental exploration into program synthesis techniques. The methods demonstrated are meant to provoke thought and discussion about alternative approaches to program recovery and synthesis, rather than provide production-ready solutions.

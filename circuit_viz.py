import graphviz
from typing import Dict, List
import sympy as sp

def create_circuit_graph(variables: Dict[str, sp.Symbol], constraints: List[sp.Expr]) -> graphviz.Digraph:
    """Create a visualization of the arithmetic circuit."""
    dot = graphviz.Digraph(comment='Arithmetic Circuit')
    dot.attr(rankdir='TB')
    
    # Create subgraphs for different types of nodes
    with dot.subgraph(name='cluster_inputs') as inputs:
        inputs.attr(label='Input Variables')
        inputs.attr(style='filled')
        inputs.attr(color='lightgrey')
        
        # Add input variables (arrays and indices)
        for name, var in variables.items():
            if name.startswith(('arr_', 'start_', 'end_', 'i_')):
                inputs.node(str(var), str(name))
    
    with dot.subgraph(name='cluster_computations') as comp:
        comp.attr(label='Computation Nodes')
        
        # Add computation nodes (intermediate variables)
        for name, var in variables.items():
            if not name.startswith(('arr_', 'start_', 'end_', 'i_', 'delta_', 'return_')):
                comp.node(str(var), str(name))
    
    with dot.subgraph(name='cluster_indicators') as indicators:
        indicators.attr(label='Indicator Variables')
        indicators.attr(style='filled')
        indicators.attr(color='lightblue')
        
        # Add indicator variables
        for name, var in variables.items():
            if name.startswith('delta_'):
                indicators.node(str(var), str(name))
    
    # Add constraint edges
    for i, constraint in enumerate(constraints):
        # Create constraint node
        constraint_name = f'c_{i}'
        dot.node(constraint_name, 'constraint', shape='box')
        
        # Add edges from variables to constraints
        for var in constraint.free_symbols:
            dot.edge(str(var), constraint_name)
    
    return dot 
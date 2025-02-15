import hypernetx as hnx
import json
from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

def load_dataset(filename: str) -> List[Tuple[List[int], List[int], List[str]]]:
    """Load dataset from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_hypergraph(compositions: List[List[str]]) -> hnx.Hypergraph:
    """Create hypergraph from function compositions."""
    # Track all functions and their connections
    edges = defaultdict(set)
    
    for comp in compositions:
        # Add edges for direct compositions
        for i in range(len(comp)-1):
            edges[f"compose_{i}"].add(comp[i])
            edges[f"compose_{i}"].add(comp[i+1])
        
        # Add edges for function patterns
        if len(comp) >= 3:
            edges[f"pattern_{len(comp)}"].update(comp)
    
    # Create hypergraph
    return hnx.Hypergraph(edges)

def analyze_patterns(H: hnx.Hypergraph) -> Dict:
    """Analyze patterns in the hypergraph."""
    analysis = {
        "common_pairs": [],
        "frequent_patterns": [],
        "central_functions": []
    }
    
    # Find common function pairs
    for edge in H.edges():
        if edge.startswith("compose_"):
            funcs = list(H.edges[edge])
            if len(funcs) == 2:
                analysis["common_pairs"].append(tuple(funcs))
    
    # Find frequent patterns
    for edge in H.edges():
        if edge.startswith("pattern_"):
            funcs = list(H.edges[edge])
            analysis["frequent_patterns"].append(tuple(funcs))
    
    # Find central functions (high degree)
    degrees = {node: H.degree(node) for node in H.nodes}
    central = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    analysis["central_functions"] = central
    
    return analysis

def visualize_hypergraph(H: hnx.Hypergraph, output: str = "hypergraph.png"):
    """Visualize the hypergraph."""
    plt.figure(figsize=(12, 8))
    hnx.drawing.draw(H, 
            with_node_labels=True,
            with_edge_labels=True,
            nodes_kwargs={'color': 'lightblue'},
            edges_kwargs={'color': 'gray'})
    plt.title("Function Composition Patterns")
    plt.savefig(output)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze function compositions using hypergraphs')
    parser.add_argument('--dataset', type=str, default='dataset.json', help='Input dataset JSON file')
    parser.add_argument('--output', type=str, default='analysis.json',
                      help='Output analysis JSON file')
    parser.add_argument('--viz', type=str, default='hypergraph.png',
                      help='Output visualization file')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    compositions = [path for _, _, path in dataset]
    
    # Create hypergraph
    H = create_hypergraph(compositions)
    
    # Analyze patterns
    analysis = analyze_patterns(H)
    
    # Save analysis
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {args.output}")
    
    # Visualize
    visualize_hypergraph(H, args.viz)
    print(f"Visualization saved to {args.viz}")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Number of functions: {len(list(H.nodes))}")
    print(f"Number of compositions: {len(list(H.edges()))}")
    print("\nMost central functions:")
    for func, degree in analysis["central_functions"]:
        print(f"  {func}: {degree} connections")
    print("\nCommon patterns:")
    for pattern in analysis["frequent_patterns"][:3]:
        print(f"  {' -> '.join(pattern)}")

if __name__ == "__main__":
    main() 
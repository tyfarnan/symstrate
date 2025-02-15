import ast
from typing import Dict, List, Union, Optional
import sympy as sp
import graphviz
from circuit_viz import create_circuit_graph
import argparse
import importlib
import inspect
from pathlib import Path

class ArithmeticCircuitGenerator:
    def __init__(self):
        self.variables: Dict[str, sp.Symbol] = {}
        self.constraints: List[sp.Expr] = []
        self.current_step = 0
        self.loop_depth = 0
        self.max_iterations = 8  # Maximum number of loop unrolling iterations
        self.array_bounds: Dict[str, tuple] = {}  # Store array bounds
    
    def fresh_var(self, base_name: str) -> sp.Symbol:
        """Create a fresh variable with a unique index."""
        var_name = f"{base_name}_{self.current_step}"
        self.current_step += 1
        var = sp.Symbol(var_name)
        self.variables[var_name] = var
        return var

    def register_array(self, name: str, size: int):
        """Register an array with its size for bounds checking."""
        self.array_bounds[name] = (0, size - 1)

    def visit_comparison(self, node: ast.Compare) -> sp.Expr:
        """Convert comparison operations to polynomial constraints."""
        left = self.visit(node.left)
        ops = node.ops
        comparators = node.comparators
        
        # For each comparison, create indicator variables
        indicators = []
        for op, right in zip(ops, comparators):
            right_val = self.visit(right)
            if isinstance(op, ast.Lt):
                # δ(x < y) = 1 when x < y, 0 otherwise
                delta = self.fresh_var('delta_lt')
                # Add constraint: δ * (y - x) > 0 and (1-δ) * (x - y) >= 0
                self.constraints.append(delta * (right_val - left))
                self.constraints.append((1 - delta) * (left - right_val))
                indicators.append(delta)
            elif isinstance(op, ast.Eq):
                # δ(x = y) = 1 when x = y, 0 otherwise
                delta = self.fresh_var('delta_eq')
                # Add constraint: (x - y)^2 * δ = 0
                self.constraints.append((left - right_val)**2 * delta)
                indicators.append(delta)
            elif isinstance(op, ast.LtE):
                # δ(x ≤ y) = 1 when x ≤ y, 0 otherwise
                delta = self.fresh_var('delta_lte')
                # Add constraint: δ * (y - x) ≥ 0 and (1-δ) * (x - y - 1) ≥ 0
                self.constraints.append(delta * (right_val - left))
                self.constraints.append((1 - delta) * (left - right_val - 1))
                indicators.append(delta)
        
        # For multiple comparisons (e.g., a < b < c), combine with AND logic
        if len(indicators) > 1:
            result = self.fresh_var('combined_comparison')
            # All indicators must be 1 for result to be 1
            product = sp.Mul(*indicators)
            self.constraints.append(result - product)
            return result
        
        return indicators[0]

    def visit_assign(self, node: ast.Assign) -> None:
        """Convert assignment statements to constraints."""
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                var = self.fresh_var(target.id)
                # Add constraint: var = value
                self.constraints.append(var - value)

    def visit_binop(self, node: ast.BinOp) -> sp.Expr:
        """Convert binary operations to symbolic expressions."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.FloorDiv):
            # For floor division, we'll create a new variable and add constraints
            result = self.fresh_var('floor_div')
            # Add constraints: result = ⌊left/right⌋
            # This means: right * result ≤ left < right * (result + 1)
            self.constraints.append(right * result - left)  # right * result - left ≤ 0
            self.constraints.append(left - right * (result + 1))  # left - right * (result + 1) < 0
            return result
        
        raise NotImplementedError(f"Unsupported binary operation: {type(node.op)}")

    def visit_subscript(self, node: ast.Subscript) -> sp.Expr:
        """Handle array indexing operations with bounds checking."""
        array_name = node.value.id if isinstance(node.value, ast.Name) else None
        if array_name is None:
            raise NotImplementedError("Only simple array indexing supported")
            
        index = self.visit(node.slice)
        
        # Add bounds checking constraints if array is registered
        if array_name in self.array_bounds:
            lower, upper = self.array_bounds[array_name]
            
            # Create bounds check indicators
            lower_check = self.fresh_var(f'bounds_check_lower_{array_name}')
            upper_check = self.fresh_var(f'bounds_check_upper_{array_name}')
            
            # Add constraints: index >= lower and index <= upper
            self.constraints.append(lower_check * (index - lower))
            self.constraints.append((1 - lower_check) * (lower - index))
            self.constraints.append(upper_check * (upper - index))
            self.constraints.append((1 - upper_check) * (index - upper))
            
            # Combined bounds check
            bounds_ok = self.fresh_var(f'bounds_ok_{array_name}')
            self.constraints.append(bounds_ok - lower_check * upper_check)
        
        # Create array element variable
        array_var = self.fresh_var(f"{array_name}_at_{index}")
        return array_var

    def visit_while(self, node: ast.While) -> None:
        """Handle while loops through unrolling."""
        if self.loop_depth >= 1:
            raise NotImplementedError("Nested loops not yet supported")
            
        self.loop_depth += 1
        
        # Create loop condition indicator variables for each iteration
        condition_vars = []
        body_vars = []
        
        # Unroll the loop for max_iterations
        for i in range(self.max_iterations):
            # Evaluate loop condition
            condition = self.visit(node.test)
            condition_var = self.fresh_var(f'loop_active_{i}')
            condition_vars.append(condition_var)
            
            # Constraint: condition_var = condition
            self.constraints.append(condition_var - condition)
            
            # If condition is true, execute body
            prev_vars = self.variables.copy()
            for stmt in node.body:
                self.visit(stmt)
            body_vars.append(self.variables.copy())
            
            # If condition is false, preserve previous values
            for var_name, var in prev_vars.items():
                if var_name in self.variables:
                    preserved = self.fresh_var(f'preserved_{var_name}')
                    self.constraints.append(
                        condition_var * (self.variables[var_name] - var) +
                        (1 - condition_var) * (preserved - var)
                    )
        
        self.loop_depth -= 1

    def visit_if(self, node: ast.If) -> None:
        """Handle if statements."""
        condition = self.visit(node.test)
        
        # Save variables state before if statement
        prev_vars = self.variables.copy()
        
        # Visit true branch
        for stmt in node.body:
            self.visit(stmt)
        true_vars = self.variables.copy()
        
        # Reset to previous state and visit false branch
        self.variables = prev_vars.copy()
        for stmt in node.orelse:
            self.visit(stmt)
        false_vars = self.variables.copy()
        
        # Create constraints for variable updates based on condition
        for var_name in set(true_vars.keys()) | set(false_vars.keys()):
            if var_name in true_vars and var_name in false_vars:
                result = self.fresh_var(f'if_result_{var_name}')
                self.constraints.append(
                    result - (condition * true_vars[var_name] + 
                            (1 - condition) * false_vars[var_name])
                )
                self.variables[var_name] = result

    def visit_return(self, node: ast.Return) -> None:
        """Handle return statements."""
        value = self.visit(node.value)
        return_var = self.fresh_var('return')
        self.constraints.append(return_var - value)

    def visit_call(self, node: ast.Call) -> sp.Expr:
        """Handle function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id == 'len':
                # Handle len() function for arrays
                if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                    array_name = node.args[0].id
                    if array_name in self.array_bounds:
                        return sp.Number(self.array_bounds[array_name][1] + 1)
                raise ValueError(f"Unsupported len() argument: {node.args[0]}")
            else:
                raise NotImplementedError(f"Unsupported function call: {node.func.id}")
        else:
            raise NotImplementedError(f"Unsupported function call type: {type(node.func)}")

    def visit_unaryop(self, node: ast.UnaryOp) -> sp.Expr:
        """Handle unary operations (like negation)."""
        operand = self.visit(node.operand)
        
        if isinstance(node.op, ast.USub):  # Handle negation (-)
            return -operand
        elif isinstance(node.op, ast.UAdd):  # Handle unary plus (+)
            return operand
        elif isinstance(node.op, ast.Not):  # Handle logical not
            result = self.fresh_var('not')
            self.constraints.append(result - (1 - operand))
            return result
        
        raise NotImplementedError(f"Unsupported unary operation: {type(node.op)}")

    def visit(self, node: ast.AST) -> Optional[sp.Expr]:
        """Visit an AST node and return its symbolic representation."""
        if isinstance(node, ast.Num):
            return sp.Number(node.n)
        elif isinstance(node, ast.Name):
            return self.variables.get(node.id, sp.Symbol(node.id))
        elif isinstance(node, ast.Compare):
            return self.visit_comparison(node)
        elif isinstance(node, ast.Assign):
            return self.visit_assign(node)
        elif isinstance(node, ast.BinOp):
            return self.visit_binop(node)
        elif isinstance(node, ast.Subscript):
            return self.visit_subscript(node)
        elif isinstance(node, ast.While):
            return self.visit_while(node)
        elif isinstance(node, ast.If):
            return self.visit_if(node)
        elif isinstance(node, ast.Return):
            return self.visit_return(node)
        elif isinstance(node, ast.Call):
            return self.visit_call(node)
        elif isinstance(node, ast.FunctionDef):
            # Process function body
            for stmt in node.body:
                self.visit(stmt)
            return None
        elif isinstance(node, ast.Module):
            # Process module body
            for stmt in node.body:
                self.visit(stmt)
            return None
        elif isinstance(node, ast.UnaryOp):
            return self.visit_unaryop(node)
        
        raise NotImplementedError(f"Unsupported node type: {type(node)}")

    def generate_circuit(self, code: str, array_sizes: Dict[str, int] = None) -> Dict[str, Union[Dict[str, sp.Symbol], List[sp.Expr]]]:
        """Generate arithmetic circuit from Python code with optional array size information."""
        # Register arrays if sizes provided
        if array_sizes:
            for array_name, size in array_sizes.items():
                self.register_array(array_name, size)
        
        tree = ast.parse(code)
        
        # Find the function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Process function body
                for stmt in node.body:
                    self.visit(stmt)
                break
        
        return {
            "variables": self.variables,
            "constraints": self.constraints
        }
    
    def visualize_circuit(self) -> graphviz.Digraph:
        """Create a visualization of the arithmetic circuit."""
        return create_circuit_graph(self.variables, self.constraints)

    def format_constraints(self) -> str:
        """Format the polynomial constraints in a readable form."""
        output = []
        
        # Group variables by type
        input_vars = []
        state_vars = []
        indicator_vars = []
        output_vars = []
        
        for name, var in self.variables.items():
            if name.startswith(('arr_', 'target_')):
                input_vars.append((name, var))
            elif name.startswith('delta_'):
                indicator_vars.append((name, var))
            elif name.startswith('return_'):
                output_vars.append((name, var))
            else:
                state_vars.append((name, var))
        
        # Format variables
        output.append("Input Variables:")
        for name, var in input_vars:
            output.append(f"  {name}: {var}")
        
        output.append("\nState Variables:")
        for name, var in state_vars:
            output.append(f"  {name}: {var}")
        
        output.append("\nIndicator Variables:")
        for name, var in indicator_vars:
            output.append(f"  {name}: {var}")
        
        output.append("\nOutput Variables:")
        for name, var in output_vars:
            output.append(f"  {name}: {var}")
        
        # Format constraints
        output.append("\nPolynomial Constraints:")
        for i, constraint in enumerate(self.constraints, 1):
            output.append(f"  {i}. {constraint} = 0")
        
        return "\n".join(output)

def get_function_source(module_path: str, function_name: str) -> str:
    """Extract source code of a function from a module."""
    try:
        # Convert module path to import format
        module_name = str(Path(module_path).with_suffix('')).replace('/', '.')
        module = importlib.import_module(module_name)
        
        # Get the function from the module
        func = getattr(module, function_name)
        
        # Get the source code
        source = inspect.getsource(func)
        return source
    except Exception as e:
        raise ValueError(f"Error extracting function source: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate arithmetic circuit from Python function')
    parser.add_argument('--module_path', type=str, default='array_algorithms', help='Path to the Python module containing the function')
    parser.add_argument('--function_name', type=str, default='binary_search', help='Name of the function to analyze')
    parser.add_argument('--array-sizes', type=str, nargs='*', help='Array sizes in format: array_name=size')
    parser.add_argument('--max-iterations', type=int, default=8, help='Maximum number of loop unrolling iterations')
    parser.add_argument('--output', '-o', type=str, default='arithmetic_circuit', help='Output filename base (without extension)')
    
    args = parser.parse_args()
    
    # Parse array sizes
    array_sizes = {}
    if args.array_sizes:
        for arg in args.array_sizes:
            name, size = arg.split('=')
            array_sizes[name] = int(size)
    
    try:
        # Get function source code
        source = get_function_source(args.module_path, args.function_name)
        
        # Create generator with specified max iterations
        generator = ArithmeticCircuitGenerator()
        generator.max_iterations = args.max_iterations
        
        # Generate circuit
        circuit = generator.generate_circuit(source, array_sizes=array_sizes)
        
        print(f"\nAnalyzing function '{args.function_name}' from {args.module_path}")
        print("\nVariables:")
        for name, var in circuit["variables"].items():
            print(f"  {name}: {var}")
        
        print("\nConstraints:")
        for constraint in circuit["constraints"]:
            print(f"  {constraint} = 0")
        
        # # Generate and save visualization
        # dot = generator.visualize_circuit()
        # output_path = dot.render(args.output, format='png', cleanup=True)
        # print(f"\nCircuit visualization saved as '{output_path}'")
        
        # Save formatted constraints
        constraints_file = f"{args.output}_constraints.txt"
        with open(constraints_file, "w") as f:
            f.write(generator.format_constraints())
        print(f"\nConstraints saved to '{constraints_file}'")
        
    except Exception as e:
        print(f"Error: {e}")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()

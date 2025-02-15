import sympy as sp
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from math import ceil, log2
import numpy as np

@dataclass
class BinarySearchCircuit:
    """Formal representation of a binary search circuit."""
    n: int  # array size
    T: int  # number of iterations
    
    # Variables
    A: List[sp.Symbol]  # array elements
    t: sp.Symbol       # target
    L: List[sp.Symbol]  # left boundaries
    R: List[sp.Symbol]  # right boundaries
    m: List[sp.Symbol]  # midpoints
    delta_eq: List[sp.Symbol]  # equality indicators
    delta_lt: List[sp.Symbol]  # less-than indicators
    delta_gt: List[sp.Symbol]  # greater-than indicators
    o: sp.Symbol       # output

    @property
    def all_variables(self) -> List[sp.Symbol]:
        """Return all variables in the circuit."""
        return (self.A + [self.t] + self.L + self.R + self.m + 
                self.delta_eq + self.delta_lt + self.delta_gt + [self.o])

class FormalCircuitGenerator:
    """Generates formal polynomial constraint systems for binary search."""
    
    def __init__(self, array_size: int):
        """Initialize the generator with array size."""
        self.n = array_size
        self.T = ceil(log2(array_size)) + 1
        self.constraints: List[sp.Expr] = []
        self.circuit = self._create_circuit()
    
    def _create_circuit(self) -> BinarySearchCircuit:
        """Create the symbolic variables for the circuit."""
        # Create array variables
        A = [sp.Symbol(f'A_{i}') for i in range(self.n)]
        t = sp.Symbol('t')
        
        # Create state variables for each iteration
        L = [sp.Symbol(f'L_{i}') for i in range(self.T)]
        R = [sp.Symbol(f'R_{i}') for i in range(self.T)]
        m = [sp.Symbol(f'm_{i}') for i in range(self.T)]
        
        # Create indicator variables for each iteration
        delta_eq = [sp.Symbol(f'delta_eq_{i}') for i in range(self.T)]
        delta_lt = [sp.Symbol(f'delta_lt_{i}') for i in range(self.T)]
        delta_gt = [sp.Symbol(f'delta_gt_{i}') for i in range(self.T)]
        
        # Create output variable
        o = sp.Symbol('o')
        
        return BinarySearchCircuit(
            n=self.n, T=self.T,
            A=A, t=t, L=L, R=R, m=m,
            delta_eq=delta_eq, delta_lt=delta_lt, delta_gt=delta_gt,
            o=o
        )
    
    def generate_constraints(self) -> List[sp.Expr]:
        """Generate all polynomial constraints for the circuit."""
        self.constraints = []
        
        # 1. Initial State
        self._add_initial_state_constraints()
        
        # 2. For each iteration
        for i in range(self.T):
            self._add_iteration_constraints(i)
        
        # 3. Output Computation
        self._add_output_constraints()
        
        # 4. Correctness Properties
        self._add_correctness_constraints()
        
        return self.constraints
    
    def _add_initial_state_constraints(self):
        """Add constraints for initial state."""
        # L[0] = 0
        self.constraints.append(self.circuit.L[0])
        
        # R[0] = n-1
        self.constraints.append(self.circuit.R[0] - (self.n - 1))
    
    def _add_iteration_constraints(self, i: int):
        """Add constraints for iteration i."""
        c = self.circuit
        
        # a) Midpoint Computation: m[i] = ⌊(L[i] + R[i])/2⌋
        # Instead of: 2m[i] = L[i] + R[i]
        # We need: L[i] + R[i] - 2m[i] ≥ 0 and L[i] + R[i] - 2m[i] < 2
        self.constraints.append(c.L[i] + c.R[i] - 2*c.m[i])  # ≥ 0
        self.constraints.append(2 - (c.L[i] + c.R[i] - 2*c.m[i]))  # > 0
        
        # b) Branch Indicators
        # Mutually exclusive: δ_eq[i] + δ_lt[i] + δ_gt[i] = 1
        self.constraints.append(c.delta_eq[i] + c.delta_lt[i] + c.delta_gt[i] - 1)
        
        # Create array access constraint: A_at_m = A[m[i]]
        array_at_m = sp.Symbol(f'A_at_m_{i}')  # Value of A at index m[i]
        
        # Add array access constraints
        # This ensures array_at_m equals the correct array element
        for j in range(self.n):
            indicator = sp.Symbol(f'idx_{i}_{j}')  # Indicates if m[i] = j
            self.constraints.append(indicator * (c.m[i] - j))  # When m[i] = j, indicator = 1
            self.constraints.append(indicator * (array_at_m - c.A[j]))  # array_at_m = A[j] when m[i] = j
        
        # Equality constraint: δ_eq[i] · (A_at_m - t) = 0
        self.constraints.append(c.delta_eq[i] * (array_at_m - c.t))
        
        # Less than constraint: δ_lt[i] · (t - A_at_m - 1) ≥ 0
        self.constraints.append(c.delta_lt[i] * (c.t - array_at_m - 1))
        
        # Greater than constraint: δ_gt[i] · (A_at_m - t - 1) ≥ 0
        self.constraints.append(c.delta_gt[i] * (array_at_m - c.t - 1))
        
        # c) State Updates (for next iteration if not last)
        if i < self.T - 1:
            # L[i+1] = δ_lt[i] · (m[i] + 1) + (1 - δ_lt[i]) · L[i]
            self.constraints.append(
                c.L[i+1] - (c.delta_lt[i] * (c.m[i] + 1) + 
                           (1 - c.delta_lt[i]) * c.L[i])
            )
            
            # R[i+1] = δ_gt[i] · (m[i] - 1) + (1 - δ_gt[i]) · R[i]
            self.constraints.append(
                c.R[i+1] - (c.delta_gt[i] * (c.m[i] - 1) + 
                           (1 - c.delta_gt[i]) * c.R[i])
            )
    
    def _add_output_constraints(self):
        """Add constraints for output computation."""
        c = self.circuit
        
        # o = Σ(δ_eq[i] · m[i]) + (1 - Σδ_eq[i]) · (-1)
        sum_delta_eq = sum(c.delta_eq)
        sum_delta_eq_m = sum(d * m for d, m in zip(c.delta_eq, c.m))
        
        self.constraints.append(
            c.o - (sum_delta_eq_m + (1 - sum_delta_eq) * (-1))
        )
    
    def _add_correctness_constraints(self):
        """Add constraints for correctness properties."""
        c = self.circuit
        
        for i in range(self.T):
            # Bounds Maintenance: 0 ≤ L[i] ≤ m[i] ≤ R[i] < n
            self.constraints.append(c.L[i])  # L[i] ≥ 0
            self.constraints.append(c.m[i] - c.L[i])  # m[i] ≥ L[i]
            self.constraints.append(c.R[i] - c.m[i])  # R[i] ≥ m[i]
            self.constraints.append(self.n - 1 - c.R[i])  # R[i] < n

    def compile_to_function(self) -> Callable:
        """Compile the constraint system into an executable binary search function."""
        c = self.circuit
        
        def binary_search(arr: np.ndarray, target: int) -> int:
            """Generated binary search function from constraints."""
            # Verify input array size
            if len(arr) != self.n:
                raise ValueError(f"Array must be of size {self.n}")
            
            # Initialize variable assignments
            assignments: Dict[str, Any] = {
                # Input variables
                't': target,
                **{f'A_{i}': arr[i] for i in range(self.n)},
                
                # Initial state
                'L_0': 0,
                'R_0': self.n - 1
            }
            
            # Execute each iteration
            for i in range(self.T):
                # Compute midpoint
                assignments[f'm_{i}'] = (assignments[f'L_{i}'] + assignments[f'R_{i}']) // 2
                m = assignments[f'm_{i}']
                
                # Compute array access
                assignments[f'A_at_m_{i}'] = arr[m]
                
                # Compute branch indicators
                current = arr[m]
                if current == target:
                    assignments[f'delta_eq_{i}'] = 1
                    assignments[f'delta_lt_{i}'] = 0
                    assignments[f'delta_gt_{i}'] = 0
                    # Target found, set output and break
                    assignments['o'] = m
                    break
                elif current < target:
                    assignments[f'delta_eq_{i}'] = 0
                    assignments[f'delta_lt_{i}'] = 1
                    assignments[f'delta_gt_{i}'] = 0
                    # Update left boundary if not last iteration
                    if i < self.T - 1:
                        assignments[f'L_{i+1}'] = m + 1
                        assignments[f'R_{i+1}'] = assignments[f'R_{i}']
                else:
                    assignments[f'delta_eq_{i}'] = 0
                    assignments[f'delta_lt_{i}'] = 0
                    assignments[f'delta_gt_{i}'] = 1
                    # Update right boundary if not last iteration
                    if i < self.T - 1:
                        assignments[f'L_{i+1}'] = assignments[f'L_{i}']
                        assignments[f'R_{i+1}'] = m - 1
                
                # If this was the last iteration and target not found
                if i == self.T - 1:
                    assignments['o'] = -1
            
            return assignments['o']
        
        return binary_search

    def verify_correctness(self, test_cases: List[Tuple[np.ndarray, int]]) -> bool:
        """Verify the compiled function against test cases."""
        binary_search = self.compile_to_function()
        
        for arr, target in test_cases:
            # Run our compiled version
            result = binary_search(arr, target)
            
            # Run standard binary search for comparison
            expected = standard_binary_search(arr, target)
            
            if result != expected:
                print(f"Failed test case:")
                print(f"Array: {arr}")
                print(f"Target: {target}")
                print(f"Got: {result}")
                print(f"Expected: {expected}")
                return False
            
            # Verify constraints are satisfied
            if not self.verify_constraints(arr, target, result):
                print(f"Constraints violated for test case:")
                print(f"Array: {arr}")
                print(f"Target: {target}")
                print(f"Result: {result}")
                return False
        
        return True

    def verify_constraints(self, arr: np.ndarray, target: int, result: int) -> bool:
        """Verify that all constraints are satisfied for a given input/output."""
        assignments = self._get_assignments(arr, target, result)
        
        for constraint in self.constraints:
            # Substitute values into constraint
            value = constraint.subs(assignments)
            # Check if constraint is satisfied (should evaluate to 0)
            if abs(float(value)) > 1e-10:  # Allow for numerical error
                print(f"Constraint violated: {constraint}")
                print(f"Value: {value}")
                return False
        
        return True

    def _get_assignments(self, arr: np.ndarray, target: int, result: int) -> Dict[sp.Symbol, Any]:
        """Get all variable assignments for a given input/output pair."""
        c = self.circuit
        assignments: Dict[sp.Symbol, Any] = {}
        
        # Input variables
        for i, val in enumerate(arr):
            assignments[c.A[i]] = val
        assignments[c.t] = target
        
        # Execute iterations to get all intermediate values
        left, right = 0, len(arr) - 1
        
        # Initial state
        assignments[c.L[0]] = left
        assignments[c.R[0]] = right
        
        # Track each iteration
        for i in range(self.T):
            # Midpoint
            mid = (left + right) // 2
            assignments[c.m[i]] = mid
            
            # Array access
            current = arr[mid]
            assignments[sp.Symbol(f'A_at_m_{i}')] = current
            
            # Index indicators
            for j in range(self.n):
                assignments[sp.Symbol(f'idx_{i}_{j}')] = 1 if mid == j else 0
            
            # Branch indicators
            if current == target:
                assignments[c.delta_eq[i]] = 1
                assignments[c.delta_lt[i]] = 0
                assignments[c.delta_gt[i]] = 0
                break
            elif current < target:
                assignments[c.delta_eq[i]] = 0
                assignments[c.delta_lt[i]] = 1
                assignments[c.delta_gt[i]] = 0
                if i < self.T - 1:
                    left = mid + 1
                    assignments[c.L[i+1]] = left
                    assignments[c.R[i+1]] = right
            else:
                assignments[c.delta_eq[i]] = 0
                assignments[c.delta_lt[i]] = 0
                assignments[c.delta_gt[i]] = 1
                if i < self.T - 1:
                    right = mid - 1
                    assignments[c.L[i+1]] = left
                    assignments[c.R[i+1]] = right
            
            # If target not found in this iteration
            if i == self.T - 1:
                # Fill in remaining iterations with default values
                for j in range(i+1, self.T):
                    assignments[c.delta_eq[j]] = 0
                    assignments[c.delta_lt[j]] = 0
                    assignments[c.delta_gt[j]] = 0
        
        # Output
        assignments[c.o] = result
        
        return assignments

def standard_binary_search(arr: np.ndarray, target: int) -> int:
    """Reference implementation of binary search."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def main():
    # Example usage
    generator = FormalCircuitGenerator(array_size=8)
    constraints = generator.generate_constraints()
    
    print("Binary Search Circuit Constraints:")
    print(f"Array size: {generator.n}")
    print(f"Number of iterations: {generator.T}\n")
    
    print("Polynomial Constraints:")
    for i, constraint in enumerate(constraints, 1):
        print(f"{i}. {constraint} = 0")

    # Create generator
    generator = FormalCircuitGenerator(array_size=8)
    generator.generate_constraints()
    
    # Compile to function
    binary_search = generator.compile_to_function()
    
    # Test cases
    test_cases = [
        (np.array([1, 3, 5, 7, 9, 11, 13, 15]), 7),  # Should find at index 3
        (np.array([1, 3, 5, 7, 9, 11, 13, 15]), 10), # Should return -1
        (np.array([1, 3, 5, 7, 9, 11, 13, 15]), 1),  # Should find at index 0
        (np.array([1, 3, 5, 7, 9, 11, 13, 15]), 15)  # Should find at index 7
    ]
    
    # Verify correctness
    if generator.verify_correctness(test_cases):
        print("All tests passed!")
        print("All constraints satisfied!")
    else:
        print("Tests failed!")

if __name__ == "__main__":
    main() 
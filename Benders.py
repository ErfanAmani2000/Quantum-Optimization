import numpy as np
import pulp
from scipy.optimize import linprog

# Example data (adjust as needed)
M = 1000000
S_u = np.array([0, 20, 30])  
E_u = np.array([100, 50, 80])
t_ij = np.array([[0, 10, 20,  M],
                 [M,  0, 15, 25],
                 [M, 15,  0,  5],
                 [M,  M,  M,  0]])

# Define the directed graph (nodes and directed edges)
nodes = [1, 2, 3, 4]
edges = [(1, 2), (1, 3), (2, 3), (3, 2), (2, 4), (3, 4)]  # Directed edges

# Define the master problem using PuLP
def master_problem():
    model = pulp.LpProblem("Master_Problem", pulp.LpMinimize)

    # Define decision variables (binary variables for x_ij^u and continuous variable for theta)
    x = {}
    for u in range(3):  # For D, R1, R2
        for (i, j) in edges:
            x[u, i, j] = pulp.LpVariable(f"x_{u}_{i}_{j}", cat=pulp.LpBinary)
    
    theta = pulp.LpVariable("theta", lowBound=0)

    # Objective function: Minimize sum of costs and theta (set only once)
    model += pulp.lpSum(t_ij[i-1][j-1] * x[u, i, j] for u in range(3) for (i, j) in edges) + theta, "Objective"

    # Constraints: Implement the constraints from the master problem
    for u in range(3):
        # Ensure each path from u is covered
        model += x[u, 1, 2] + x[u, 1, 3] == 1  # Node 1 must go to 2 or 3
        model += x[u, 2, 4] + x[u, 3, 4] == 1  # Node 2 or 3 must go to 4

        # Implement flow conservation for the directed graph (enforcing flow direction)
        model += x[u, 1, 2] + x[u, 2, 3] - x[u, 3, 2] - x[u, 2, 4] == 0
        model += x[u, 1, 3] + x[u, 3, 4] - x[u, 2, 3] - x[u, 3, 2] == 0

    # Linking constraints between x^D, x^R1, x^R2
    for (i, j) in edges:
        model += x[1, i, j] >= x[0, i, j]  # x^R1 >= x^D
        model += x[2, i, j] >= x[0, i, j]  # x^R2 >= x^D
        # model += x[0, i, j] != 1 + 2 * x[1, i, j]  # x^D != 1 + 2x^R1
        # model += x[0, i, j] != 1 + 2 * x[2, i, j]  # x^D != 1 + 2x^R2

    # Solve the master problem
    model.solve()

    # Check solver status and print more debug info
    if pulp.LpStatus[model.status] != "Optimal":
        print(f"Master problem status: {pulp.LpStatus[model.status]}")
        print("Solver error or infeasible problem.")
        print(f"Solver status: {pulp.LpStatus[model.status]}")
        print(f"Objective value: {model.objective.value()}")
        return None, None
    
    # Return the solution as a dictionary of variable names and values
    solution = {var.name: var.varValue for var in model.variables()}
    objective_value = pulp.value(model.objective)
    return solution, objective_value

# Define the sub-problem (linear programming)
def sub_problem(master_solution):
    if master_solution is None:
        return None

    # Extract the solution (we need the values of the decision variables)
    x_values = master_solution  # master_solution is a dictionary, access it directly

    # Debugging print to check the variable values
    print(f"Master solution keys: {list(x_values.keys())}")

    # Example of how to access one decision variable (for testing)
    for key in x_values:
        print(f"{key}: {x_values[key]}")  # Print variable names and their values
    
    # Dummy setup for the sub-problem (replace with actual sub-problem formulation)
    c = np.zeros(6)  # Example: dummy zero objective
    bounds = [(0, None)] * 6  # Non-negative constraints

    # Set up constraints for the sub-problem (this can be customized further)
    A_ub = np.array([[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [-1, 1, 0, 0, 1, -1]])
    b_ub = np.array([0, 0, 0])

    # Solve the sub-problem using scipy linprog
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x  # Return dual variables (alpha, beta, gamma, delta, lambda, mu)
    else:
        print("Sub-problem is infeasible.")
        return None

# Main loop for Benders' Decomposition
def benders_decomposition():
    max_iterations = 100
    convergence_tolerance = 1e-5
    theta = float('inf')
    k = 0

    while k < max_iterations:
        print(f"Iteration {k + 1}")

        # Solve master problem
        master_solution, master_obj = master_problem()
        if master_solution is None or master_obj is None:
            print("No optimal solution found for the master problem.")
            break
        
        print(f"Master problem objective: {master_obj}")

        # Solve sub-problem
        sub_solution = sub_problem(master_solution)
        
        if sub_solution is not None:  # Check if the sub-problem was successful
            print(f"Sub-problem solution: {sub_solution}")
        else:
            print("Sub-problem is infeasible.")
            break
        
        # Check for convergence
        if np.abs(master_obj - theta) < convergence_tolerance:
            print(f"Converged with objective: {master_obj}")
            break
        
        # Update the theta value for the next iteration
        theta = master_obj  # Update theta
        k += 1

    return master_solution, theta

# Run Benders' Decomposition
solution, final_obj = benders_decomposition()
print("Final Solution:", solution)
print("Final Objective Value:", final_obj)

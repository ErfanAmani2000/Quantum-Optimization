import numpy as np
from scipy.optimize import linprog

# Example data for t_ij (4x4 matrix representing times between nodes)
t_ij = np.array([[0, 10, 20, 40],
                 [10, 0, 15, 25],
                 [20, 15, 0, 5],
                 [40, 25, 5, 0]])

# Example X values (solution from master problem), you should modify these as per your problem setup
X = {
    'x_0_1_2': 0.0, 'x_0_1_3': 0.0, 'x_0_2_3': 0.0, 'x_0_2_4': 0.0, 'x_0_3_2': 0.0, 'x_0_3_4': 0.0,
    'x_1_1_2': 0.0, 'x_1_1_3': 0.0, 'x_1_2_3': 0.0, 'x_1_2_4': 0.0, 'x_1_3_2': 0.0, 'x_1_3_4': 0.0,
    'x_2_1_2': 0.0, 'x_2_1_3': 0.0, 'x_2_2_3': 0.0, 'x_2_2_4': 0.0, 'x_2_3_2': 0.0, 'x_2_3_4': 0.0
}

# Create x_values dictionary from X
x_values = {key: X[key] for key in X}

# We now flatten the t_ij matrix to match the length of the decision variables (x_ij).
c = np.array([t_ij[i-1][j-1] for i in range(1, 5) for j in range(1, 5)])

# Bounds for the decision variables (x_ij), assume binary values (0 or 1)
bounds = [(0, 1) for _ in range(len(c))]

# The number of rows in A_ub should match the number of variables in c (16 variables).
A_ub = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [-1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

b_ub = np.array([0, 0, 0])  # Dummy constraints for testing

# Solve the sub-problem using linprog
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Check if the sub-problem was solved successfully
if result.success:
    sub_problem_solution = result.x
    print("Sub-problem solution found:")
    print(sub_problem_solution)
else:
    print("Sub-problem is infeasible.")

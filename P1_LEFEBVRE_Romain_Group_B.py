import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.optimize import minimize

def objective_function(xy):
    """
    Calculate the total profit based on the given values of x and y.
    
    Args:
        xy: A list or tuple containing the values of x and y.

    Returns:
        Total profit as a float.
    """
    x, y = xy
    wheat_profit = 2 * x**2 + 3 * x * y - y**2  # Profit from wheat
    barley_profit = x**2 - 4 * x * y + 5 * y**2  # Profit from barley
    return wheat_profit + barley_profit

def negative_objective_function(xy):
    """
    Calculate the negative total profit to use in optimization.
    Used in the solver for minimizing.
    
    Args:
        xy: A list or tuple containing the values of x and y.

    Returns:
        Negative total profit as a float.
    """
    return -objective_function(xy) 

def constraint_1(xy):
    """
    Perimeter constraint.
    
    Args:
        xy: A list or tuple containing the values of x and y.

    Returns:
        Value representing the difference from the maximum allowed perimeter.
    """
    x, y = xy
    return 100 - 2 * (x + y)  # Maximum perimeter is 100

def constraint_2(xy):
    """
    Area constraint.
    
    Args:
        xy: A list or tuple containing the values of x and y.

    Returns:
        Value representing the difference from the maximum allowed area.
    """
    x, y = xy
    return 600 - x * y  # Maximum area is 600

def constraint_3(xy):
    """
    Non-linear constraint for y based on x.
    
    Args:
        xy: A list or tuple containing the values of x and y.

    Returns:
        Value representing the difference from the calculated boundary for y.
    """
    x, y = xy
    return y - ((x**2 / 50) + 2)  # Non-linear constraint for y

def parse_arguments():
    """
    Parse command-line arguments for variable bounds.
    Includes documentations and help.
    
    Returns:
        Parsed arguments containing bounds for x and y.
    """
    parser = argparse.ArgumentParser(description="Optimized profit based on constraints.")
    
    # Define argument for lower and upper bounds of x and y
    parser.add_argument('--x_lower', type=float, required=True, help="Lower bound for variable x")
    parser.add_argument('--x_upper', type=float, required=True, help="Upper bound for variable x")
    parser.add_argument('--y_lower', type=float, required=True, help="Lower bound for variable y")
    parser.add_argument('--y_upper', type=float, required=True, help="Upper bound for variable y")
    
    return parser.parse_args() 

def solve_problem(list_vars):
    """
    Solve the optimization problem using SLSQP method.
    
    Args:
        list_vars: List of variable bounds for x and y.

    Returns:
        Optimization result from the minimize function.
    """
    # Initial guess for x and y values based on the documentation
    initial_guess = [(list_vars[0][0] + list_vars[0][1]) / 2, (list_vars[1][0] + list_vars[1][1]) / 2]

    # Define constraints for the optimization problem
    constraints = [{'type': 'ineq', 'fun': constraint_1},
                   {'type': 'ineq', 'fun': constraint_2},
                   {'type': 'ineq', 'fun': constraint_3}]

    # Bounds for x and y
    bounds = [list_vars[0], list_vars[1]]

    # Perform optimization using SLSQP by scipy
    result = minimize(negative_objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result 

def display_plot_objective_feasible_region(result, list_vars):
    """
    Display a 3D plot of the objective function and feasible region.
    
    Args:
        result: Optimization result containing the optimal values of x and y.
        list_vars: List of variable bounds for x and y.
    """
    x_vals = np.linspace(list_vars[0][0], list_vars[0][1], 200)
    y_vals = np.linspace(list_vars[1][0], list_vars[1][1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calculate Z values for the objective function
    Z = np.array([[objective_function([x, y]) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the objective function surface with transparency so that the feasible region is more visible
    ax.plot_surface(X, Y, Z, cmap='inferno', alpha=0.4, edgecolor='none')

    # Evaluate constraints to determine the feasible region
    Z_perimeter = np.array([[constraint_1([x, y]) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])
    Z_area = np.array([[constraint_2([x, y]) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])
    Z_nonlinear = np.array([[constraint_3([x, y]) for x, y in zip(X_row, Y_row)] for X_row, Y_row in zip(X, Y)])

    # Combine constraints to find the feasible region
    feasible_region = (Z_perimeter >= 0) & (Z_area >= 0) & (Z_nonlinear >= 0)

    # Mask the objective function where constraints are not satisfied
    Z_masked = np.where(feasible_region, Z, np.nan)
    ax.plot_surface(X, Y, Z_masked, color='green', alpha=0.9)  # Plot feasible region

    # Plot the optimal point
    optimal_x = result.x[0]
    optimal_y = result.x[1]
    optimal_z = objective_function([optimal_x, optimal_y])
    optimal_point = ax.scatter(optimal_x, optimal_y, optimal_z, color='blue', s=100, label='Optimized solution', zorder=10)
    
    # Set plot titles and labels and view angle
    ax.set_title('Results of the calculations')
    ax.set_xlabel('x (Length)')
    ax.set_ylabel('y (Width)')
    ax.set_zlabel('Profit')
    ax.view_init(30, 220) 

    # Add legend
    proxy = Patch(color=plt.cm.inferno(0.5), label='Objective function')
    proxy_1 = Patch(color='green', label='Feasible region')
    plt.legend(handles=[proxy, proxy_1, optimal_point])

    # Show the figure in interactive mode
    plt.ion()
    plt.show()
    plt.pause(999999)

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    list_vars = [[args.x_lower, args.x_upper], [args.y_lower, args.y_upper]]
    
    # Solve the optimization problem
    result = solve_problem(list_vars)

    # Output the solution
    if result.success:
        print(f"Optimal solution: x = {result.x[0]:.2f}, y = {result.x[1]:.2f}")
        print(f"Maximum Profit: {-result.fun:.2f}")
    else:
        print("Optimization failed!")

    # Plot the feasible region and the objective function with the optimal point
    display_plot_objective_feasible_region(result, list_vars)
import numpy as np

def verify_concavity_from_grid(f_grid, h_x=0.01, h_y=0.01):
    """
    [Version 2: Adapted for row-column layout]
    Uses numerical Hessian estimation to verify the local concavity at the center
    of a 3x3 function value grid.

    This function is designed for maximization tasks (e.g., R@50), and checks
    whether the function has a "hill-shaped" surface at the center.

    Args:
        f_grid (np.ndarray or list of lists): A 3x3 matrix of function values (e.g., R@50)
            from 9 experimental points. The layout must strictly follow the structure:
            [ [f(x-h, y-h), f(x-h, y), f(x-h, y+h)],  <- top row (x_center - h_x)
              [f(x,   y-h), f(x,   y), f(x,   y+h)],  <- middle row (x_center)
              [f(x+h, y-h), f(x+h, y), f(x+h, y+h)] ] <- bottom row (x_center + h_x)
        h_x (float): Step size for the first hyperparameter (corresponds to row direction).
        h_y (float): Step size for the second hyperparameter (corresponds to column direction).

    Returns:
        dict: A dictionary containing the concavity check result and detailed computations.
    """
    f_grid = np.array(f_grid)
    if f_grid.shape != (3, 3):
        raise ValueError("Input matrix f_grid must have shape 3x3.")

    # Assign descriptive variable names for all 9 values
    # Indexing [row, col] -> [x_offset, y_offset]
    f_c_yp = f_grid[1, 2]  # f(x, y+h)
    f_c_ym = f_grid[1, 0]  # f(x, y-h)

    f_xp_c = f_grid[2, 1]  # f(x+h, y)
    f_xm_c = f_grid[0, 1]  # f(x-h, y)

    f_c_c = f_grid[1, 1]   # f(x, y) - center point

    f_xp_yp = f_grid[2, 2] # f(x+h, y+h)
    f_xm_yp = f_grid[0, 2] # f(x-h, y+h)
    f_xp_ym = f_grid[2, 0] # f(x+h, y-h)
    f_xm_ym = f_grid[0, 0] # f(x-h, y-h)

    # --- Compute Hessian components ---
    H11 = (f_xp_c - 2 * f_c_c + f_xm_c) / (h_x ** 2)  # ∂²f/∂x²
    H22 = (f_c_yp - 2 * f_c_c + f_c_ym) / (h_y ** 2)  # ∂²f/∂y²
    H12 = (f_xp_yp - f_xm_yp - f_xp_ym + f_xm_ym) / (4 * h_x * h_y)  # ∂²f/∂x∂y

    # --- Check concavity conditions ---
    determinant = H11 * H22 - H12 ** 2

    # Concavity condition: H11 < 0 and determinant > 0
    is_concave = (H11 < 0) and (determinant > 0)

    # Generate human-readable conclusion message
    if is_concave:
        message = "Passed: Function is locally concave (hill-shaped)."
    elif H11 > 0 and determinant > 0:
        message = "Failed: Function is locally convex (bowl-shaped)."
    elif determinant < 0:
        message = "Failed: Function is a saddle point or undefined shape at this location."
    else:
        message = "Inconclusive: Conditions for concave/convex/saddle are not clearly met."

    results = {
        'is_concave': is_concave,
        'message': message,
        'H11': H11,
        'H22': H22,
        'H12': H12,
        'determinant': determinant
    }

    return results


# --- Fill in your 9 experimental results here ---
# Please follow the coordinate layout strictly
# For example, r50_grid[0][0] corresponds to (x-h, y+h)
# r50_grid[1][1] is the center point result

r50_grid_from_experiments = np.array([
    [0.4105, 0.4069, 0.4073],  # Results for experiments 1, 2, 3 (top row)
    [0.4106, 0.4123, 0.4086],  # Results for experiments 4, 5, 6 (middle row)
    [0.4119, 0.4086, 0.4134]   # Results for experiments 7, 8, 9 (bottom row)
])

# Define step size
h = 0.01

# Run concavity verification
# Both step sizes are set to 0.01
results = verify_concavity_from_grid(r50_grid_from_experiments, h_x=h, h_y=h)

# Print analysis report
print("="*40)
print("Hessian Concavity Analysis Report")
print("="*40)
print(f"Step size h: {h}")
print("\n--- Conclusion ---")
print(results['message'])
print("\n--- Detailed Results ---")
print(f"H11 (curvature along factor_upper): {results['H11']:.4f}")
print(f"H22 (curvature along factor_lower): {results['H22']:.4f}")
print(f"H12 (mixed curvature):              {results['H12']:.4f}")
print(f"Hessian determinant:                {results['determinant']:.4f}")

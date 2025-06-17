import numpy as np

def Maximize(objective_function: callable, initial_theta: np.ndarray, num_iterations: int=100,
                a: float=0.1, c: float=0.1, A: float=None, alpha: float=0.602, gamma: float=0.101,
                verbose: bool=False):
    """
    Implements the Simultaneous Perturbation Stochastic Approximation (SPSA)
    algorithm for maximization.

    Args:
        objective_function (callable): The function to maximize. It should take
                                    a NumPy array (parameters) as input and
                                    return a single scalar value.
        initial_theta (np.ndarray): A NumPy array representing the initial
                                    guess for the parameters (theta).
        num_iterations (int): The maximum number of iterations to perform.
        a (float): Positive constant in the gain sequence ak.
                Determines the initial step size.
        c (float): Positive constant in the gain sequence ck.
                Determines the initial perturbation magnitude.
        A (float, optional): Non-negative constant in the gain sequence ak.
                            Used to stabilize the gain sequence.
                            If None, it defaults to num_iterations * 0.1.
        alpha (float): Exponent for the gain sequence ak, typically 0.602.
        gamma (float): Exponent for the gain sequence ck, typically 0.101.
        verbose (bool): If True, prints progress updates during optimization.

    Returns:
        np.ndarray: The estimated optimal parameters (theta) found by the algorithm.
        list: A list of the objective function values at each iteration.
        list: A list of the parameter arrays at each iteration.
    """
    theta = np.asarray(initial_theta, dtype=float)
    p = len(theta)  # Number of parameters

    if A is None:
        A = num_iterations * 0.1 # A common heuristic for A

    # Store history for analysis
    history_theta = [theta.copy()]
    history_obj_val = [objective_function(theta)]

    if verbose:
        print(f"Starting SPSA Optimization (Maximization) with {p} parameters.")
        print(f"Initial theta: {theta}")
        print(f"Initial objective value: {history_obj_val[0]:.4f}")

    for k in range(num_iterations):
        # 1. Generate gain sequences ak and ck
        ak = a / (k + 1 + A)**alpha
        ck = c / (k + 1)**gamma

        # 2. Generate perturbation vector (Delta_k)
        # Bernoulli random variables {-1, 1} are common for SPSA
        delta_k = (np.random.randint(0, 2, p) * 2 - 1).astype(float)

        # 3. Perturb parameters in two directions
        theta_plus = theta + ck * delta_k
        theta_minus = theta - ck * delta_k

        # 4. Evaluate the objective function at perturbed points
        y_plus = objective_function(theta_plus)
        y_minus = objective_function(theta_minus)

        # 5. Estimate the gradient (g_hat_k)
        # For maximization, we move in the direction of the estimated gradient
        # The formula for gradient estimation remains the same.
        # This is the finite difference approximation along the perturbation.
        g_hat_k = (y_plus - y_minus) / (2 * ck * delta_k)

        # 6. Update parameters
        # For maximization, we ADD the scaled gradient estimate
        theta = theta + ak * g_hat_k

        # Store current state
        current_obj_val = objective_function(theta)
        history_theta.append(theta.copy())
        history_obj_val.append(current_obj_val)

        if verbose and (k + 1) % (num_iterations // 10 if num_iterations >= 10 else 1) == 0:
            print(f"Iteration {k+1}/{num_iterations}:")
            print(f"  Current theta: {np.round(theta, 4)}")
            print(f"  Current objective value: {current_obj_val:.4f}")

    if verbose:
        print("\nOptimization Finished.")
        print(f"Final estimated optimal theta: {np.round(theta, 4)}")
        print(f"Final objective value: {history_obj_val[-1]:.4f}")

    return theta, history_obj_val, history_theta
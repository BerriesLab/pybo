def _validate_XY(X, Y):
    if X is None or Y is None:
        raise ValueError("X and Y cannot be none")
    if not X.shape[0] == Y.shape[0]:
        raise ValueError("The number of samples in X_init and Y_init must match.")


def _validate_f0(n_objectives, f0):
    if f0 is not None and not n_objectives == len(f0):
        raise ValueError("The number of objectives must equal the number of objective functions.")


def _validate_bounds(bounds, n):
    if not isinstance(bounds, list):
        raise ValueError("Bounds must be a list.")
    else:
        for bound in bounds:
            if not len(bound) == 2:
                raise ValueError("Bounds must be a list of tuples of two numeric values.")

            if not bound[0] < bound[1]:
                raise ValueError("Lower bound must be smaller than upper bound.")

            if not isinstance(bound[0], (int, float)) and not isinstance(bound[1], (int, float)):
                raise ValueError("Bounds must be a list of tuples of two numeric values.")

    if not len(n) == len(bounds):
        raise ValueError("The number of samples must be the same length as the number of bounds.")
    else:
        for n_ in n:
            if not isinstance(n_, int):
                raise ValueError("The number of samples must be a list of integers.")
            if n_ <= 0:
                raise ValueError("The number of samples must be a positive integer.")
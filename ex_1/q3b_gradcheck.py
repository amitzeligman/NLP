import random

import numpy as np


def gradcheck_naive(f, x, gradient_text="gradcheck function."):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradient_text -- a string detailing some context about the gradient computation
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4         # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute numerical
        # gradients (numgrad).

        # Use the centered difference of the gradient.
        # It has smaller asymptotic error than forward / backward difference
        # methods. If you are curious, check out here:
        # https://math.stackexchange.com/questions/2326181/when-to-use-forward-or-central-difference-approximations

        # Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.



        random.setstate(rndstate)
        numgrad = (f(x[ix] + h)[0] - f(x[ix] - h)[0]) / (2 * h)


        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for {}.".format(gradient_text))
            print("First gradient error found at index {} in the vector of gradients".format(ix))
            print("Your gradient: {} \t Numerical gradient: {}".format(grad[ix], numgrad))
            return

        it.iternext()  # Step to next dimension

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print()


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """

    print("Running your sanity checks...")
    quad = lambda x: (np.sum(x ** 3), 3 * (x ** 2))
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print()
    quad = lambda x: (np.sum(2 * x ** 5 + 2 * x), 10 * x ** 4 + 2)
    gradcheck_naive(quad, np.array(123.456))  # scalar test
    gradcheck_naive(quad, np.random.randn(3, ))  # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))  # 2-D test
    print()


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

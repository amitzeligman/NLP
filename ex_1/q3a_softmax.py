import numpy as np

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        numerator = np.exp(x)
        denominator = np.sum(numerator, axis=1)
        denominator = np.expand_dims(denominator, axis=0).repeat(orig_shape[-1], axis=0)
        x = np.divide(numerator, denominator.transpose())
        ### END YOUR CODE
    else:
        # Vector
        numerator = np.exp(x)
        denominator = np.sum(numerator)
        x = numerator / denominator
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1, 2], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1, -2]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def test_softmax_on_your_own():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    test1 = softmax(np.array([[1, 0.5, 0.2, 3],
                      [1,  -1,   7, 3],
                      [2,  12,  13, 3]]))
    ans1 = np.array([[1.05877e-01, 6.42177e-02, 4.75736e-02, 7.82332e-01],
                     [2.42746e-03, 3.28521e-04, 9.79307e-01, 1.79366e-02],
                     [1.22094e-05, 2.68929e-01, 7.31025e-01, 3.31885e-05]])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)
    test2 = softmax(np.array([0.1, 2, 4, 0.03, 10, 100]))
    ans2 = np.array([4.11132e-44, 2.74879e-43, 2.03109e-42, 3.83337e-44, 8.19401e-40, 1.00000e+00])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)



if __name__ == "__main__":
    test_softmax_basic()
    test_softmax_on_your_own()
import numpy as np


def stab_bb(x0, costFn, gradFn, bb=1, deltaUpdateStrategy='adaptative',
            deltaInput=1e6, c=0.2, maxIt=10000, tol=1e-7, verbose=False,):
    """Wrapper of the stab_BB function to work with scalar functions

    x0 : python scalar | np.array
        First initial point
    costFn: callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (n,)
    gradFn : callabe
        Function to evaluate the gradient of the cost function being minimized
    bb: {1,2}, optional (default 'adaptative')
        Whether to use BB1 or BB2 method
    deltaUpdateStrategy: {'adaptative','constant'}
        Whether to use a constant or adaptative delta parameter
    deltaInput: number, optional (default 1e6)
        If ``deltaUpdateStrategy=='adaptative'`` it's the delta used until the
        fourth iteration. For this case it's suggested to use a large value like
        1e6.
        Else it will be the delta used throughout the experiment.
        The paper was tested setting this value to be 2 or 0.01, 0.1 or 1.0.
    c: number, optional (default 1e6)
        Paramater for adaptative choice of delta. Must be set if
        ``deltaUpdateStrategy=='adaptative'``.
        The paper was tested using 0.1,0.2,0.25 or 0.3 for quadratic cost
        functions and also using 0.1, 0.5 or 1.0 for nonquadratic ones.
    maxIt : int, optional (defalt 500)
        Maximum number of iteration
    tol : float, optional (default 1e-7)
        Value that will cause the algorithm to stop
    verbose : bool, optional (defalt False)
        Whether to print partial results after each iteration
    Returns
    -------
    bestX, (xHistory, alphaHistory, normGradHistory)
        ``bestX`` is (n,) ndarray representing the best evaluated point
        ``xHistory`` is a list of all evaluated points
        ``alphaHistory`` is float list of all evaluate alphas
        ``normGradHistory`` is float list of all evaluated norm of gradients
    References
    ----------
    .. [1] Burdakov, Oleg & Dai, Yu-Hong & Huang, Na. (2019). Stabilized
        Barzilai-Borwein.

    """
    from stabbb import stab_BB
    if not isinstance(x0, np.ndarray):
        x0 = np.asanyarray(x0)
    if x0.ndim == 0:
        x0.shape = (1,)
    return stab_BB(x0, costFn, gradFn, bb, deltaUpdateStrategy, deltaInput, c,
                   maxIt, tol, verbose)

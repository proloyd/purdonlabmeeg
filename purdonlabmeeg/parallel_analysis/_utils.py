import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg, signal, stats
from ._parallel_analysis import parallel_analysis as pa


def _viz_score_correlations(orig_scores, orig_evs, pa_evs, fig):
    gs = fig.add_gridspec(orig_evs.shape[0], orig_evs.shape[0])
    den = orig_scores.shape[1] - np.arange(orig_scores.shape[1])
    xlim = max(orig_scores.shape[1] // 50, 20)
    lags = np.arange(-orig_scores.shape[1]+1, orig_scores.shape[1])
    den = np.hstack((den[1:][::-1], den))
    for i in range(orig_scores.shape[0]):
        for j in range(i+1):
            ax = fig.add_subplot(gs[i, j])
            score_corr = signal.correlate(orig_scores[i], orig_scores[j]) / den
            divider = np.sqrt(np.abs(orig_evs[i] * orig_evs[j]))
            ax.stem(lags, score_corr / divider, markerfmt=',')
            ax.set_xlim([-xlim, xlim])
            if i == j:
                ax.set_title(f'{orig_evs[i]:.2}({pa_evs[i]:.2})')
            if j == 0:
                ax.set_ylabel(f'y{i}')
            if i == orig_scores.shape[0]-1:
                ax.set_xlabel(f'y{j}')
            ax.set_ylim([-1, 1])
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig


def inspect_order(x, max_order=20, user_inspection=False):
    # 1. Set `1 = 0`.
    l = 0
    rs = []
    r_news = []
    print('l \t No. of var \t No. of PCs \t r(l) \t r_new(l)')
    if user_inspection:
        fig = plt.figure(num=-1)
        fig.suptitle('AR order inspection (DO NOT CLOSE THE WINDOW!)', fontsize=16)
        fig.show()
    while True:
        if l > max_order:
            print(f'max_order={max_order} reached, try with higher max_order')
            break
        # 2. Form data matrix `X = [X(k)X(k - 1). . . X(k - l)`.
        xs = [x[:, l:]]
        if l > 0:
            xs.extend([x[:, i:i-l] for i in reversed(range(l))])
        X = np.vstack(xs)
        # 3. Perform PCA and calculate all the principal scores.
        X = stats.zscore(X, axis=1)
        corr = np.dot(X, X.T) / X.shape[1]
        # orig_evs = linalg.eigvalsh(corr)[::-1]
        orig_evs, loadings = linalg.eigh(corr)
        orig_evs = orig_evs[::-1]
        loadings = loadings[:, ::-1]
        orig_scores = loadings.T.dot(X)
        pa_evs = pa(*X.shape, repeat=100)
        pa_evs = pa_evs.mean(axis=0)
        last_principle_ev = np.nonzero(orig_evs > pa_evs)[0][-1]
        # User inspection
        if user_inspection:
            _viz_score_correlations(orig_scores, orig_evs, pa_evs, fig)
            last_principle_ev_ = input(
                "Enter last principle eigen-value number (in pythonic numbering): ")
            last_principle_ev = max(last_principle_ev, int(last_principle_ev_))
            fig.clear()
        # 4. Set `j = n X (l + 1)` and `r(l) = 0`.
        # 5. Determine if the `j`th component represents a linear relation.
        # If yes proceed, if no go to step 7
        # 6. Set j = j - 1 and r(l) = r(l) + 1, repeat 5
        this_r = X.shape[0] - last_principle_ev - 1
        rs.append(this_r)
        # 7. Calculate the number of new relationships
        this_r_new = (this_r -
                      sum((l - np.arange(l) + 1) * np.asanyarray(r_news)))
        r_news.append(this_r_new)
        print(f'{l} \t {X.shape[0]} \t {last_principle_ev + 1}' +
              f' \t {this_r} \t {this_r_new}')
        # 8. If r_new(l) <= 0, go to step 10, otherwise proceed
        if r_news[-1] <= 0:
            if l > 0 and r_news[-2] > 0:
                l -= 1
                break
        # 9. Set 1 = 1 + 1, go to step 2.
        l += 1
        # 10. Stop
    if user_inspection:
        plt.close(fig)
    print(f"Chosen order of the system: {l}")
    return l, rs, r_news


def test_inspect_order():
    from numpy.random import default_rng
    rng = default_rng(seed=0)
    w = rng.standard_normal(1000)
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal()
    z[0] = rng.standard_normal()
    for k in range(1, 1000):
        u[k] = 0.7 * u[k-1] + w[k-1]
        z[k] = 0.8 * z[k-1] + u[k-1]
    x = np.vstack((z, u))

    inspect_order(x)


def test_inspect_order2():
    from numpy.random import default_rng
    from math import sqrt
    a = np.array([[0.118, -0.191], [0.847, 0.264]])
    b = np.array([[1, 2], [3, -4]])
    c = np.array([[0.811, -0.226], [0.477, 0.415]])
    d = np.array([[0.193, 0.689], [-0.320, -0.749]])
    rng = default_rng(seed=0)
    w = rng.standard_normal((2, 100)).T
    v = sqrt(0.1) * rng.standard_normal((2, 100)).T
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal(2)
    z[0] = rng.standard_normal(2)
    for k in range(1, 100):
        u[k] = c.dot(u[k-1]) + d.dot(w[k-1])
        z[k] = a.dot(z[k-1]) + b.dot(u[k-1])
    y = z + v
    x = np.vstack((y.T, u.T))

    inspect_order(x)

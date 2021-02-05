import numpy as np
from scipy import linalg


def lattice(p, x, method='NS'):
	"""Computes multichannel autoregressive parameter matrix using either the
	Vieira-Morf algorithm (multi-channel generalization of the single-channel
	geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
	generalization of the single-channel Burg lattice algorithm).
	
	Parameters:
	p: int
		order of multichannel AR filter
	x: ndarray
		sample data array: x(channel #, sample #)
	method: str 
		’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)
		
	Returns:
	rho_f: ndarray
		forward linear prediction error/white noise covariance matrix
	a: list of ndarrays
		block vector of forward linear prediction/autoregressive matrix elements
	rho_b: ndarray
		backward linear prediction error/white noise covariance matrix
	b: list of ndarrays
		block vector of backward linear prediction/autoregressive matrix elements
	
	NOTE: 'VM'-method is more resilient to additive noise.
	"""
	factors, n = x.shape
	# e_f = x.copy()
	# e_b = x.copy()
	# _e_f = np.empty_like(x)
	# _e_b = np.empty_like(x)
	# _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
	# _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
	# _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
	rho_f = x.dot(x.T) / n
	rho_b = rho_f.copy()
	_rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n
	_rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n
	_rho_fb = x[:, p+1:].dot(x[:, p:-1].T) / n
	a = [np.eye(factors)]
	b = [np.eye(factors)]
	return _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method)


def lattice_kalman_ss(p, x, cov, cross_cov, method='NS',):
	"""Computes multichannel autoregressive parameter matrix using either the
	Vieira-Morf algorithm (multi-channel generalization of the single-channel
	geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
	generalization of the single-channel Burg lattice algorithm).
	
	Parameters:
	p: int
		order of multichannel AR filter
	x: ndarray
		sample data array: x(channel #, sample #)
	method: str 
		’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)
	cov: ndarray
		estimation error covariance matrix
	cross_cov: ndarray
		estimation error cross-covariance matrix
	Returns:
	rho_f: ndarray
		forward linear prediction error/white noise covariance matrix
	a: list of ndarrays
		block vector of forward linear prediction/autoregressive matrix elements
	rho_b: ndarray
		backward linear prediction error/white noise covariance matrix
	b: list of ndarrays
		block vector of backward linear prediction/autoregressive matrix elements
	
	NOTE: 'VM'-method is more resilient to additive noise.
	"""
	factors, n = x.shape
	# e_f = x.copy()
	# e_b = x.copy()
	# _e_f = np.empty_like(x)
	# _e_b = np.empty_like(x)
	# _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
	# _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
	# _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
	rho_f = x.dot(x.T) / n + cov
	rho_b = rho_f.copy()
	_rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n + cov
	_rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n + cov
	_rho_fb = x[:, p+1:].dot(x[:, p:-1].T) / n + cross_cov
	a = [np.eye(factors)]
	b = [np.eye(factors)]
	return _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method)


def lattice_kalman(p, x, cov, cross_cov, method='NS',):
	"""Computes multichannel autoregressive parameter matrix using either the
	Vieira-Morf algorithm (multi-channel generalization of the single-channel
	geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
	generalization of the single-channel Burg lattice algorithm).
	
	Parameters:
	p: int
		order of multichannel AR filter
	x: ndarray
		sample data array: x(channel #, sample #)
	method: str 
		’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)
	cov: ndarray
		estimation error covariance matrix
	cross_cov: ndarray
		estimation error cross-covariance matrix
	Returns:
	rho_f: ndarray
		forward linear prediction error/white noise covariance matrix
	a: list of ndarrays
		block vector of forward linear prediction/autoregressive matrix elements
	rho_b: ndarray
		backward linear prediction error/white noise covariance matrix
	b: list of ndarrays
		block vector of backward linear prediction/autoregressive matrix elements
	
	NOTE: 'VM'-method is more resilient to additive noise.
	"""
	factors, n = x.shape
	# e_f = x.copy()
	# e_b = x.copy()
	# _e_f = np.empty_like(x)
	# _e_b = np.empty_like(x)
	# _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
	# _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
	# _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
	rho_f = x.dot(x.T) / n + cov.sum(axis=0) / n
	rho_b = rho_f.copy()
	_rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n + cov[p+1:].sum(axis=0) / n
	_rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n + cov[p:-1].sum(axis=0) / n
	_rho_fb = x[:, p+1:].dot(x[:, p:-1].T) / n + cross_cov[p+1:].sum(axis=0) / n
	a = [np.eye(factors)]
	b = [np.eye(factors)]
	return _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method)


def _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method='NS',):
	for i in range(1, p+1):
		if method == 'NS':
			rho_f_inv = linalg.inv(rho_f)
			rho_b_inv = linalg.inv(rho_b)
			ip_a = _rho_f.dot(rho_f_inv)
			ip_b = _rho_b.dot(rho_b_inv)
			ip_q = 2 * _rho_fb
			delta = linalg.solve_sylvester(ip_a, ip_b, ip_q)		
			a_p = - delta.dot(rho_b_inv)
			b_p = - delta.T.dot(rho_f_inv)
		elif method == 'VM':
			_rho_f_sqrt_inv = linalg.inv(linalg.cholesky(_rho_f, lower=True))
			_rho_b_sqrt_inv = linalg.inv(linalg.cholesky(_rho_b, lower=True))
			lambdap = _rho_f_sqrt_inv.dot(_rho_fb.dot(_rho_b_sqrt_inv.T))
			rho_f_sqrt = linalg.cholesky(rho_f, lower=True)
			rho_b_sqrt = linalg.cholesky(rho_b, lower=True)
			rho_f_sqrt_inv = linalg.inv(rho_f_sqrt)
			rho_b_sqrt_inv = linalg.inv(rho_b_sqrt)
			a_p = - rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt_inv))
			b_p = - rho_b_sqrt.dot(lambdap.dot(rho_f_sqrt_inv))
			delta = rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt.T))

		# a, b updates
		a.append(np.zeros_like(a_p))  # replace 0 by zeros
		b.insert(0, np.zeros_like(b_p))  # replace 0 by zeros
		for a_i, b_i in zip(a, b):
			new_a_i = a_i + a_p.dot(b_i) 
			new_b_i = b_i + b_p.dot(a_i) 
			a_i[:] = new_a_i
			b_i[:] = new_b_i
		## rho updates
		rho_f = rho_f + a_p.dot(delta.T)
		rho_b = rho_b + b_p.dot(delta)
		## error updates
		# _e_f[:] = e_f[:]
		# _e_b[:] = e_b[:]
		# e_f[:, 1:] += a_p.dot(_e_b[:, :-1])
		# e_b[:, 1:] += b_p.dot(_e_f[:, :-1])
		a_p_rho_b = a_p.dot(_rho_b)
		b_p_rho_f = b_p.dot(_rho_f)
		_rho_fb_a_pT = _rho_fb.dot(a_p.T)
		b_p_rho_fb = b_p.dot(_rho_fb)
		__rho_f = _rho_f + a_p_rho_b.dot(a_p.T) + _rho_fb_a_pT + _rho_fb_a_pT.T
		__rho_b = _rho_b + b_p_rho_f.dot(b_p.T) + b_p_rho_fb.T + b_p_rho_fb
		__rho_fb = _rho_fb + _rho_fb_a_pT.T.dot(b_p.T) + a_p_rho_b + b_p_rho_f.T
		
		_rho_f = __rho_f
		_rho_b = __rho_b
		_rho_fb = __rho_fb
	
	return rho_f, a, rho_b, b


def __lattice(p, x, method='NS'):
	"""Computes multichannel autoregressive parameter matrix using either the
	Vieira-Morf algorithm (multi-channel generalization of the single-channel
	geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
	generalization of the single-channel Burg lattice algorithm).
	
	Parameters:
	p: int
		order of multichannel AR filter
	x: ndarray
		sample data array: x(channel #, sample #)
	method: str 
		’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)
		
	Returns:
	rho_f: ndarray
		forward linear prediction error/white noise covariance matrix
	a: list of ndarrays
		block vector of forward linear prediction/autoregressive matrix elements
	rho_b: ndarray
		backward linear prediction error/white noise covariance matrix
	b: list of ndarrays
		block vector of backward linear prediction/autoregressive matrix elements
	
	NOTE: 'VM'-method is more resilient to additive noise.
	"""
	factors, n = x.shape
	e_f = x.copy()
	e_b = x.copy()
	_e_f = np.empty_like(x)
	_e_b = np.empty_like(x)
	_rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
	_rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
	_rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
	rho_f = x.dot(x.T) / n
	rho_b = rho_f.copy()
	a = [np.eye(factors)]
	b = [np.eye(factors)]
	for i in range(1, p+1):
		if method == 'NS':
			rho_f_inv = linalg.inv(rho_f)
			rho_b_inv = linalg.inv(rho_b)
			ip_a = _rho_f.dot(rho_f_inv)
			ip_b = _rho_b.dot(rho_b_inv)
			ip_q = 2 * _rho_fb
			delta = linalg.solve_sylvester(ip_a, ip_b, ip_q)		
			a_p = - delta.dot(rho_b_inv)
			b_p = - delta.T.dot(rho_f_inv)
		elif method == 'VM':
			_rho_f_sqrt_inv = linalg.inv(linalg.cholesky(_rho_f, lower=True))
			_rho_b_sqrt_inv = linalg.inv(linalg.cholesky(_rho_b, lower=True))
			lambdap = _rho_f_sqrt_inv.dot(_rho_fb.dot(_rho_b_sqrt_inv.T))
			rho_f_sqrt = linalg.cholesky(rho_f, lower=True)
			rho_b_sqrt = linalg.cholesky(rho_b, lower=True)
			rho_f_sqrt_inv = linalg.inv(rho_f_sqrt)
			rho_b_sqrt_inv = linalg.inv(rho_b_sqrt)
			a_p = - rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt_inv))
			b_p = - rho_b_sqrt.dot(lambdap.dot(rho_f_sqrt_inv))
			delta = rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt.T))

		# a, b updates
		a.append(np.zeros_like(a_p))  # replace 0 by zeros
		b.insert(0, np.zeros_like(b_p))  # replace 0 by zeros
		for a_i, b_i in zip(a, b):
			new_a_i = a_i + a_p.dot(b_i) 
			new_b_i = b_i + b_p.dot(a_i) 
			a_i[:] = new_a_i
			b_i[:] = new_b_i
		## rho updates
		rho_f = rho_f + a_p.dot(delta.T)
		rho_b = rho_b + b_p.dot(delta)
		## error updates
		_e_f[:] = e_f[:]
		_e_b[:] = e_b[:]
		e_f[:, 1:] += a_p.dot(_e_b[:, :-1])
		e_b[:, 1:] += b_p.dot(_e_f[:, :-1])
		_rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
		_rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
		_rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n

	
	return rho_f, a, rho_b, b


def test_lattice():
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
		z[k] =  a.dot(z[k-1]) + b.dot(u[k-1])
	y = z + v
	x = np.vstack((y.T, u.T))
	
	results_ = lattice(1, x, 'VM',)
	results = __lattice(1, x, 'VM',)
	assert np.allclose(results[1][1], results_[1][1])
	
	results_ = lattice(1, x, 'NS',)
	results = __lattice(1, x, 'NS',)
	assert np.allclose(results[1][1], results_[1][1])
	
	
import matplotlib.pyplot as plt
def scatter_with_errorbars(x, y, xerr, yerr, capsize=10.0, linestyle='None', color='b', marker='o', ax=None):
	if ax is None:
		fig, ax = plt.subplots(1)
	ax.errorbar(a, b, yerr=yerr, xerr=xerr, fmt=color+marker, capsize=capsize)
	return ax
	
	
if __name__ == '__main__':
	a = [1,3,5,7]
	b = [11,-2,4,19]
	c_a = [1,3,2,1]
	c_b = [3,2,1,1]
	fig, ax = plt.subplots(1)
	scatter_with_errorbars(a, b, c_a, c_b)
	fig.show()
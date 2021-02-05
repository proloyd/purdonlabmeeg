import numpy as np
from scipy.spatial.distance import pdist, is_valid_y, squareform


class _GPKernel:
	__func_name__ = None
	def __init__(self, X=None, metric='euclidean', dist=None, fixed_params=None):
		if X is None and dist is None:
			raise ValueError(f'Both X and dist cannot be None')
		elif dist is None:
			self._dist = pdist(X, metric)
			self._metric = metric
			self.ndim = X.shape[0]
		elif X is None:
			self._dist = dist.copy()
			self._metric = 'Unknown'
			self.ndim = dist.shape[0]
		self._fixed_params = fixed_params
			
	def __repr__(self, ):
		return f'GPKernel(func={self.__func_name__}) on {self.ndim} points with {self._fixed_params} fixed params)'
	
	def __func__(self, params):
		raise NotImplementedError
	
	def __diff_func__(self, params):
		raise NotImplementedError
	
	def cov(self, **params):
		all_params = self._fixed_params.copy()
		all_params.update(params)
		return self.__func__(**all_params)
	
	def diff_cov(self, **params):
		all_params = self._fixed_params.copy()
		all_params.update(params)
		return self.__diff_func__(**all_params)
	

class SquaredExponentailGPKernel(_GPKernel):
	__func_name__ = 'Squared Error'
	
	def __func__(self, theta, sigma_s):
		theta = np.squeeze(theta)
		temp = self._dist ** 2 / np.exp(2*theta)
		k = np.exp(- temp / 2)
		k *=  sigma_s ** 2 
		k = squareform(k) if is_valid_y(k) else k
		# c.flat[::self.ndim+1] += 1 - sigma_s**2
		k.flat[::self.ndim+1] = 1
		return k
	
	def __diff_func__(self, theta, sigma_s):
		theta = np.squeeze(theta)
		temp = self._dist ** 2 / np.exp(2 * theta)
		diff1 = temp * np.exp(- temp / 2)
		diff1 *= sigma_s**2
		diff2 = np.zeros_like(diff1)
		return [squareform(diff) if is_valid_y(diff) else diff for diff in (diff1, diff2)]

	
	
class MaternGPKernel(_GPKernel):
	__func_name__ = 'Matern (eta=3/2)'
	
	def __func__(self, theta, sigma_s):
		temp = self._dist / np.exp(theta)
		temp *= np.sqrt(3)
		k = (1 + temp) * np.exp(-temp)
		k *= sigma_s**2
		k = squareform(k) if is_valid_y(k) else k
		k.flat[::self.ndim+1] = 1
		return k

	def __diff_func__(self, theta, sigma_s):
		temp = self._dist / np.exp(theta)
		temp *= np.sqrt(3)
		diff_k = temp * temp
		diff_k *= np.exp(-temp)
		diff_k *= sigma_s**2
		diff_k = squareform(diff_k) if is_valid_y(diff_k) else diff_k
		return diff_k
	

class MaternGPKernel2(_GPKernel):
	__func_name__ = 'Matern (eta=5/2)'
	
	def __func__(self, theta, sigma_s):
		temp = self._dist / np.exp(theta)
		temp *= np.sqrt(5)
		k = (1 + temp + temp * temp / 3) * np.exp(-temp)
		k *= sigma_s**2
		k = squareform(k) if is_valid_y(k) else k
		k.flat[::self.ndim+1] = 1
		return k

	def __diff_func__(self, theta, sigma_s):
		temp = self._dist / np.exp(theta)
		temp *= np.sqrt(5)
		diff_k = temp * temp
		diff_k *= (temp + 1) / 3
		diff_k *= np.exp(-temp)
		diff_k *= sigma_s**2
		diff_k = squareform(diff_k) if is_valid_y(diff_k) else diff_k
		return diff_k
	

class GammaExpGPKernel(_GPKernel):
	__func_name__ = 'gamma Exponential'
	
	def __func__(self, theta, sigma_s, gamma):
		temp = self._dist / np.exp(theta)
		temp **= gamma
		k = np.exp(-temp)
		k *= sigma_s**2
		k = squareform(k) if is_valid_y(k) else k
		k.flat[::self.ndim+1] = 1
		return k

	def __diff_func__(self, theta, sigma_s, gamma):
		temp = self._dist / np.exp(theta)
		temp **= gamma
		diff_k = np.exp(-temp) * temp
		diff_k *= gamma
		diff_k *= sigma_s**2
		diff_k = squareform(diff_k) if is_valid_y(diff_k) else diff_k
		return diff_k

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	from _bb_utils import stab_bb
	x = np.random.rand(1000) * 100
	x = np.sort(x)
	# x = np.arange(100)
	x.shape= (1000, 1)
	y = 1 / (1 + np.exp(-10 * (x - 49))) - 1 / (1 + np.exp(-10 * (x - 51)))
	sigma = 0.1
	y_noisy = y + sigma * np.random.randn(*y.shape)
	
	theta = 5
	sigma = 0.10	
	def f(theta, gpkernel):
		k = gpkernel.cov(theta=theta)
		e, v = np.linalg.eigh(k)
		# e[e<0] = 0
		temp = v.T.dot(y_noisy)
		val = np.log(e + sigma ** 2).sum()
		# import ipdb; ipdb.set_trace()
		val += ((temp * temp) / (e + sigma ** 2)[:, None]).sum()
		out = val / 2
		out = np.asanyarray(out)
		out.shape = (1, )
		return out

	def df(theta, gpkernel):
		k = gpkernel.cov(theta=theta)
		dk = gpkernel.diff_cov(theta=theta)
		e, v = np.linalg.eigh(k)
		# e[e<0] = 0
		kinv = (v / (e + sigma ** 2)[None, :]).dot(v.T)
		k.flat[::e.shape[0]+1] += sigma ** 2
		try:
			assert np.allclose(kinv.dot(k), np.eye(k.shape[0]))
		except AssertionError:
			import ipdb; ipdb.set_trace()
		temp = kinv.dot(y_noisy)
		diffk = kinv - temp.dot(temp.T)
		out =  (diffk * dk).sum() / 2
		out = np.asanyarray(out)
		out.shape = (1, )
		return out		
	
	def check_grad(x0, gpkernel):
		Fx = lambda x: f(x, gpkernel)
		gradFx = lambda x: df(x, gpkernel)
		for delta in reversed(np.logspace(-13, -2, 10)):
			true_diff = Fx(x0+delta) - Fx(x0)
			estimated_diff = gradFx(x0) * delta
			print(true_diff, estimated_diff, (true_diff - estimated_diff)/delta)
		
	mgpkernel = MaternGPKernel(X=x, fixed_params={'sigma_s':np.sqrt(1)})
	fx = [f(i, mgpkernel) for i in np.arange(np.log(5), np.log(100), 10)]
	x0 = np.arange(np.log(5), np.log(100), 10)[np.asarray(fx).argmin()]
	besttheta, res_m = stab_bb(x0 = x0, costFn=lambda x: f(x, mgpkernel),
						gradFn=lambda x: df(x, mgpkernel), tol=1e-4)
	print(f'Best theta: {np.exp(besttheta)}')
	k = mgpkernel.cov(theta=besttheta)
	temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
	y_est_m = k.dot(temp)
	
	mgpkernel2 = MaternGPKernel2(X=x, fixed_params={'sigma_s':np.sqrt(1)})
	fx = [f(i, mgpkernel2) for i in np.arange(np.log(5), np.log(100), 10)]
	x0 = np.arange(np.log(5), np.log(100), 10)[np.asarray(fx).argmin()]
	besttheta, res_m = stab_bb(x0 = x0, costFn=lambda x: f(x, mgpkernel2),
						gradFn=lambda x: df(x, mgpkernel2), tol=1e-4)
	print(f'Best theta: {np.exp(besttheta)}')
	k = mgpkernel2.cov(theta=besttheta)
	temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
	y_est_m2 = k.dot(temp)
	
	
	segpkernel = SquaredExponentailGPKernel(X=x, fixed_params={'sigma_s':np.sqrt(0.999)})
	fx = [f(i, segpkernel) for i in np.arange(np.log(5), np.log(100), 10)]
	besttheta, res_se = stab_bb(x0 =x0, costFn=lambda x: f(x, segpkernel),
						gradFn=lambda x: df(x, segpkernel), tol=1e-4,)
	print(f'Best theta: {np.exp(besttheta)}')
	k = segpkernel.cov(theta=besttheta)
	temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
	y_est_se = k.dot(temp)
	
	gegpkernel = GammaExpGPKernel(X=x, fixed_params={'gamma':1.5, 'sigma_s':np.sqrt(1)})
	fx = [f(i, gegpkernel) for i in np.arange(np.log(5), np.log(100), 10)]
	besttheta, res_se = stab_bb(x0 =x0, costFn=lambda x: f(x, gegpkernel),
						gradFn=lambda x: df(x, gegpkernel), tol=1e-4,)
	print(f'Best theta: {np.exp(besttheta)}')
	k = gegpkernel.cov(theta=besttheta)
	temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
	y_est_ge = k.dot(temp)	
	
	fig, ax = plt.subplots(1)
	ax.scatter(x, y_noisy)
	ax.plot(x, y, label='True')
	ax.plot(x, y_est_m, color='red', label='Matern GP prior')
	ax.plot(x, y_est_m2, color='cyan', label='Matern GP prior-2')
	ax.plot(x, y_est_se, color='green', label='Squared Exponential GP prior')
	ax.plot(x, y_est_ge, color='magenta', label='Gamma-Exponential GP prior')
	ax.legend()
	fig.show()
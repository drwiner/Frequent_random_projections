# (C) David R. Winer, 2017-04-05
# edit of "(C) Mathieu Blondel, November 2013" but uses 'S' reconstruction
# License: BSD 3 clause
#

import numpy as np
# from scipy.linalg import svd
import math
from clockdeco import clock

def errnorm(A, B):
	AA = np.dot(A.T, A)
	BB = np.dot(B.T, B)
	return np.linalg.norm(AA - BB, 2)

# @clock
def frequent_directions(A, ell, verbose=False):
	"""
	Return the sketch of matrix A.

	Parameters
	----------
	A : array-like, shape = [n_samples, n_features]
		Input matrix.

	ell : int
		Number of rows in the matrix sketch.

	Returns
	-------
	B : array-like, shape = [ell, n_features]

	Reference
	---------
	Edo Liberty, "Simple and Deterministic Matrix Sketching", ACM SIGKDD, 2013.
	"""
	if ell % 2 == 1:
		raise ValueError("ell must be an even number.")

	n_samples, n_features = A.shape

	if ell > n_samples:
		raise ValueError("ell must be less than n_samples.")

	# if ell > n_features:
	# 	raise ValueError("ell must be less than n_features.")

	B = np.zeros((ell, n_features), dtype=np.float64)
	ind = np.arange(ell)

	for i in range(n_samples):
		zero_rows = ind[np.sum(np.abs(B) <= 1e-12, axis=1) == n_features]
		if len(zero_rows) >= 1:
			B[zero_rows[0]] = A[i]
		else:
			U, sigma, V = np.linalg.svd(B, full_matrices=0, compute_uv=1)

			# S - reconstruction
			S = np.zeros((len(U),len(sigma)), dtype=complex)
			S[:len(sigma), :len(sigma)] = np.diag(sigma)
			delta = S[math.floor(ell / 2 - 1)] ** 2
			S[:len(sigma), :len(sigma)] = np.sqrt(np.maximum(np.diag(sigma) ** 2 - delta, 0))

			B = np.dot(S, V)

		if verbose:
			errnorm(A, B)

	return B, error


def fnorm(Q):
	return np.linalg.norm(Q, 'fro')


import random
def N_01_divL(L):
	return random.random() / math.sqrt(L)

@clock
def make_S(L, N):
	# make an L by N matrix whose values are all N_01_divL
	z = np.random.random((L, N))
	for x in np.nditer(z, op_flags=['readwrite']):
		x[...] = np.random.normal(0, 1) / math.sqrt(L)
	return z

if __name__ == '__main__':
	# np.random.seed(0)
	# A = np.random.random((100, 20))
	A = np.loadtxt('A.dat')
	two_a = False

	U, sigma, V = np.linalg.svd(A, full_matrices=1, compute_uv=1)
	k = 2
	U2 = U[:, 0:k]
	S2 = np.diag(sigma[0:k])
	# S[:len(sigma), :len(sigma)] = np.diag(sigma)
	V2 = V.T[:, 0:k]
	A2 = np.dot(U2, np.dot(S2, V2.T))

	one_to_beat = (fnorm(A - A2) ** 2) / 10

	if two_a:


		# one_to_beat = (fnorm(A)**2)/10
		# one_to_beat = 20000
		# error = 1000
		print(one_to_beat)
		for z in range(2, 90, 2):
			# z = 36
			B, error = frequent_directions(A, z, verbose=True)
			if error < one_to_beat:
				print(z)
				break
		print(error)
		print('A-A2 fro')
		print((fnorm(A-A2)**2)/10)
		print('A fro')
		print((fnorm(A)**2)/10)

	# L = 6
	one_to_beat = one_to_beat = (fnorm(A)**2)/10
	for L in range(60, 10000, 2):
		print(L)
		S_200 = [make_S(L, len(A)) for i in range(200)]
		total = 0
		for S_L in S_200:
			B = np.dot(S_L, A)
			error = errnorm(A, B)
			total += error
		print(total/200)
		if total / 200 <= one_to_beat:
			print(total/200)
			print(L)
			print('beat')
			break
		# print(error)
		# if error < one_to_beat:
		# 	print(error)
		# 	print(L)
		# 	break

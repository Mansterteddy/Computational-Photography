import numpy as np

a = np.random.randn(9, 6) + 1j * np.random.randn(9, 6)
print a 
#SVD
U, s, V = np.linalg.svd(a, full_matrices = True)

print U.shape, V.shape, s.shape

#Reconstruction based on full SVD
S = np.zeros((9, 6), dtype = complex)
S[:6, :6] = np.diag(s)
print np.allclose(a, np.dot(U, np.dot(S, V)))

#Reconstruction based on reduced SVD
U, s, V = np.linalg.svd(a, full_matrices = False)
print U.shape, V.shape, s.shape
S = np.diag(s)
print np.allclose(a, np.dot(U, np.dot(S, V)))

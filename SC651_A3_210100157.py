import numpy as np
from scipy.linalg import polar


vi = np.array([(0.63, 0.45, 0.87) ,
(0.03, 0.40, 0.86) ,
(0.86, 0.17, 0.16) ,
(0.29, 0.79, 0.04) ,
(0.60, 0.38, 0.15) ,
(0.99, 0.43, 0.76) ])

vi = vi.T/np.linalg.norm(vi, axis=1)

wi = np.array([(0.17, 0.03, 0.50),
(0.26, 0.37, 0.15),
(0.78, 0.65, 0.67),
(0.83, 0.96, 0.82),
(0.69, 0.72, 0.04),
(0.64, 0.77, 0.34)])


wi = wi.T/np.linalg.norm(wi, axis=1)


######### WAHBA
M = []

A = np.matmul(wi, vi.T)

U, P = polar(A)

N, D, _ = np.linalg.svd(P)
S = np.diag(D)
l = len(S)
X_ = np.identity(l)
X_[l-1][l-1] = -1
M = np.matmul(U,np.matmul(N.T, np.matmul(X_.T, N)))
print("M Wahba \n",M)
############ 

########### TRIAD first 2 vectors
vi_t = vi.T[0:2]
wi_t = wi.T[0:2]

qr = vi_t[0]
rr = np.cross(qr, vi_t[1])/np.linalg.norm(np.cross(qr, vi_t[1]))
sr = np.cross(qr, rr)

Mr = np.array([qr,
               rr,
               sr]).T

qb = wi_t[0]
rb = np.cross(qb, wi_t[1])/np.linalg.norm(np.cross(qb, wi_t[1]))
sb = np.cross(qb, rb)

Mb = np.array([qb,
               rb,
               sb]).T

A = Mb @ np.linalg.inv(Mr)

print("A TRIAD \n", A)

############# QUEST

# ai = np.abs(np.random.normal(0, 1, len(vi.T)))
ai = np.ones(len(vi.T))
ai = np.array([ai/np.linalg.norm(ai, ord=1)])

B =   wi @ (vi.T * ai.T)

S = B + B.T
sig = np.trace(B)
Z = ai @ np.cross(wi.T, vi.T)

K = np.zeros((4,4))
K[0:3, 0:3] = S - sig*np.identity(len(S))
K[0:3, 3] = Z.T[:,0]
K[3, 0:3] = Z
K[3,3] = sig

val, vec = np.linalg.eig(K)

q_bar = vec[np.argmax(val)]

Q = np.array([q_bar[0:3]]).T
q = q_bar[3]

def cross(vec):
    return np.array([[0, vec[2], -vec[1]],
                     [-vec[2], 0, vec[0]],
                     [vec[1], -vec[0], 0]])
    
A_Quest = (q**2 - np.dot(Q.T,Q)[0][0]) * np.identity(3)  +   2 * Q @ Q.T   +   2 * q * cross(Q.T[0])
print("A QUEST \n",A_Quest)

print()
ew = 0
et = 0
eq = 0
E = [(0,0,0)]
for i in range(len(vi.T)):
    ew += np.linalg.norm(wi.T[i] - M @ vi.T[i])
    et += np.linalg.norm(wi.T[i] - A @ vi.T[i])
    eq += np.linalg.norm(wi.T[i] - A_Quest @ vi.T[i])

    print("Error progression Wahba, Triad, QUEST : ", ew, et, eq)
    E.append((ew, et, eq))

import matplotlib.pyplot as plt
X = np.linspace(0, len(vi.T)+1, len(vi.T)+1)
plt.plot(X, E)
plt.title("Error progression with number of observations")
plt.ylabel("Error")
plt.xlabel("Number of observations")
plt.legend(("Wahba", "TRIAD", "QUEST"))
plt.show()
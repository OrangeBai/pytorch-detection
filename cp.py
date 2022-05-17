import cvxpy as cp

# import cvxpy as cp
import numpy as np


#
# # Generate a random SDP.
# n = 3
# p = 3
# np.random.seed(1)
# C = np.random.randn(n, n)
# A = []
# b = []
# for i in range(n):
#     A.append(cp.Variable())
#     b.append(np.random.randn())
# # Define and solve the CVXPY problem.
# # Create a symmetric matrix variable.
# rho = cp.Variable(1)
# mat = cp.diag(cp.Variable(n))
# eye = np.eye(n)
# X = np.random.randn(n, n)
#
# constraints = [mat @ X - rho * eye >> 0]
# constraints += [
#     mat[i, i] <= b[i] for i in range(n)
# ]
# constraints += [
#     mat[i, i] >= b[i] - 0.5 for i in range(n)
# ]
# prob = cp.Problem(cp.Maximize(rho),
#                   constraints)
# prob.solve()
#
# print("The optimal value is", prob.value)
# print("A solution X is")
# print(rho.value)


def solve_SDP(weight, lb, up, max=True):
    num = weight.shape[0]
    vec = cp.Variable(num, nonneg=True)
    mat = cp.diag(vec)

    rho = cp.Variable(1, nonneg=True)
    eye = np.eye(weight.shape[1])
    constraints = []

    constraints += [vec <= up]
    constraints += [vec >= lb]

    if max:
        constraints += [weight.T @ mat @ weight - rho * eye << 0]

        prob = cp.Problem(cp.Minimize(rho), constraints)
    else:
        constraints += [mat @ weight - rho * eye >> 0]

        prob = cp.Problem(cp.Maximize(rho), constraints)
    prob.solve(verbose=True)
    print("The optimal value is", prob.value)
    print("A solution rho is")
    print(rho.value)
    return prob.value


if __name__ == '__main__':
    weight = np.random.randn(500, 500)
    low = np.random.randn(500)
    up = low + 0.5
    solve_SDP(weight, low, up)
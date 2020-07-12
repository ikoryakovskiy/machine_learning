"""
MaxEnt: Maximum entropy
http://bjlkeng.github.io/posts/maximum-entropy-distributions/

A die has been tossed a very large number N of times, and we are told that the average number
of spots per toss was not 3.5, as we might expect from an honest die, but 4.5. Translate this
information into a probability assignment 𝑝𝑛,𝑛=1,2,…,6, for the 𝑛-th face to come up on the next toss.
"""

from numpy import exp
from scipy.optimize import newton

a, b, B = 1, 6, 4.5

# Equation 15
def z(lamb):
    return 1. / sum(exp(-k*lamb) for k in range(a, b + 1))

# Equation 16
def f(lamb, B=B):
    y = sum(k * exp(-k*lamb) for k in range(a, b + 1))
    return y * z(lamb) - B

# Equation 17
def p(k, lamb):
    return z(lamb) * exp(-k * lamb)

lamb = newton(f, x0=0.5)
print("Lambda = %.4f" % lamb)
for k in range(a, b + 1):
    print("p_%d = %.4f" % (k, p(k, lamb)))

# Output:
#   Lambda = -0.3710
#   p_1 = 0.0544
#   p_2 = 0.0788
#   p_3 = 0.1142
#   p_4 = 0.1654
#   p_5 = 0.2398
#   p_6 = 0.3475
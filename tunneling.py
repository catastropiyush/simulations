import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import root_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import expm
from IPython.display import HTML

# Define the functions for the potential barrier and transmission probability
def rectangular_potential_barrier(x, V0, a):
    """Rectangular potential barrier of height V0 and width a."""
    return np.where((0 <= x) & (x < a), V0, 0.0)

def transmission_probability(E, V0, a):
    """Transmission probability through a rectangular potential barrier."""
    k = (V0 * np.sinh(a * np.sqrt(2 * (V0 - E))))**2 / (4 * E * (V0 - E))
    return 1 / (1 + k)

# Gaussian wave packet
def gaussian_wavepacket(x, x0, sigma0, p0):
    norm = (2 * np.pi * sigma0**2)**(-0.25)
    return norm * np.exp(-(x - x0)**2 / (4 * sigma0**2)) * np.exp(1j * p0 * x)

# Hamiltonian construction
def hamiltonian(N, dx, V):
    laplacian = -2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    laplacian /= dx**2
    H = -0.5 * laplacian + np.diag(V)
    return H

# Simulation function using matrix exponential for time evolution
def simulate(psi0, H, dt):
    U = expm(-1j * H * dt)  # Time evolution operator
    psi = psi0
    while True:
        psi = U @ psi
        yield psi

# Parameters
N = 1280
x, dx = np.linspace(-80, +80, N, endpoint=False, retstep=True)

T, r = 0.20, 3/4
k1 = root_scalar(lambda a: transmission_probability(0.5*r, 0.5, a) - T,
                 bracket=(0.0, 10.0)).root

a = 1.25
V0 = ((k1 / a)**2) / 2
E = r * V0

x0, sigma0, p0 = -48.0, 3.0, np.sqrt(2*E)
psi0 = gaussian_wavepacket(x, x0=x0, sigma0=sigma0, p0=p0)

V = rectangular_potential_barrier(x, V0, a)
H = hamiltonian(N, dx, V=V)
sim = simulate(psi0, H, dt=0.2)

# Setup for animation
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi0)**2)
ax.set_xlim(-80, 80)
ax.set_ylim(0, np.max(np.abs(psi0)**2))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$|\psi(x)|^2$')

barrier_start, barrier_end = 0, a
ax.axvspan(barrier_start, barrier_end, color='red', alpha=0.3, label='Potential Barrier')


def update(frame):
    psi = next(sim)
    line.set_ydata(np.abs(psi)**2)
    return line,

# Create and display the animation
ani = FuncAnimation(fig, update, frames=500, interval=50, blit=True)
HTML(ani.to_jshtml())

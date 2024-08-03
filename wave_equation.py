import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from ipywidgets import interact, FloatSlider

# Parameters
Lx, Ly = 1, 1  # Domain size
Nx, Ny = 50, 50  # Number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
T = 10  # Total simulation time
dt = 0.002  # Time step

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition for displacement
# def initial_condition(X, Y):
#     return np.exp(-10*((X-0.25)**2 + (Y-0.25)**2))  # Gaussian pulse
def initial_condition(X, Y):
    k_x, k_y = 5, 5  # Wave numbers
    return np.sin(k_x*X + k_y*Y)

# Initial condition for velocity 
def initial_velocity(X, Y):
    return np.zeros_like(X)

from numpy.fft import fft2, ifft2

# def solve_2d_wave_equation_spectral(u, u_prev, c, dx, dy, dt):
#     Nx, Ny = u.shape
#     kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
#     ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
#     KX, KY = np.meshgrid(kx, ky)
#     K_sq = KX**2 + KY**2
#     U = fft2(u)
#     U_prev = fft2(u_prev)
#     U_next = 2*U - U_prev - (c*dt)**2 * K_sq * U
#     return np.real(ifft2(U_next))

# Function to solve the 2D wave equation
def solve_2d_wave_equation(u, u_prev, c, dx, dy, dt):
    u_next = np.zeros_like(u)
    u_next[1:-1, 1:-1] = 2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (c*dt/dx)**2 * (
        u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]
    ) + (c*dt/dy)**2 * (
        u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]
    )
    return u_next
  
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(initial_condition(X, Y), cmap='coolwarm', animated=True, vmin=-1, vmax=1)
fig.colorbar(im, label='Displacement')
ax.set_title('2D Wave Equation')

# Global variables for the current and previous state
u = None
u_prev = None
c = None
# update animation
def update(frame):
    global u, u_prev
    for _ in range(10):  # Solve multiple time steps per frame for smoother animation
        u_next = solve_2d_wave_equation(u, u_prev, c, dx, dy, dt)
        u_prev = u
        u = u_next
    im.set_array(u)
    ax.set_title(f'2D Wave Equation (t = {frame * 10 * dt:.3f})')
    return [im]
  
def create_animation(wave_speed):
    global u, u_prev, c
    u = initial_condition(X, Y)
    u_prev = u - dt * initial_velocity(X, Y)
    c = wave_speed
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    anim.save('wave_animation.gif', writer='pillow')
    return HTML(anim.to_jshtml())

# Interactive widget for wave speed c
interact(create_animation,
         wave_speed=1.0)


from deepxde.backend import tf
import deepxde as dde
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go
import sympy as sp
import scipy.integrate as spi
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



sigma = 0.5
R = 5


def alcubierre_shape_function(r,R,sigma): 
    sigma = 0.5
    R = 5
    f = (((np.tanh(sigma*(r+R)))-(np.tanh(sigma*(r-R))))/(2*np.tanh(sigma*R)))
    return f


def alcubierre_shape_derivative(r, epsilon=1e-7):
    derivative = ((alcubierre_shape_function(r + epsilon,R,sigma)) - alcubierre_shape_function(r - epsilon,R,sigma)) / (2 * epsilon)
    return derivative


def fitness_parameter(r):
    return  (r**2)*(alcubierre_shape_derivative(r)**2)




def plot_alcubierre_york_time(rho, z, epsilon=1e-7):
    R = 5  
    sigma = 0.5  
    
    r = np.sqrt(rho**2 + z**2)
    derivative = (alcubierre_shape_function(r + epsilon, R, sigma) - alcubierre_shape_function(r - epsilon, R, sigma)) / (2 * epsilon)
    theta = (z / r) * derivative
    return theta


def plot_alcubierre_eulerian_violation(x, z, epsilon=1e-7):
    R = 5  # Given value of R
    sigma = 0.5  # Given value of sigma
    
    r = np.sqrt(x**2 + z**2)
    derivative = (alcubierre_shape_function(r + epsilon, R, sigma) - alcubierre_shape_function(r - epsilon, R, sigma)) / (2 * epsilon)
    t00 = r * (derivative**2)
    return t00



r_values = np.linspace(0, 10, 400) 
f_values = alcubierre_shape_function(r_values,R,sigma)
df_values = alcubierre_shape_derivative(r_values)
fitness_values = fitness_parameter(r_values)
rho_values = np.linspace(-10, 10, 200)
z_values = np.linspace(-10, 10, 200)
rho_mesh, z_mesh = np.meshgrid(rho_values, z_values)
theta_mesh = plot_alcubierre_york_time(rho_mesh, z_mesh)


# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(r_values, f_values, label='Alcubierre Shape Function, ' + 'R=' + str(R) + ', sigma=' + str(sigma))

plt.axvline(x=5, color='blue', linestyle='--', label='R = '+ str(R))
plt.title('Alcubierre Shape Function')
plt.xlabel('r')
plt.ylabel('f(r)')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 6))
plt.plot(r_values, df_values, label='Alcubierre Shape Function Derivative, ' + 'R=' + str(R) + ', sigma=' + str(sigma))

plt.axvline(x=5, color='blue', linestyle='--', label='R = '+ str(R))
plt.title('Alcubierre Shape Function Derivative')
plt.xlabel('r')
plt.ylabel('df(r)')
plt.legend()
plt.grid(True)
plt.show()



plt.figure(figsize=(8, 6))
plt.plot(r_values, fitness_values, label='Fitness Parameter, ' + 'R=' + str(R) + ', sigma=' + str(sigma))

plt.axvline(x=5, color='blue', linestyle='--', label='R = '+ str(R))
plt.title('Fitness Parameter')
plt.xlabel('r')
plt.ylabel('L(r)')
plt.legend()
plt.grid(True)
plt.show()

alcubierre_energy_requirement, _ = spi.quad(fitness_parameter, 0, 10)

print("Alcubierre Total Energy Requirement: "+ str(alcubierre_energy_requirement))




# Create a larger 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a galaxy-themed color map
galaxy_cmap = plt.get_cmap('plasma')

# Plot the surface with the galaxy-themed color map
surface = ax.plot_surface(rho_mesh, z_mesh, theta_mesh, cmap=galaxy_cmap)

# Set labels for the axes
ax.set_xlabel('rho')
ax.set_ylabel('z')
ax.set_zlabel('theta')

# Set a title for the plot
ax.set_title('Alcubierre York Time')

# Add color bar
cbar = fig.colorbar(surface, ax=ax, pad=0.1)
cbar.set_label('Theta')

# Show the plot
plt.show()



# Generate a grid of x and z values
x_vals = np.linspace(-10, 10, 100)
z_vals = np.linspace(-10, 10, 100)
X, Z = np.meshgrid(x_vals, z_vals)

# Calculate the t00 values for each combination of x and z
t00_vals = plot_alcubierre_eulerian_violation(X, Z)

# Create a density plot
plt.figure(figsize=(10, 8))
plt.contourf(X, Z, t00_vals, levels=20, cmap='viridis')
plt.colorbar(label='t00 values')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Density Plot of t00 with respect to x and z')
plt.show()
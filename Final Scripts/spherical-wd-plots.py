
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
import pickle
import os


sigma = 0.5
R = 5
alcubierre_energy_requirement = 4.496030169993188

def generate_evolution_model(domain, boundary, width, depth, rate, epochs):
    
    def pde(x, y):
        # y = (f)
        # x = (r,t)
        df_r = dde.grad.jacobian(y, x, i=0, j=0)
        return (x[:,0:1]*(df_r**2))**2

    
    model_frames = []

    main_domain = dde.geometry.Interval(0,10)
    geom = main_domain


    ic = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[0],0),
    )
    
    ic2 = dde.icbc.DirichletBC(
        geom,
        lambda x: 0,
        lambda x, on_boundary:  np.isclose(x[0],10),
    )

#     dic = dde.icbc.OperatorBC(
#     geom,
#     lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
#     lambda _, on_initial: on_initial,)


    data = dde.data.PDE(
        geom, pde, [ic, ic2], num_domain=domain, num_boundary=boundary)


    net = dde.nn.FNN([1] + [width] * depth + [1], "tanh", "Glorot normal")


    model = dde.Model(data, net)
    
    for i in range(10):
        model.compile("adam", lr=rate)
        model.train(iterations=int(epochs/10))
        model_frames.append(model)
        print(model_frames[i].predict([[1]]))
        print("Epoch Batch " + str(i) + " Completed")
    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    
    print("Training Complete!")
    
    for i in range(len(model_frames)):
        print (model_frames[i].predict([[1]]))
    print (model_frames)

    return model_frames


model = generate_evolution_model(100,100,100,2,0.01,10000)



os.mkdir("shape")
os.mkdir("derivative")
os.mkdir("fitness")
os.mkdir("york")
os.mkdir("violation")


f_values = []
df_values = []
fit_f_values = []

print(len(model))


for i in range(10):
    # Your existing code
    print(len(f_values))

    r_values = np.linspace(0, 10, 400)
    f_values.append(model[i].predict(r_values.reshape(-1, 1)).flatten())
    print(type(model[i].predict(r_values.reshape(-1, 1)).flatten()))
    print(f_values)
    print(len(f_values))

    # Create an interpolation function
    model_shape_function = interp1d(r_values, f_values[i], kind='linear', fill_value="extrapolate")


    def model_shape_derivative(r):
        epsilon = 1e-7
        derivative = ((model_shape_function(r + epsilon)) - model_shape_function(r - epsilon)) / (2 * epsilon)
        return derivative


    def model_fitness_parameter(r):
        return  (r**2)*(model_shape_derivative(r)**2)
    

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(r_values, f_values[i], label='Model Shape Function, ' + 'R=' + str(R) + ', sigma=' + str(sigma))
    plt.axvline(x=5, color='blue', linestyle='--', label='R = '+ str(R))
    plt.title('Model Shape Function')
    plt.xlabel('r')
    plt.ylabel('f(r)')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"shape/shape_{i}")



    # Generate a range of r values
    r_values = np.linspace(0, 10, 400)  # Adjust the range as needed
    df_values.append(model_shape_derivative(r_values))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(r_values, df_values[i], label='Model Shape Function Derivative, ' + 'R=' + str(R) + ', sigma=' + str(sigma))

    plt.axvline(x=5, color='blue', linestyle='--', label='R = '+ str(R))
    plt.title('Model Shape Function Derivative')
    plt.xlabel('r')
    plt.ylabel('df(r)')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"derivative/derivative_{i}")


    # Generate a range of r values
    r_values = np.linspace(0, 10, 400)  # Adjust the range as needed
    fit_f_values.append(model_fitness_parameter(r_values))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(r_values, fit_f_values[i], label='Fitness Parameter, ' + 'R=' + str(R) + ', sigma=' + str(sigma))

    plt.axvline(x=5, color='blue', linestyle='--', label='R = '+ str(R))
    plt.title('Fitness Parameter')
    plt.xlabel('r')
    plt.ylabel('f(r)')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"fitness/fitness_{i}")



    model_energy_requirement, _ = spi.quad(model_fitness_parameter, 0, 10)

    print("Model Total Energy Requirement: " + str(model_energy_requirement))
    print("Relative Percent Difference from Alcubierre: " + str((model_energy_requirement/alcubierre_energy_requirement)*100) + "%")



    def plot_alcubierre_york_time(rho, z, epsilon=1e-7):
        R = 5  # Given value of R
        sigma = 0.5  # Given value of sigma
        
        r = np.sqrt(rho**2 + z**2)
        derivative = (model_shape_function(r + epsilon) - model_shape_function(r - epsilon)) / (2 * epsilon)
        theta = (z / r) * derivative
        return theta

    # Define the range of rho and z values
    rho_vals = np.linspace(-2, 2, 200)
    z_vals = np.linspace(-2, 2, 200)
    rho_mesh, z_mesh = np.meshgrid(rho_vals, z_vals)

    # Calculate theta values for each combination of rho and z
    theta_mesh = plot_alcubierre_york_time(rho_mesh, z_mesh)

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
    ax.set_title('Model York Time')

    # Add color bar
    cbar = fig.colorbar(surface, ax=ax, pad=0.1)
    cbar.set_label('Theta')

    # Show the plot
    # plt.show()
    plt.savefig(f"york/york_{i}")



    def plot_alcubierre_eulerian_violation(x, z, epsilon=1e-7):
        R = 5  # Given value of R
        sigma = 0.5  # Given value of sigma
        
        r = np.sqrt(x**2 + z**2)
        derivative = (model_shape_function(r + epsilon) - model_shape_function(r - epsilon)) / (2 * epsilon)
        t00 = r * (derivative**2)
        return t00



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
    # plt.show()
    plt.savefig(f"violation/violation_{i}")



    

"""Backend: tensorflow.compat.v1"""

# Import tf if using backend tensorflow.compat.v1 or tensorflow
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
import os
import argparse



def save_description(model_name):
    description = input("Enter a short description for the model: ")
    description_filename = os.path.join(model_name, "description.txt")
    with open(description_filename, "w") as file:
        file.write(description)
    print("Description saved:", description_filename)


def predict_shape_function(domain, boundary, width, depth, rate, epochs):
    
    def pde(x, y):
        # y = (f)
        # x = (x,y,z)
        # x = x[:,0:1]
        # y = x[:,1:2]
        # z = x[:,2:]
        df_x = dde.grad.jacobian(y, x, i=0, j=0)
        df_y = dde.grad.jacobian(y, x, i=0, j=1)
        df_z = dde.grad.jacobian(y, x, i=0, j=2)
        first_factor = (x[:,0:1]**2 + x[:,1:2]**2)/((x[:,0:1]**2 + x[:,1:2]**2 + x[:,2:3]**2)**(3/2))
        squared_terms = ((x[:,0:1]**2)*(df_x)) + ((x[:,1:2]**2)*(df_y)) + ((x[:,2:]**2)*(df_z))
        cross_terms =  (x[:,0:1] * x[:,1:2] * df_x * df_y)+(x[:,0:1] * x[:,2:] * df_x * df_z)+(x[:,1:2] * x[:,2:] * df_y * df_z)
        return (squared_terms + (2*cross_terms)) 

    

    geom = dde.geometry.geometry_3d.Cuboid([-10,-10,-10],[10,10,10])

    
    passenger = dde.geometry.geometry_3d.Cuboid([-2,-2,-2],[2,2,2])


    geom = geom - passenger

    
    def boundary_inner(x, on_boundary):
        return on_boundary and passenger.on_boundary(x)
    
    
    
    def boundary_outer(x, on_boundary):
        return on_boundary and geom.on_boundary(x)
    
    
    
    asymptotic_bc = dde.icbc.DirichletBC(geom, 
                                        lambda x: 0, 
                                        boundary_outer)
    
    
    passenger_bc = dde.icbc.DirichletBC(geom, 
                                        lambda x: 1.5, 
                                        boundary_inner)
    
    
    passenger_bc_2 = dde.icbc.NeumannBC(geom, 
                                        lambda x: 0, 
                                        boundary_inner)
    
    
    data = dde.data.PDE(
        geom, pde, [asymptotic_bc,
                   passenger_bc,
                   passenger_bc_2], 
                   num_domain=domain,
                   num_boundary=boundary)


    net = dde.nn.FNN([3] + [width] * depth + [1], "tanh", "Glorot normal")


    model = dde.Model(data, net)
    model.compile("adam", lr=rate)
    losshistory, train_state = model.train(iterations=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    
    
    print("Training Complete!")
    

    



    return model



def main(args):

    domain = args.domain
    boundary = args.boundary
    width = args.width
    depth = args.depth
    rate = args.rate
    epochs = args.epochs
    model_name = args.model_name
    
    
    os.mkdir(model_name)   
    save_description(model_name)



    # Generate the evolution model
    model = predict_shape_function(domain, boundary, width, depth, rate, epochs)




    # Generate sample data points
    x_values = np.linspace(-10, 10, 30)
    y_values = np.linspace(-10, 10, 30)
    z_values = np.linspace(-10, 10, 30)
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)

    # Evaluate the model on the data points
    input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    predictions = model.predict(input_data)
    predictions = predictions.reshape(X.shape)


    # Save the data to binary files
    
    np.save(f'{model_name}/x.npy', x_values)
    np.save(f'{model_name}/y.npy', y_values)
    np.save(f'{model_name}/z.npy', z_values)
    np.save(f'{model_name}/pred.npy', predictions)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save shape function prediction")
    parser.add_argument("--domain", type=int, required=True, help="Number of domain points")
    parser.add_argument("--boundary", type=int, required=True, help="Number of boundary points")
    parser.add_argument("--width", type=int, required=True, help="Width of the neural network layers")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the neural network")
    parser.add_argument("--rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the saved model")


    args = parser.parse_args()
    main(args)




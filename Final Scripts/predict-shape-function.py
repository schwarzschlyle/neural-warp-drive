
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
from tqdm import tqdm
import imageio
import io
from PIL import Image



def save_description(model_name, domain, boundary, width, depth, rate, epochs):
    description = input("Enter a short description for the model: ")
    description_filename = os.path.join(model_name, "description.txt")
    with open(description_filename, "w") as file:
        file.write(description)
        file.write("Model Name: " + str(model_name))
        file.write("Domain: " + str(domain))
        file.write("Boundary: " + str(boundary))
        file.write("Width: " + str(width))
        file.write("Depth:  "+ str(depth))
        file.write("Rate: " + str(rate))
        file.write("Epochs: " + str(epochs))
        
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
        first_factor = (x[:,0:1]**2 + x[:,1:2]**2)/((x[:,0:1]**2 + x[:,1:2]**2 + x[:,2:3]**2)**(2))
        squared_terms = ((x[:,0:1]**2)*(df_x)**2) + ((x[:,1:2]**2)*(df_y)**2) + ((x[:,2:3]**2)*(df_z)**2)
        cross_terms =  (x[:,0:1] * x[:,1:2] * df_x * df_y)+(x[:,0:1] * x[:,2:3] * df_x * df_z)+(x[:,1:2] * x[:,2:3] * df_y * df_z)
        return first_factor*(squared_terms + (2*cross_terms)) 

    

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
                                        lambda x: 2, 
                                        boundary_inner)
    
    
    # passenger_bc_2 = dde.icbc.NeumannBC(geom, 
    #                                     lambda x: 0, 
    #                                     boundary_inner)
    
    
    data = dde.data.PDE(
        geom, pde, [asymptotic_bc,
                   passenger_bc], 
                   num_domain=domain,
                   num_boundary=boundary)


    net = dde.nn.FNN([3] + [width] * depth + [1], "tanh", "Glorot normal")


    model = dde.Model(data, net)
    model.compile("adam", lr=rate)
    losshistory, train_state = model.train(iterations=epochs)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    
    
    print("Training Complete!")
    

    



    return model




def predict_shape_function_with_history(domain, boundary, width, depth, rate, epochs, model_name):
    
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
    checker = dde.callbacks.ModelCheckpoint(
    f"{model_name}/temp_model_history/model", save_better_only=False, period= (epochs/100))
    model.compile("adam", lr=rate)
    losshistory, train_state = model.train(iterations=epochs, callbacks = [checker])
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    
    
    # print("Training Complete!")
    

    



    return model




def main(args):

    domain = args.domain
    boundary = args.boundary
    width = args.width
    depth = args.depth
    rate = args.rate
    epochs = args.epochs
    model_name = args.model_name
    history = args.history
    
    os.mkdir(model_name)   
    save_description(model_name, domain, boundary, width, depth, rate, epochs)


    if history == False:
       
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

        # Calculate derivatives using numerical differentiation
        dx = x_values[1] - x_values[0]
        dy = y_values[1] - y_values[0]
        dz = z_values[1] - z_values[0]

        dfdx = np.gradient(predictions, dx, axis=0)
        dfdy = np.gradient(predictions, dy, axis=1)
        dfdz = np.gradient(predictions, dz, axis=2)

        # Save the data to binary files
    
        np.save(f'{model_name}/x.npy', x_values)
        np.save(f'{model_name}/y.npy', y_values)
        np.save(f'{model_name}/z.npy', z_values)
        np.save(f'{model_name}/pred.npy', predictions)
        np.save(f'{model_name}/dfxpred.npy', dfdx)
        np.save(f'{model_name}/dfypred.npy', dfdy)
        np.save(f'{model_name}/dfzpred.npy', dfdz)

        
    if history == True:

        model = predict_shape_function_with_history(domain, boundary, width, depth, rate, epochs, model_name)

     
        # # Generate sample data points
        # x_values = np.linspace(-10, 10, 30)
        # y_values = np.linspace(-10, 10, 30)
        # z_values = np.linspace(-10, 10, 30)
        # X, Y, Z = np.meshgrid(x_values, y_values, z_values)

        # # Evaluate the model on the data points
        # input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        # predictions = model.predict(input_data)
        # predictions = predictions.reshape(X.shape)

        # # Save the data to binary files
        
        # np.save(f'{model_name}/x.npy', x_values)
        # np.save(f'{model_name}/y.npy', y_values)
        # np.save(f'{model_name}/z.npy', z_values)
        # np.save(f'{model_name}/pred.npy', predictions)

        # Generate sample data points
        x_values = np.linspace(-10, 10, 30)
        y_values = np.linspace(-10, 10, 30)
        z_values = np.linspace(-10, 10, 30)
        X, Y, Z = np.meshgrid(x_values, y_values, z_values)

        # Evaluate the model on the data points
        input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        predictions = model.predict(input_data)
        predictions = predictions.reshape(X.shape)

        # Calculate derivatives using numerical differentiation
        dx = x_values[1] - x_values[0]
        dy = y_values[1] - y_values[0]
        dz = z_values[1] - z_values[0]

        dfdx = np.gradient(predictions, dx, axis=0)
        dfdy = np.gradient(predictions, dy, axis=1)
        dfdz = np.gradient(predictions, dz, axis=2)

        # Save the data to binary files
    
        np.save(f'{model_name}/x.npy', x_values)
        np.save(f'{model_name}/y.npy', y_values)
        np.save(f'{model_name}/z.npy', z_values)
        np.save(f'{model_name}/pred.npy', predictions)
        np.save(f'{model_name}/dfxpred.npy', dfdx)
        np.save(f'{model_name}/dfypred.npy', dfdy)
        np.save(f'{model_name}/dfzpred.npy', dfdz)






        model_history = []

        

        for i in range (1,(int(epochs/100)+1)):
            model.restore(f"{model_name}/temp_model_history/model-{((i)*100)}.ckpt", verbose=1)
            model_history.append(model)  # Replace ? with the exact filename


        frames = []
        frames_xc = []
        frames_yc = []
        frames_zc = []
        frames_xs = []
        frames_ys = []
        frames_zs = []


        for i in range(len(model_history)):
            # Evaluate the model on the data points
            
            
            input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
            predictions = model.predict(input_data)
            predictions = predictions.reshape(X.shape)
            

            fig = go.Figure(data=[
                go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
                            marker=dict(size=3, colorscale='Inferno',
                                        color=predictions.ravel(),
                                        colorbar=dict(title='Normalized f(x,y,z) value'),
                                        cmin=np.min(predictions), cmax=np.max(predictions))),
            ])

            fig.update_layout(scene=dict(
                                xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z'),
                            title=f'Warp Drive Shape Function (Epoch = {(i+1)*100})')

            # Convert Plotly figure to bytes
            img_bytes = fig.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames.append(np.array(img_pil))

        
        
            # Define the slices for x=0, y=0, z=0
            x_slice = 0
            y_slice = 0
            z_slice = 0

            # Plot the 3D surface for x=0 slice
            fig_xc = go.Figure(data=[go.Contour(x=y_values, y=z_values, z=predictions[1, :, :])])
            fig_xc.update_layout(title=f'Contour slice for x=0 (Epoch = {(i+1)*100})')



            # Convert Plotly figure to bytes
            img_bytes = fig_xc.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames_xc.append(np.array(img_pil))


        
            # Plot the 3D surface for y=0 slice
            fig_yc = go.Figure(data=[go.Contour(x=x_values, y=z_values, z=predictions[:, y_slice, :])])
            fig_yc.update_layout(title=f'Contour slice for y=0 (Epoch = {(i+1)*100})')
    

            # Convert Plotly figure to bytes
            img_bytes = fig_yc.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames_yc.append(np.array(img_pil))


            
        
            # Plot the 3D surface for z=0 slice
            fig_zc = go.Figure(data=[go.Contour(x=x_values, y=y_values, z=predictions[:, :, z_slice])])
            fig_zc.update_layout(title=f'Contour slice for z=0 (Epoch = {(i+1)*100})')
        
            
            
            # Convert Plotly figure to bytes
            img_bytes = fig_zc.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames_zc.append(np.array(img_pil))

        
            

            
            
            

            # Plot the 3D surface for x=0 slice
            fig_xs = go.Figure(data=[go.Surface(x=y_values, y=z_values, z=predictions[1, :, :])])
            fig_xs.update_layout(title=f'Surface slice for x=0 (Epoch = {(i+1)*100})')
        


            # Convert Plotly figure to bytes
            img_bytes = fig_xs.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames_xs.append(np.array(img_pil))

            # Save the frames as a GIF

            


            # Plot the 3D surface for y=0 slice
            fig_ys = go.Figure(data=[go.Surface(x=x_values, y=z_values, z=predictions[:, y_slice, :])])
            fig_ys.update_layout(title=f'Surface slice for y=0 (Epoch = {(i+1)*100})')
        

            # Convert Plotly figure to bytes
            img_bytes = fig_ys.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames_ys.append(np.array(img_pil))

            # Save the frames as a GIF

            
            

            # Plot the 3D surface for z=0 slice
            fig_zs = go.Figure(data=[go.Surface(x=x_values, y=y_values, z=predictions[:, :, z_slice])])
            fig_zs.update_layout(title=f'Surface slice for z=0 (Epoch = {(i+1)*100})')
        
            
            # Convert Plotly figure to bytes
            img_bytes = fig_zs.to_image(format="png")
            
            # Convert bytes to PIL image
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Convert PIL image to numpy array and append to frames list
            frames_zs.append(np.array(img_pil))

            # Save the frames as a GIF


            
            imageio.mimsave(f'{model_name}/shape_function.gif', frames, duration=0.1)
            imageio.mimsave(f'{model_name}/xc.gif', frames_xc, duration=0.1)
            imageio.mimsave(f'{model_name}/yc.gif', frames_yc, duration=0.1)
            imageio.mimsave(f'{model_name}/zc.gif', frames_zc, duration=0.1)
            imageio.mimsave(f'{model_name}/xs.gif', frames_xs, duration=0.1)
            imageio.mimsave(f'{model_name}/ys.gif', frames_ys, duration=0.1)
            imageio.mimsave(f'{model_name}/zs.gif', frames_zs, duration=0.1)

        os.rmdir(f"/{model_name}/temp_model_history")

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save shape function prediction")
    parser.add_argument("--domain", type=int, required=True, help="Number of domain points")
    parser.add_argument("--boundary", type=int, required=True, help="Number of boundary points")
    parser.add_argument("--width", type=int, required=True, help="Width of the neural network layers")
    parser.add_argument("--depth", type=int, required=True, help="Depth of the neural network")
    parser.add_argument("--rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the saved model")
    parser.add_argument("--history", action="store_true", help="Would you like to save a GIF file of the learning process?")

    args = parser.parse_args()
    main(args)




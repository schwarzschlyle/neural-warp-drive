import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.graph_objects as go
import sympy as sp
import scipy.integrate as spi
from scipy.interpolate import interp1d
import os
import argparse
from plotly.io import write_image









def main(args):
    model_name = args.model_name
    

        
    # # Generate sample data points
    # x_values = np.load(f'{model_name}/x.npy')
    # y_values = np.load(f'{model_name}/y.npy')
    # z_values = np.load(f'{model_name}/z.npy')
    # X, Y, Z = np.meshgrid(x_values, y_values, z_values)

    # # Evaluate the model on the data points
    # input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    # predictions = np.load(f'{model_name}/pred.npy')

    # # Normalize the predictions
    # normalized_predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

    # # Calculate the 90th percentile value
    # percentile_10 = np.percentile(normalized_predictions, 10)

    # # Create a custom colorscale for shading
    # colorscale = [
    #     [0, 'rgb(255, 255, 255)'],
    #     [percentile_10, 'rgb(200, 200, 200)'],
    #     [1, 'rgb(0, 0, 0)']
    # ]

    # # Create the 3D density plot with modified color scale
    # fig = go.Figure(data=[
    #     go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
    #                 marker=dict(size=3, color=normalized_predictions.ravel(), colorbar=dict(title='Density'),
    #                             colorscale=colorscale)),
    # ])

    # fig.update_layout(scene=dict(
    #                     xaxis_title='X',
    #                     yaxis_title='Y',
    #                     zaxis_title='Z'),
    #                 title='Assymetric Warp Drive Shape Function')

    # fig.show()

   

    # Generate sample data points
    x_values = np.load(f'{model_name}/x.npy')
    y_values = np.load(f'{model_name}/y.npy')
    z_values = np.load(f'{model_name}/z.npy')
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)

    # Evaluate the model on the data points
    input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    predictions = np.load(f'{model_name}/pred.npy')

    # Create the 3D density plot with Plotly using the 'Viridis' color scale
    fig = go.Figure(data=[
        go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
                    marker=dict(size=3, colorscale = 'Inferno',
                    color=predictions.ravel(), colorbar=dict(title='Normalized f(x,y,z) value'))),
    ])

    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'),
                    title='Warp Drive Shape Function')

    fig.show()
    fig.write_json(f"{model_name}/shape.json")




    # Define the slices for x=0, y=0, z=0
    x_slice = 0
    y_slice = 0
    z_slice = 0

    # Plot the 3D surface for x=0 slice
    fig_x = go.Figure(data=[go.Contour(x=y_values, y=z_values, z=predictions[1, :, :])])
    fig_x.update_layout(title='Slice for x=0')
    fig_x.show()
    fig_x.write_json(f"{model_name}/xc.json")




    # Plot the 3D surface for y=0 slice
    fig_y = go.Figure(data=[go.Contour(x=x_values, y=z_values, z=predictions[:, y_slice, :])])
    fig_y.update_layout(title='Slice for y=0')
    fig_y.show()
    fig_y.write_json(f"{model_name}/yc.json")



    # Plot the 3D surface for z=0 slice
    fig_z = go.Figure(data=[go.Contour(x=x_values, y=y_values, z=predictions[:, :, z_slice])])
    fig_z.update_layout(title='Slice for z=0')
    fig_z.show()
    fig_z.write_json(f"{model_name}/zc.json")


    # Plot the 3D surface for x=0 slice
    fig_x = go.Figure(data=[go.Surface(x=y_values, y=z_values, z=predictions[1, :, :])])
    fig_x.update_layout(title='Slice for x=0')
    fig_x.show()
    fig_x.write_json(f"{model_name}/xs.json")




    # Plot the 3D surface for y=0 slice
    fig_y = go.Figure(data=[go.Surface(x=x_values, y=z_values, z=predictions[:, y_slice, :])])
    fig_y.update_layout(title='Slice for y=0')
    fig_y.show()
    fig_y.write_json(f"{model_name}/ys.json")


    # Plot the 3D surface for z=0 slice
    fig_z = go.Figure(data=[go.Surface(x=x_values, y=y_values, z=predictions[:, :, z_slice])])
    fig_z.update_layout(title='Slice for z=0')
    fig_z.show()
    fig_z.write_json(f"{model_name}/zs.json")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize shape function prediction")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the saved model")


    args = parser.parse_args()
    main(args)




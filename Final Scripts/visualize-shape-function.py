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


import plotly.io as pio


# Set the default paper and plot background colors to transparent
default_template = go.layout.Template()
default_template.layout.paper_bgcolor = 'rgba(0, 0, 0, 0.1)'
default_template.layout.plot_bgcolor = 'rgba(0, 0, 0, 0.1)'

# Create a rectangle for each text label with white background
rectangles = [
    go.layout.Shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=0, x1=1,
        y0=0, y1=1,
        fillcolor="white",
        opacity=0.7,  # Adjust the opacity as needed
        layer="below",
    )
]

# Update the layout with the rectangles
default_template.layout.shapes = rectangles

# Set font color to dark black
default_template.layout.font.color = 'rgb(0, 0, 0)'

pio.templates.default = default_template

def main(args):
    model_name = args.model_name


   

    # Generate sample data points
    x_values = np.load(f'{model_name}/x.npy')
    y_values = np.load(f'{model_name}/y.npy')
    z_values = np.load(f'{model_name}/z.npy')
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)

    # Evaluate the model on the data points
    input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    predictions = np.load(f'{model_name}/pred.npy')


    # Here are thy derivatives
    dfx = np.load(f'{model_name}/dfxpred.npy')
    dfy = np.load(f'{model_name}/dfypred.npy')
    dfz = np.load(f'{model_name}/dfzpred.npy')




    coordinate_factor = ((x_values**2)+(y_values**2))/(((x_values**2)+(y_values**2)+(z_values**2))**2)
    squared_derivatives_factor = (((x_values**2)*(dfx**2))+((y_values**2)*(dfy**2))+((z_values**2)*(dfz**2)))
    cross_derivatives_factor = (x_values*y_values*dfx*dfy) +(x_values*z_values*dfx*dfy)+(y_values*z_values*dfy*dfz)
    energy_density = coordinate_factor * (squared_derivatives_factor + (2*cross_derivatives_factor))
    total_energy = np.trapz(np.trapz(np.trapz(energy_density, z_values, axis=0), y_values, axis=0), x_values, axis=0)
    york = (z_values/(((x_values**2)+(y_values**2)+(z_values**2))**(3/2)))*((x_values*dfx)+(y_values*dfy)+(z_values*dfz))

    print("Integrated energy density:", total_energy)


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
    fig_x = go.Figure(data=[go.Contour(x=y_values, y=z_values, z=predictions[1, :, :])], )
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




    # Plotting Energy Density


      # Create the 3D density plot with Plotly using the 'Viridis' color scale
    efig = go.Figure(data=[
        go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
                    marker=dict(size=3, colorscale = 'Inferno',
                    color=energy_density.ravel(), colorbar=dict(title='Normalized f(x,y,z) value'))),
    ])

    efig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'),
                    title='Warp Drive Energy Density')

    efig.show()
    efig.write_json(f"{model_name}/eshape.json")




    # Define the slices for x=0, y=0, z=0
    x_slice = 0
    y_slice = 0
    z_slice = 0

    # Plot the 3D surface for x=0 slice
    efig_x = go.Figure(data=[go.Contour(x=y_values, y=z_values, z=energy_density[1, :, :])])
    efig_x.update_layout(title='Slice for x=0')
    efig_x.show()
    efig_x.write_json(f"{model_name}/exc.json")




    # Plot the 3D surface for y=0 slice
    efig_y = go.Figure(data=[go.Contour(x=x_values, y=z_values, z=energy_density[:, y_slice, :])])
    efig_y.update_layout(title='Slice for y=0')
    efig_y.show()
    efig_y.write_json(f"{model_name}/eyc.json")



    # Plot the 3D surface for z=0 slice
    efig_z = go.Figure(data=[go.Contour(x=x_values, y=y_values, z=energy_density[:, :, z_slice])])
    efig_z.update_layout(title='Slice for z=0')
    efig_z.show()
    efig_z.write_json(f"{model_name}/ezc.json")


    # Plot the 3D surface for x=0 slice
    efig_x = go.Figure(data=[go.Surface(x=y_values, y=z_values, z=energy_density[1, :, :])])
    efig_x.update_layout(title='Slice for x=0')
    efig_x.show()
    efig_x.write_json(f"{model_name}/exs.json")



    # Plot the 3D surface for y=0 slice
    efig_y = go.Figure(data=[go.Surface(x=x_values, y=z_values, z=energy_density[:, y_slice, :])])
    efig_y.update_layout(title='Slice for y=0')
    efig_y.show()
    efig_y.write_json(f"{model_name}/eys.json")


    # Plot the 3D surface for z=0 slice
    efig_z = go.Figure(data=[go.Surface(x=x_values, y=y_values, z=energy_density[:, :, z_slice])])
    efig_z.update_layout(title='Slice for z=0')
    efig_z.show()
    efig_z.write_json(f"{model_name}/ezs.json")




    # Plotting Energy Density


      # Create the 3D density plot with Plotly using the 'Viridis' color scale
    yfig = go.Figure(data=[
        go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
                    marker=dict(size=3, colorscale = 'Inferno',
                    color=york.ravel(), colorbar=dict(title='Normalized f(x,y,z) value'))),
    ])

    yfig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'),
                    title='York T')

    yfig.show()
    yfig.write_json(f"{model_name}/yshape.json")




    # Define the slices for x=0, y=0, z=0
    x_slice = 0
    y_slice = 0
    z_slice = 0

    # Plot the 3D surface for x=0 slice
    yfig_x = go.Figure(data=[go.Contour(x=y_values, y=z_values, z=york[1, :, :])])
    yfig_x.update_layout(title='Slice for x=0')
    yfig_x.show()
    yfig_x.write_json(f"{model_name}/yxc.json")




    # Plot the 3D surface for y=0 slice
    yfig_y = go.Figure(data=[go.Contour(x=x_values, y=z_values, z=york[:, y_slice, :])])
    yfig_y.update_layout(title='Slice for y=0')
    yfig_y.show()
    yfig_y.write_json(f"{model_name}/yyc.json")



    # Plot the 3D surface for z=0 slice
    yfig_z = go.Figure(data=[go.Contour(x=x_values, y=y_values, z=york[:, :, z_slice])])
    yfig_z.update_layout(title='Slice for z=0')
    yfig_z.show()
    yfig_z.write_json(f"{model_name}/yzc.json")


    # Plot the 3D surface for x=0 slice
    yfig_x = go.Figure(data=[go.Surface(x=y_values, y=z_values, z=york[1, :, :])])
    yfig_x.update_layout(title='Slice for x=0')
    yfig_x.show()
    yfig_x.write_json(f"{model_name}/yxs.json")



    # Plot the 3D surface for y=0 slice
    yfig_y = go.Figure(data=[go.Surface(x=x_values, y=z_values, z=york[:, y_slice, :])])
    yfig_y.update_layout(title='Slice for y=0')
    yfig_y.show()
    yfig_y.write_json(f"{model_name}/yys.json")


    # Plot the 3D surface for z=0 slice
    yfig_z = go.Figure(data=[go.Surface(x=x_values, y=y_values, z=york[:, :, z_slice])])
    yfig_z.update_layout(title='Slice for z=0')
    yfig_z.show()
    yfig_z.write_json(f"{model_name}/yzs.json")


     



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize shape function prediction")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the saved model")


    args = parser.parse_args()
    main(args)




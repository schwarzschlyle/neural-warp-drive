"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

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





def generate_evolution_model(domain, boundary, width, depth, rate, epochs):
    
    def pde(x, y):
        # y = (f)
        # x = (x,y,z)
        # x = x[:,0:1]
        # y = x[:,1:2]
        # z = x[:,2:]
        df_x = dde.grad.jacobian(y, x, i=0, j=0)
        df_y = dde.grad.jacobian(y, x, i=0, j=1)
        df_z = dde.grad.jacobian(y, x, i=0, j=2)
        first_factor = (x[:,0:1]**2 + x[:,1:2]**2)/(x[:,0:1]**2 + x[:,1:2]**2 + x[:,2:3]**2)
        squared_terms = ((x[:,0:1]**2)*(df_x)) + ((x[:,1:2]**2)*(df_y)) + ((x[:,2:]**2)*(df_z))
        cross_terms =  (x[:,0:1] * x[:,1:2] * df_x * df_y)+(x[:,0:1] * x[:,2:] * df_x * df_z)+(x[:,1:2] * x[:,2:] * df_y * df_z)
        return first_factor * (squared_terms + (2*cross_terms)) - 0.5

    

    geom = dde.geometry.geometry_3d.Cuboid([-10,-10,-10],[10,10,10])
   
    # singularity = dde.geometry.geometry_3d.Sphere([0,0,0], 0.1)



    icx_1 = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[0],1),
    )
    
    
    
    icx_2 = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[0],10),
    )
    
    
    
    
    icy_1 = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[1],-1),
    )
    
    icy_2 = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[1],10),
    )
    
    
    
    
    
    
    icz_1 = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[2],-1),
    )
    
    icz_2 = dde.icbc.DirichletBC(
        geom,
        lambda x: 1,
        lambda x, on_boundary:  np.isclose(x[2],10),
    )
    
    
    
    

    data = dde.data.PDE(
        geom, pde, [icx_2, icy_2,icz_2], num_domain=domain, num_boundary=boundary)


    net = dde.nn.FNN([3] + [width] * depth + [1], "tanh", "Glorot normal")


    model = dde.Model(data, net)
    model.compile("adam", lr=rate)
    losshistory, train_state = model.train(iterations=epochs)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    
    
    print("Training Complete!")

    return model


model = generate_evolution_model(500, 500, 10, 1, 0.01, 10000)

import numpy as np
import plotly.graph_objs as go

# Generate sample data points
x_values = np.linspace(-10, 10, 30)
y_values = np.linspace(-10, 10, 30)
z_values = np.linspace(-10, 10, 30)
X, Y, Z = np.meshgrid(x_values, y_values, z_values)

# Evaluate the model on the data points
input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
predictions = model.predict(input_data)
predictions = predictions.reshape(X.shape)

# Create the 3D density plot with Plotly using the 'Viridis' color scale
fig = go.Figure(data=[
    go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), mode='markers',
                 marker=dict(size=3, color=predictions.ravel(), colorscale='viridis', colorbar=dict(title='Density'))),
])

fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                  title='Assymetric Warp Drive Shape Function')

fig.show()


import numpy as np
import plotly.subplots as sp
import plotly.graph_objs as go

# Generate sample data points
x_values = np.linspace(-10, 10, 30)
y_values = np.linspace(-10, 10, 30)
z_values = np.linspace(-10, 10, 30)
X, Y, Z = np.meshgrid(x_values, y_values, z_values)

# Evaluate the model on the data points (assuming you have 'model' defined somewhere)
input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
# Assuming you have some model defined that returns predictions
predictions = model.predict(input_data)
predictions = predictions.reshape(X.shape)

# Create contour plots for slices x=0, y=0, and z=0
contour_x0 = go.Contour(z=predictions[15, :, :], colorscale='Viridis', hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>Value: %{z:.2f}')
contour_y0 = go.Contour(z=predictions[:, 15, :], colorscale='Viridis', hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>Value: %{z:.2f}')
contour_z0 = go.Contour(z=predictions[:, :, 15], colorscale='Viridis', hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}<br>Value: %{z:.2f}')

# Create subplots
fig = sp.make_subplots(rows=1, cols=3, subplot_titles=('Slice x=0', 'Slice y=0', 'Slice z=0'))

# Add contour plots to subplots
fig.add_trace(contour_x0, row=1, col=1)
fig.add_trace(contour_y0, row=1, col=2)
fig.add_trace(contour_z0, row=1, col=3)

# Update layout
fig.update_layout(title='Contours of Slices x=0, y=0, and z=0')

fig.show()



import numpy as np
import plotly.subplots as sp
import plotly.graph_objs as go

# Generate sample data points
x_values = np.linspace(-10, 10, 30)
y_values = np.linspace(-10, 10, 30)
z_values = np.linspace(-10, 10, 30)
X, Y, Z = np.meshgrid(x_values, y_values, z_values)

# Evaluate the model on the data points (assuming you have 'model' defined somewhere)
input_data = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
# Assuming you have some model defined that returns predictions
predictions = model.predict(input_data)
predictions = predictions.reshape(X.shape)

# Create a 3D surface plot
surface_plot = go.Surface(x=X, y=Y, z=Z, surfacecolor=predictions, colorscale='Viridis', colorbar=dict(title='Value'))

# Create a subplot
fig = sp.make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])

# Add surface plot to the subplot
fig.add_trace(surface_plot)

# Update layout
fig.update_layout(title='3D Surface Plot')

fig.show()

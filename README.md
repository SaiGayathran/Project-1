# Project-1
A regression model which predicts FPS from CPU &amp; GPU details and a gradient descent visualizer
The project contains 2 parts:
a) A regression mode which predicts the FPS you get using parameters like CPU cores,threads, turbo boost and GPU boost clock, shading units and memory size.The data has been taken from a kaggle dataset(https://www.kaggle.com/datasets/ulrikthygepedersen/fps-benchmark). Out of the many features available only 6 were selected considering the information about which the user shall have. Due to this it has resulted in the coefficient of determination being too low, which was further improved by engineering some features using claude. The final output results in 15-20 FPS off from the actual output.
b) A gradient descent visualizer which takes the values of w,b, learning rate alpha and number of iterations from the user then a dataset is contructed randomly using python's numpy module. Then 2 parameters are generated using the random module to design a y dataset(y=p*x+q). Then using matplotlib w,b and cost function is traced on a 3d projection of the surface.
Finally everything is put together using streamlit and deployed using the same.

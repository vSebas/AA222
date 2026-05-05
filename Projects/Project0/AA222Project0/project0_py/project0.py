#
# File: __init__.py
#

from project0_py.helpers import *

## top-level submission file


### imports here - TODO: remember to add them to your environment 
import numpy as np
import matplotlib.pyplot as plt
###


def f(a, b):
	'''
	Function for adding two numbers
	Args:
		a (float): first number
		b (float): second number
	Returns:
		c (float): result
	'''

	## TODO: Fill your code in here 
	c = a + b

	##

	return c

# Create a contour plot: see Tutorial 3 or https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html

bottom_limit_a = -10
top_limit_a = 10
num_steps_a = 100

bottom_limit_b = -10
top_limit_b = 10
num_steps_b = 100

a_vals = np.linspace(bottom_limit_a, top_limit_a, num_steps_a)	# TODO: use np.linspace() to define range of values
b_vals = np.linspace(bottom_limit_b, top_limit_b, num_steps_b)	# TODO: use np.linspace() to define range of values

# Generates a 2D grid of function values
A, B = np.meshgrid(a_vals, b_vals)
Z = f(A,B)

# TODO: Replace with the appropriate arguments 
plt.contour(A,B,Z)  
plt.title("Contour Plot of f(a, b)")
plt.xlabel("a")
plt.ylabel("b")

# show plot
plt.show()

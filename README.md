# P1 Optimization and Operational Research - LEFEBVRE Romain Group B

## Overview

This project aims to optimize the profit from wheat and barley cultivation based on specified constraints related to perimeter, area, and non-linear conditions. The optimization is performed using SLSQP method from the scipy library, and results are visualized in a 3D plot showing the feasible region and the objective function with the optimized solution.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Known Issues](#known-issues)

## Features

- **Profit Optimization**: Computes the optimized size of the area (length & width) with the maximum profit possible based on the constraints and the bounds entered.
- **Constraint Implementation**: Perimeter, area, and non-linear constraints are implemented
- **Visualization**: Displays a 3D plot of the objective function and the feasible region with the optimal solution found.
- **Command-Line Arguments**: Allows to chose the bounds of the variables

## Getting Started

### Prerequisites

- **Python 3.X**
- **Matplotlib**
- **Numpy**
- **Scipy**

You can install them using pip:

```bash
pip install numpy scipy matplotlib
```
### Execution

You can execute this script using a console with command-line arguments.
Here is an example with the bounds of x and y set up to [1,100] and [20,40] respectively:

```bash
python P1_LEFEBVRE_Romain_Group_B.py --x_lower 1 --x_upper 100 --y_lower 20 --y_upper 40
```
You can get some help with the following command:
```bash
python P1_LEFEBVRE_Romain_Group_B.py -h
```
## Known Issues

There is a known issue regarding the 3D plot. With certain bounds, the color representing the feasible region is not exactly the same as in the code and as the legend says.
Usually, if this issue happens, the color shown is darker than expected.

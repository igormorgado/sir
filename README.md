Pythoh SIR Model simulation
===========================

Only works on Python 3

Numba must be version 0.48, otherwise it will not work.

# TODO

Fix issue with new numba versions.

# sir.py

Module to run stocastic and deterministic sir simulations

# timeitplot.py

Module to graph performance data 

# randomwalk.py

A sample random walk simulation to test stochastic optimization methods

To execute call:  python3 randomwalk.py


# paper_cba.py

Creates the images for the paper to CBA

To execute call:  python3 paper_cba.py

# Notes

Images should be plotted as 300 dpi. Sizes should be

  [6, 4.24]
  [3, 2.125]
  [6, 3.375]
  [3, 1.6875]

These sizes keep the font size optimal for paper. Pay attention that 3xN data
are very small and legend should not be drawn.

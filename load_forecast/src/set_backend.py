"""
Set the matplotlib backend to a non-interactive one for headless operation.
This should be imported before any other matplotlib imports.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

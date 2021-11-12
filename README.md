# SIRS
Python SIRS model on a 2D lattice

Usage: SIRS.py Lattice Size, Sweeps, p1, p2, p3, immune fraction, mode 

p1 - probability a susceptible site becomes infected, give it has at least one infected nearest neighbour 
p2 - probability an infected site recovers
p3 - probability a recovered site becomes susceptible again 
immune fraction - fraction of sites incapable of becoming infected 

Modes: 
animate - will show an animation of the model, number of sweeps is irrelavent 
infected fraction - returns fraction of sites infected for a given p2 and variable p1&3, then returns fraction of sites infected for fixed p2&3 and variable p1
immune fraction - returns fraction of sites infected for a give immunity fraction

"""
SIRS Model
"""

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import random as r
from matplotlib.animation import FuncAnimation
from matplotlib import cm

class SIRS(object):

    def __init__(self, shape, sweeps, i_prob, r_prob, s_prob, frac_im):

        self.shape = shape
        self.sweeps = sweeps
        self.i_prob = i_prob
        self.r_prob = r_prob
        self.s_prob = s_prob
        self.frac_im = frac_im
        self.stopsim = False
        self.create_lattice()

    """Stop the visulation when the window is closed"""
    def stop(self):

        self.stopsim = True

    """Impliments the periodic boundary conditions"""
    def pbc(self, i):

        if (i > self.shape-1) or (i < 0):
            i = np.mod(i, self.shape)
            return i
        else:
            return i

    """Create an NxN lattice with sites randomly assigned a state"""
    def create_lattice(self):

        #Everything that isn't immune is equally likely
        self.lattice = np.random.choice(4, [self.shape, self.shape], p=[(1-self.frac_im)/3.0, (1-self.frac_im)/3.0, (1-self.frac_im)/3.0, self.frac_im])

#######################Update and Rules ##########################

    """Update the lattice using the SIRS rules"""
    def update(self):

        for n in range(self.shape*self.shape):
            i = np.random.randint(self.shape)
            j = np.random.randint(self.shape)

            if self.lattice[i][j] == 0:
                outcome = self.infection(i,j)
                if outcome == True:
                    self.lattice[i][j] = 1
            elif self.lattice[i][j] == 1:
                outcome = self.recovery()
                if outcome == True:
                    self.lattice[i][j] = 2
            elif self.lattice[i][j] == 2:
                outcome = self.suceptible()
                if outcome == True:
                    self.lattice[i][j] = 0

    """Test to see if an S site should become an I site, using Monte Carlo methods"""
    def infection(self, i, j):

        if self.lattice[self.pbc(i+1)][j] ==1:
            random = r.random()
            if random <= self.i_prob:
                return True
        elif self.lattice[self.pbc(i-1)][j] == 1:
            random = r.random()
            if random <= self.i_prob:
                return True
        elif self.lattice[i][self.pbc(j-1)] == 1:
            random = r.random()
            if random <= self.i_prob:
                return True
        elif self.lattice[i][self.pbc(j+1)] == 1:
            random = r.random()
            if random <= self.i_prob:
                return True

    """Test to see if an I site should become an R site, using Monte Carlo methods"""
    def recovery(self):

        random = r.random()
        if random <= self.r_prob:
            return True

    """Test to see if an R site should become an S site, using Monte Carlo methods"""
    def suceptible(self):

        random = r.random()
        if random <= self.s_prob:
            return True

#########################Observables############################

    """Calculate the total number of infected sites"""
    def total_infected(self):

        infected = 0.0
        for i in range(self.shape):
            for j in range(self.shape):
                if self.lattice[i][j] == 1.0:
                    infected += 1.0

        return infected

#######################Phase Diagrams###########################

    """Calculate the data to create the phase diagrams"""
    def infected_fraction(self, p1_range, p3_range):

        #Create a 2D array to store the data in
        infected = np.empty((p1_range.size, p3_range.size))
        infected_square = np.empty((p1_range.size, p3_range.size))

        #Loop over each of the probabilities
        for i in range(p1_range.size):
            for j in range(p3_range.size):
                self.i_prob = p1_range[i]
                self.s_prob = p3_range[j]
                self.create_lattice()

                #Lists for the data for a specific (p1,p3)
                infected_ij = []
                infected_ij_square = []

                for n in range(self.sweeps):
                    self.update()

                    #Give 100 sweeps to equilibiriate
                    if n > 100:
                        total_infected = self.total_infected()
                        infected_ij.append(total_infected)
                        infected_ij_square.append(total_infected**2.0)

                #Set to zero if an absorbing state is reached at any point
                if any(infected_ij) == 0.0:
                    infected[i,j] = 0.0
                    infected_square[i,j] = 0.0
                else:
                    infected[i,j] = np.mean(infected_ij)
                    infected_square[i,j] = np.mean(infected_ij_square)

        #Calculate the fraction and variance
        infected_fraction = infected/(self.shape*self.shape)
        var = (infected_square - infected**2.0)/(self.shape*self.shape)

        return infected_fraction, var

######################Cut Plane#################################

    """Calculate the data for a cut plane with p2=p3=0.5"""
    def cut_plane(self, p1_range):

        #Create arrays to store the data in
        infected = np.empty(p1_range.size)
        infected_square = np.empty(p1_range.size)
        error = np.empty(p1_range.size)

        #Loop over the given p1 range
        for i in range(p1_range.size):
            print(p1_range[i])
            self.i_prob = p1_range[i]
            self.create_lattice()

            #Lists to store the data in
            infected_ij = []
            infected_ij_square = []
            for n in range(self.sweeps):
                self.update()

                #Give 100 sweeps to equilibiriate
                if n > 100:
                    total_infected = self.total_infected()
                    infected_ij.append(total_infected)
                    infected_ij_square.append(total_infected**2.0)

            #Set equal to 0 if an absorber state is reached
            if any(infected_ij) == 0.0:
                infected[i] = 0.0
                infected_square[i] = 0.0
            else:
                infected[i] = np.mean(infected_ij)
                infected_square[i] = np.mean(infected_ij_square)

            #Calculate the error
            error[i] = self.var_error(infected_ij, infected_ij_square)

        #Calculate the fraction and variance
        infected_fraction = infected/(self.shape*self.shape)
        var = (infected_square - infected**2.0)/(self.shape*self.shape)

        return infected_fraction, var, error

    """Calculate the error on the variance of the cut plane using boot strap method"""
    def var_error(self, infected_fraction, infected_square):

        #Lists for the errors
        var = []
        var_square = []

        #Resample 20 times
        for i in range(30):
            print(i)
            #List to put the resamples indices in
            samples = []
            #Pick the indices to resample from
            for j in range(len(infected_fraction)):
                samples.append(np.random.randint(len(infected_fraction)))
            #Put the resamples in lists
            frac_samples = []
            frac_square_samples = []
            for k in samples:
                frac_samples.append(infected_fraction[k])
                frac_square_samples.append(infected_square[k])

            #Calaculate the new variance
            sample_mean = np.mean(frac_samples)
            sample_square_mean = np.mean(frac_square_samples)

            var_sample =  (sample_square_mean - sample_mean**2.0)/(self.shape*self.shape)
            var.append(var_sample)
            var_square.append(var_sample**2.0)

        #Calculate the errors
        mean = np.mean(var)
        mean_square = np.mean(var_square)
        error = math.sqrt(mean_square-mean**2.0)

        return error

#######################Immune Fraction###########################

    """Calculate the average infected fraction for varying immunity fractions"""
    def immune_fraction(self, frac_im_range):

        #empty array for data
        infected = np.empty(frac_im_range.size)

        #Loop over the set immune fractions
        for i in range(frac_im_range.size):
            print(frac_im_range[i])
            self.frac_im = frac_im_range[i]
            self.create_lattice()

            #Store date for specific fraction
            infected_ij = []
            for n in range(self.sweeps):
                self.update()

                #100 sweeps to equilibiriate
                if n > 100:
                    total_infected = self.total_infected()
                    infected_ij.append(total_infected)

            #set equal to 0 for an absorbing state
            if any(infected_ij) == 0.0:
                infected[i] = 0.0
            else:
                infected[i] = np.mean(infected_ij)

        #infected_fraction = infected/(self.shape*self.shape)

        return infected


####################Write Output Files#############################

    """Write output data for the phase diagrams"""
    def write_phase_data(self, p1_range, p3_range, fraction, variance):

        file = open('phase_output.dat', 'w')
        for i in range(p1_range.size):
            for j in range(p3_range.size):
                file.write(str(p1_range[i]) + ',')
                file.write(str(p3_range[j]) + ',')
                file.write(str(fraction.T[i][j]) + ',')
                file.write(str(variance.T[i][j]) + '\n')
        file.close()

    """Write output data for the cut plane"""
    def write_cut_plane_data(self, p1_range, fraction, variance, error):

        file = open('cut_plane_output.dat', 'w')
        for i in range(p1_range.size):
            file.write(str(p1_range[i]) + ',')
            file.write(str(fraction[i]) + ',')
            file.write(str(variance[i]) + ',')
            file.write(str(error[i]) + '\n')
        file.close()

    """Write output data for the immunity fraction"""
    def write_immune_data(self, immune, infected, error):

        file = open('immune_output.dat', 'w')
        for i in range(immune.size):
            file.write(str(immune[i]) + ',')
            file.write(str(infected[i]) + ',')
            file.write(str(error[i]) + '\n')
        file.close()

#######################Run Models##############################

    """Run the phase diagrams and cut plane data collection"""
    def run_infected_fraction(self):

        #Set the ranges for p1 and p2
        p1_range = np.arange(0.0, 1.05, 0.05)
        p3_range = np.arange(0.0, 1.05, 0.05)

        #Collect data
        infected, var = self.infected_fraction(p1_range, p3_range)

        #Change range for p1 and the number of sweeps
        p1_cut_range = np.arange (0.2, 0.525, 0.025)
        self.sweeps *= 10

        #Collect data
        cut_fraction, cut_var, cut_error = self.cut_plane(p1_cut_range)

        #Write outputs and save numpy arrays to make ploting easier
        self.write_phase_data(p1_range, p3_range, infected, var)
        np.save('p1_range.npy', p1_range)
        np.save('p3_range.npy', p3_range)
        np.save('infected_fraction.npy', infected)
        np.save('infected_var.npy', var)

        self.write_cut_plane_data(p1_cut_range, cut_fraction, cut_var, cut_error)
        np.save('p1_cut_range.npy', p1_cut_range)
        np.save('cut_fraction.npy', cut_fraction)
        np.save('cut_var.npy', cut_var)
        np.save('cut_error.npy', cut_error)

    """Run the immune fraction data collection"""
    def run_immune_fraction(self):

        #Set the number of times to run this
        reruns = 30

        #Set range of fractions
        frac_im_range = np.arange(0.0, 0.51, 0.01)

        #Data collection arrays
        infected = np.empty((reruns,len(frac_im_range)))
        infected_square = np.empty((reruns, len(frac_im_range)))


        for i in range(reruns):
            print('Run number ', i)
            fraction = self.immune_fraction(frac_im_range)
            for j in range(frac_im_range.size):
                infected[i,j] = fraction[j]
                infected_square[i,j] = fraction[j]**2.0


        infected_fraction = np.empty(len(frac_im_range))
        error = np.empty(len(frac_im_range))

        N = self.shape*self.shape
        for i in range(len(frac_im_range)):
            infected_fraction[i] = np.mean(infected.T[i])/(self.shape*self.shape)
            sd = math.sqrt((np.mean(infected_square.T[i]) - np.mean(infected.T[i])**2.0)/N)
            error[i] = sd/math.sqrt(N)


        self.write_immune_data(frac_im_range, infected_fraction, error)

        np.save('frac_im_range.npy', frac_im_range)
        np.save('frac_im_fraction.npy', infected_fraction)
        np.save('frac_im_error.npy', error)

    def run_animation_in_file(self):

        self.update()

    def animate(self, i):
        self.run_animation_in_file()
        self.mat.set_data(self.lattice)
        return self.mat,

    def run(self):
        fig, ax = plt.subplots()
        self.mat = ax.imshow(self.lattice, cmap='seismic')
        fig.colorbar(self.mat)
        ani = FuncAnimation(fig, self.animate, interval = 100, blit = True)

        plt.show()

# ########################Threading animation#######################
#
#     def run_animation(self):
#         self.stopsim = False
#
#         for i in range(self.sweeps):
#             if self.stopsim: break
#             self.update()


if __name__ == "__main__":

    if len(sys.argv) != 7:
        print("Incorrect Number of Arguments Presented.")
        print("Usage: " + sys.argv[0] + "lattice Size, Number of sweeps, infection prob, recovery prob, suceptible prob, imune fraction")
        quit()
    else:
        shape = int(sys.argv[1])
        sweeps = int(sys.argv[2])
        i_prob = float(sys.argv[3])
        r_prob = float(sys.argv[4])
        s_prob = float(sys.argv[5])
        frac_im = float(sys.argv[6])


    model = SIRS(shape, sweeps, i_prob, r_prob, s_prob, frac_im)
    #model.run_infected_fraction()
    #model.run_immune_fraction()
    model.run()

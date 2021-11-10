"""
SIRS Model
"""

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from matplotlib import cm

class SIRS(object):

    def __init__(self, shape, sweeps, infect_prob, recover_prob, sus_prob, immune_frac):

        self.shape = shape
        self.sweeps = sweeps
        self.infect_prob = infect_prob
        self.recover_prob = recover_prob
        self.sus_prob = sus_prob
        self.immune_frac = immune_frac
        self.create_lattice()

    """Create an NxN lattice with sites randomly assigned a state"""
    def create_lattice(self):
        #Everything that isn't immune is equally likely
        # 0=sus, 1=infected, 2=recovered, 3=immune
        self.lattice = np.random.choice(4, [self.shape, self.shape], p = [(1-self.immune_frac)/3.0, (1-self.immune_frac)/3.0, (1-self.immune_frac)/3.0, self.immune_frac])

########## Update the lattice ##########

    """Impliments the periodic boundary conditions"""
    def pbc(self, i):

        if (i > self.shape-1) or (i < 0):
            i = np.mod(i, self.shape)
            return i
        else:
            return i

    """Update the lattice using the SIRS rules"""
    def update(self):

        for n in range(self.shape*self.shape):
            i = np.random.randint(self.shape)
            j = np.random.randint(self.shape)

            #If site is sus check to see if it should be infected
            if self.lattice[i][j] == 0:
                outcome = self.infection(i, j)
                if outcome == True:
                    self.lattice[i][j] =1

            #If site is infected check to see if it should recover
            elif self.lattice[i][j] == 1:
                outcome = self.recovery()
                if outcome == True:
                    self.lattice[i][j] =2

            #If site it recovered check to see if it should beomce sus
            elif self.lattice[i][j] == 2:
                outcome = self.suceptible()
                if outcome == True:
                    self.lattice[i][j] = 0

    """Test to see if an S site should become an I site, using Monte Carlo methods"""
    def infection(self, i, j):

        #If S site has at least one infected nearest neighbour infect with probability infect_prob
        if (self.lattice[self.pbc(i+1)][j]) ==1 or (self.lattice[self.pbc(i-1)][j] == 1) or (self.lattice[i][self.pbc(j-1)] == 1) or (self.lattice[i][self.pbc(j+1)] == 1):
            random_number = random.random()
            if random_number <= self.infect_prob:
                return True

    """Test to see if an I site should become an R site, using Monte Carlo methods"""
    def recovery(self):

        random_number = random.random()
        if random_number <= self.recover_prob:
            return True

    """Test to see if an R site should become an S site, using Monte Carlo methods"""
    def suceptible(self):

        random_number = random.random()
        if random_number <= self.sus_prob:
            return True

########## Data Collection ##########

    """Calculate the total number of infected sites"""
    def total_infected(self):

        infected = 0.0
        for i in range(self.shape):
            for j in range(self.shape):
                if self.lattice[i][j] == 1.0:
                    infected += 1.0

        return infected

    """Calculate the data to create the phase diagrams"""
    def infected_fraction(self, infect_range, sus_range):

        #2D arrays for the data
        infected = np.empty((infect_range.size, sus_range.size))
        infected_square = np.empty((infect_range.size, sus_range.size))

        #Loop over the probabilities
        for i, i_prob in enumerate(infect_range):
            for j, s_prob in enumerate(sus_range):
                self.infect_prob = i_prob
                self.sus_prob = s_prob

                #Lists for specific (i_prob, s_prob)
                infected_is = []
                infected_is_sqr = []

                for n in range(self.sweeps):
                    self.update()

                    #Give 100 sweeps to equilibiriate
                    if n > 100:
                        total_infected = self.total_infected()
                        infected_is.append(total_infected)
                        infected_is_sqr.append(total_infected**2.0)

                #Set to zero if an absorbing state is reached at any point
                if 0.0 in infected_is:
                    infected[i,j] = 0.0
                    infected_square[i,j] = 0.0
                else:
                    infected[i,j] = np.mean(infected_is)
                    infected_square[i,j] = np.mean(infected_is_sqr)

        #Calculate fraction and variance
        infected_fraction = infected/(self.shape*self.shape)
        var = (infected_square - infected**2.0)/(self.shape*self.shape)

        return infected_fraction, var

    """Calculate the data for a cut plane with recover_prob=sus_prob=0.5"""
    def cut_plane(self, infect_range):

        #arrays for data
        infected = np.empty(infect_range.size)
        infected_square = np.empty(infect_range.size)
        error = np.empty(infect_range.size)

        #Loop over given infect_prob range
        for i, i_prob in enumerate(infect_range):
            self.infect_prob = i_prob
            self.create_lattice()

            #Lists to store data
            infected_i = []
            infected_i_sqr = []
            for n in range(self.sweeps):
                self.update()

                #Give 100 sweeps to equilibiriate
                if n > 100:
                    total_infected = self.total_infected()
                    infected_i.append(total_infected)
                    infected_i_sqr.append(total_infected**2.0)

            #Set to zero if an absorbing state is reached at any point
            if 0.0 in infected_i:
                infected[i] = 0.0
                infected_i_sqr[i] = 0.0
            else:
                infected[i] = np.mean(infected_i)
                infected_square[i] = np.mean(infected_i_sqr)

            #Calculate error
            error[i] = self.bootstrap_error(infected_i, infected_i_sqr)

        #Calculate fraction and variance
        infected_fraction = infected/(self.shape*self.shape)
        var = (infected_square - infected**2.0)/(self.shape*self.shape)

        return infected_fraction, var, error

    """Use bootstrap method to Calculate errors"""
    def bootstrap_error(self, input, input_sqr):

        #Collect values
        values = []
        values_sqr = []

        # Resample 30 times
        for n in range(30):
            #Randomly select sample points
            samples = [np.random.randint(len(input)) for i in range(len(input))]
            #Find samples
            input_samples = [input[sample] for sample in samples]
            input_sqr_samples = [input_sqr[sample] for sample in samples]

            #Take means
            sample_mean = np.mean(input_samples)
            sample_sqr_mean = np.mean(input_sqr_samples)

            #calculate required value
            value = (sample_sqr_mean - sample_mean**2.0)/(self.shape*self.shape)

            #Append
            values.append(value)
            values_sqr.append(value**2.0)

        #Calculate error on value
        mean = np.mean(values)
        mean_sqr = np.mean(values_sqr)

        error = math.sqrt(mean_sqr-mean**2.0)

        return error

    """Calculate the average infected fraction for varying immunity fractions"""
    def immune_fraction(self, immune_frac_range):

        #Array for data
        infected = np.empty(immune_frac_range.size)

        #Loop over set immune fractions
        for i, immune_frac in enumerate(immune_frac_range):
            self.immune_frac = immune_frac
            self.create_lattice()

            #Data for specific fraction
            infected_i = []

            for n in range(self.sweeps):
                self.update()

                #Give 100 sweeps to equilibiriate
                if n > 100:
                    infected_i.append(self.total_infected())

            #Set to 0 if absorbing state reached
            if 0.0 in infected_i:
                infected[i] = 0.0
            else:
                infected[i] = np.mean(infected_i)

        return infected

    """Write output data for the phase diagrams"""
    def write_phase_data(self, infect_range, sus_range, fraction, variance):

        file = open('phase_output.dat', 'w')
        for i in range(infect_range.size):
            for j in range(sus_range.size):
                file.write(str(infect_range[i]) + ',')
                file.write(str(sus_range[j]) + ',')
                file.write(str(fraction.T[i][j]) + ',')
                file.write(str(variance.T[i][j]) + '\n')
        file.close()

    """Write output data for the cut plane"""
    def write_cut_plane_data(self, infect_range, fraction, variance, error):

        file = open('cut_plane_output.dat', 'w')
        for i in range(infect_range.size):
            file.write(str(infect_range[i]) + ',')
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

########## Run data collection #########

    """Run the phase diagrams and cut plane data collection"""
    def run_infected_fraction(self):

        #Set probability ranges
        infect_range = np.arange(0.0, 1.05, 0.05)
        sus_range = np.arange(0.0, 1.05, 0.05)

        #Collect data
        infected, variance = self.infected_fraction(infect_range, sus_range)

        #Write outputs and save numpy arrays to make ploting easier
        self.write_phase_data(infect_range, sus_range, infected, variance)
        np.save('infect_range.npy', infect_range)
        np.save('sus_range.npy', sus_range)
        np.save('infected_fraction.npy', infected)
        np.save('infected_variance.npy', variance)

        #Change range for infect_prob and increase sweep number
        infect_range = np.arange(0.2, 0.525, 0.025)
        self.sweeps *= 10

        #Collect data
        fraction, variance, error = self.cut_plane(infect_range)

        #Write outputs and save numpy arrays to make ploting easier
        self.write_cut_plane_data(infect_range, fraction, variance, error)
        np.save('infect_cut_range.npy', infect_range)
        np.save('cut_fraction.npy', fraction)
        np.save('cut_variance.npy', variance)
        np.save('cut_error.npy', error)

    def run_immune_fraction(self):

        #Set the number of times to run this
        reruns = 30

        #Set range of fractions
        immune_frac_range = np.arange(0.0, 0.51, 0.01)

        #Data collection arrays
        infected = np.empty((reruns, len(immune_frac_range)))
        infected_square = np.empty((reruns, len(immune_frac_range)))

        for i in range(reruns):
            fraction = self.immune_fraction(immune_frac_range)
            for j in range(immune_frac_range.size):
                infected[i,j] = fraction[j]
                infected_square[i,j] = fraction[j]**2.0

        infected_fraction = np.empty(immune_frac_range.size)
        error = np.empty(immune_frac_range.size)

        for i in range(immune_frac_range.size):
            infected_fraction[i] = np.mean(infected.T[i])/(self.shape*self.shape)
            sd =  math.sqrt((np.mean(infected_square.T[i]) - np.mean(infected.T[i])**2.0)/(self.shape*self.shape))
            error[i] = sd/self.shape

        self.write_immune_data(immune_frac_range, infected_fraction, error)

        np.save('immune_frac_range.npy', immune_frac_range)
        np.save('immune_frac_fraction.npy', infected_fraction)
        np.save('immune_frac_error.npy', error)

########## Run animation #########

    def update_animation(self):

        self.update()

    def animate(self, i):
        self.update_animation()
        self.mat.set_data(self.lattice)
        return self.mat,

    def run_animation(self):
        fig, ax = plt.subplots()
        self.mat = ax.imshow(self.lattice, cmap = 'seismic')
        fig.colorbar(self.mat)
        ani = FuncAnimation(fig, self.animate, interval= 1, blit = False)

        plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 8:
        print("Incorrect Number of Arguments Presented.")
        print("Usage: " + sys.argv[0] + "lattice Size, Number of sweeps, infection prob, recovery prob, suceptible prob, imune fraction, mode")
        quit()
    elif sys.argv[7] not in ['infected fraction', 'immune fraction', 'animate']:
        print('Please choose one of the following \n infected fraction, immune fraction, animate')
        quit()
    else:
        shape = int(sys.argv[1])
        sweeps = int(sys.argv[2])
        i_prob = float(sys.argv[3])
        r_prob = float(sys.argv[4])
        s_prob = float(sys.argv[5])
        frac_im = float(sys.argv[6])
        mode = sys.argv[7]


    model = SIRS(shape, sweeps, i_prob, r_prob, s_prob, frac_im)
    if mode == 'infected fraction':
        model.run_infected_fraction()
    elif mode == 'immune_fraction':
        model.run_immune_fraction()
    elif mode == 'animate':
        model.run_animation()

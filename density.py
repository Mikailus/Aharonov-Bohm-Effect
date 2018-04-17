import numpy as np
import cmath
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.special as scsp
from multiprocessing import Pool
import time
import os

def density_formula(xyaks):
    # x - x coordinate
    # y - y coordinate
    # alfa - parameter related to magnetic field
    # k - wavenumber
    # suma - number of elements in sum used to evaluating density for given (x,y) coordinates
    # h - Planck constant divided by 2 pi
    # m - particle's mass
    x, y, alfa, k, suma = xyaks
    h, m = 1, 1                   
    r = (x**2 + y**2)**0.5      
    theta = np.arctan2(y, x)
    psi = np.sum(scsp.jv(abs(np.arange(-suma, suma+1)+alfa), k * r) * np.exp(1j*np.arange(-suma, suma+1)*theta) * np.power(-1j, abs(np.arange(-suma, suma+1)+alfa)))
    return np.linalg.norm(psi)**2


def density(alfa,k,suma,xmin,xmax,nx,ymin,ymax,ny):
    tic = time.clock()
    mypath = 'Results/' + 'alfa=' + alfa + '/' + 'k=' + k
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    path_x = mypath + '/' + 'x_density.txt' # array of x coordinates saved in a file
    path_y = mypath + '/' + 'y_density.txt' # array of y coordinates saved in a file
    path_dens = mypath + '/' + 'density.txt' # array of densities for given coordinates (x,y) saved in a file

    #Initializing array z with number of rows equal to nx and number of columns equal to ny, filled with 0's
    z = np.zeros((nx, ny),dtype=np.float64)                                             
    # 1D array with x coordinates
    x = np.linspace(xmin, xmax, nx)                                                     
    # 1D array with y coordinates
    y = np.linspace(ymin, ymax, ny)                                                     
    # Initializing list which will become a stack for tasks to be done
    stack = []                                                                          
    # Setting number of processes used in parallel computing
    nproc = 8                                                                           
    # This condition is required on Windows
    if __name__ == '__main__':                                                         
        p = Pool(nproc)                                                                 
        for i in range(0, nx):                                                          
            for j in range(0, ny):                                                      
                # Appending tasks to stack list
                stack.append((j,i))                                                     
        while len(stack):                                                               
            # Takes nproc last element from stack list
            temps = stack[-nproc:]                                                      
            # Creates 8 element list of parameters stored in tuple
            xys = [(x[i], y[j],float(alfa), float(k), int(suma)) for i, j in temps]     
            # Results of last nproc tasks evaluated in parallel
            results = p.map(density_formula, xys)                                            
            for proc in range(len(temps)):                                              
                # Storing last nproc results in z 2D array
                z[temps[proc][1],temps[proc][0]] = results[proc]                        
           # Deleting last nproc from stack
            del stack[-nproc:]                                                          

        # Saving x coordinates, y coordinates, density matrix in txt files
        np.savetxt(path_x, x, delimiter=',')
        np.savetxt(path_y, y, delimiter=',')
        np.savetxt(path_dens, z, delimiter=',')

        outfile = open(mypath + '/' + 'info_dens.txt', 'w') # some basic information about the output and image
        outfile.write('alfa = ' + alfa + '\n')
        outfile.write('k = ' + k + '\n')
        outfile.write('Number of elements = ' + str(2*int(suma)+1) + '\n')
        outfile.write('xmin = ' + str(xmin) + '\n')
        outfile.write('xmax = ' + str(xmax) + '\n')
        outfile.write('Density of points on x axis = ' + str(nx) + '\n')
        outfile.write('ymin = ' + str(ymin) + '\n')
        outfile.write('ymax = ' + str(ymax) + '\n')
        outfile.write('Density of points on y axis = ' + str(ny) + 's\n')

        toc = time.clock()
        outfile.write('Evaluating time = ' + str(toc-tic) + '\n')
        outfile.close()
        # Printing time elapsed on evaluating
        print(toc-tic)
        # Creates pcolor plot of density
        plt.pcolor(x, y, z)
        plt.colorbar()
        # Saves that plot
        plt.savefig(mypath +'/density.png')
        # And shows
        plt.show()

#Initial parameters passed to density function
alfa = 0.5 # parameter related to magnetic field
k = 1.0 # parameter related to energy of a particle
suma = 1000 # number of elements to summarize
xmin=-100.5 # starting x axis value
xmax=100.5 # finishing x axis value
nx = 101 # number of divisions of x axis
ymin=-12.5 # starting y axis value
ymax=12.5 # finishing y axis value
ny = 101 # number of divisions of y axis
density(str(alfa),str(k),str(suma),xmin,xmax,nx,ymin,ymax,ny)
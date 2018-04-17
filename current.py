import numpy as np
import cmath
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.special as scsp
from multiprocessing import Pool
import time
import os


def current_x_formula(xyaks):
    # Returns x component of density probability current
    # x - x coordinate
    # y - y coordinate
    # a - parameter related to magnetic field
    # k - wavenumber
    # suma - number of elements in sum used to evaluating density for given (x,y) coordinates
    # h - Planck constant divided by 2 pi
    # m - particle's mass
    x, y, a, k, suma = xyaks
    h, mass = 1, 1
    r = (x ** 2 + y ** 2) ** 0.5
    if r==0:
        r=0.00000001
    # Vector potential A
    A = -(a*h)/(mass*r)
    theta = np.arctan2(y, x)
    # Final result for given (x,y) coordinates is divided into 3 parts for better visibility and to emphasize that each has different multiplying factor
    part1 = 0.
    part1 = np.sum(scsp.jv(abs(np.arange(-suma, suma + 1) + a), k * r) * np.exp(-1j * np.arange(-suma, suma + 1) * theta) * np.power(1j, abs(np.arange(-suma, suma + 1) + a)))
    part2_1 = 0.
    part2_1 = np.sum(np.power(-1j, abs(np.arange(-suma, suma + 1) + a))*(scsp.jv(abs(np.arange(-suma, suma + 1) + a)-1, k * r) - scsp.jv(abs(np.arange(-suma, suma + 1) + a)+1, k * r))*np.exp(1j * np.arange(-suma, suma + 1) * theta) )
    part2_1 = part2_1*(x*k)/(2*r)
    part2_2 = 0.
    part2_2 = np.sum(np.power(-1j, abs(np.arange(-suma, suma + 1) + a))*scsp.jv(abs(np.arange(-suma, suma + 1) + a), k * r)*np.exp(1j * np.arange(-suma, suma + 1) * theta)*np.arange(-suma,suma+1))
    part2_2 = (-1j) * y/(r**2) * part2_2
    part3 = 0.
    part3 = part1 * (part2_1 + part2_2)
    return h/mass*part3.imag + A*y/r*np.linalg.norm(part1)**2

def current_y_formula(xyaks):
    # Returns y component of density probability current
    # x - x coordinate
    # y - y coordinate
    # a - parameter related to magnetic field
    # k - wavenumber
    # suma - number of elements in sum used to evaluating density for given (x,y) coordinates
    # h - Planck constant divided by 2 pi 
    # m - particle's mass
    x, y, a, k, suma = xyaks

    h, mass = 1, 1                      
    r = (x ** 2 + y ** 2) ** 0.5        

    theta = np.arctan2(y, x)
    if r==0:
        r=0.00000001
     # Vector potential A
    A = -(a * h) / (mass * r)
    # Final result for given (x,y) coordinates is divided into 3 parts for better visibility and to emphasize that each has different multiplying factor
    part1 = 0.
    part1 = np.sum(scsp.jv(abs(np.arange(-suma, suma + 1) + a), k * r) * np.exp(-1j * np.arange(-suma, suma + 1) * theta) * np.power(1j, abs(np.arange(-suma, suma + 1) + a)))
    part2_1 = 0.
    part2_1 = np.sum(np.power(-1j, abs(np.arange(-suma, suma + 1) + a))*(scsp.jv(abs(np.arange(-suma, suma + 1) + a)-1, k * r) - scsp.jv(abs(np.arange(-suma, suma + 1) + a)+1, k * r))*np.exp(1j * np.arange(-suma, suma + 1) * theta) )
    part2_1 = part2_1*(y*k)/(2*r)
    part2_2 = 0.
    part2_2 = np.sum(np.power(-1j, abs(np.arange(-suma, suma + 1) + a))*scsp.jv(abs(np.arange(-suma, suma + 1) + a), k * r)*np.exp(1j * np.arange(-suma, suma + 1) * theta)*np.arange(-suma,suma+1))
    part2_2 = (1j) * x/(r**2) * part2_2
    part3 = 0.
    part3 = part1 * (part2_1 + part2_2)
    return h/mass*part3.imag - A*x/r*np.linalg.norm(part1)**2

def current(alfa,k,suma,xmin,xmax,nx,ymin,ymax,ny):
    tic = time.clock()
    mypath = 'Results/' + 'alfa=' + alfa + '/' + 'k=' + k
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    path_x = mypath + '/' + 'x_coordinate_current.txt' # array of x coordinates saved in a file
    path_y = mypath + '/' + 'y_coordinate_current.txt' # array of y coordinates saved in a file
    path_current_x = mypath + '/' + 'current_x.txt' # array of x component of current for given coordinates (x,y) saved in a file
    path_current_y = mypath + '/' + 'current_y.txt' # array of y component of current for given coordinates (x,y) saved in a file

    #Initializing arrays of current with number of rows equal to nx and number of columns equal to ny, filled with 0's
    current_x = np.zeros((nx, ny), dtype=np.float64)
    current_y = np.zeros((nx, ny), dtype=np.float64)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    # Initializing list which will become a stack for tasks to be done
    stack = []                                                                          
    # Setting number of processes used in parallel computing
    nproc = 8
    # This condition is required on Windows
    if __name__ == '__main__':
        p = Pool(nproc)
        for h in range(0, nx):
            for j in range(0, ny):
                # Appending tasks to stack list
                stack.append((h,j))
        while len(stack):
            # Takes nproc last element from stack list
            temps = stack[-nproc:]
            # Creates 8 element list of parameters stored in tuple
            xys = [(x[i], y[j], float(alfa), float(k), int(suma)) for i, j in temps]
            # Results of last nproc tasks evaluated in parallel
            results_x = p.map(current_x_formula, xys)
            results_y = p.map(current_y_formula, xys)
            for proc in range(len(temps)):
                # Storing last nproc results in z 2D array
                current_x[temps[proc][1], temps[proc][0]] = results_x[proc]
                current_y[temps[proc][1], temps[proc][0]] = results_y[proc]
            # Deleting last nproc from stack
            del stack[-nproc:]

        # Saving x coordinates, y coordinates, current matrices in txt files
        np.savetxt(path_x, x, delimiter=',')
        np.savetxt(path_y, y, delimiter=',')
        np.savetxt(path_current_x, current_x, delimiter=',')
        np.savetxt(path_current_y, current_y, delimiter=',')
        
        outfile = open(mypath + '/' + 'info_current.txt', 'w')
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

        # Creates streamplot for current
        normalising_factor = np.sqrt(current_x**2 + current_y**2)
        plt.figure()
        plt.streamplot(x,y,current_x,current_y,color=normalising_factor,cmap=cm.jet,linewidth=2,arrowstyle='->',arrowsize=1.5)
        plt.colorbar()

        # Saves that plot
        plt.savefig(mypath + '/' + 'current.png')
        # And shows
        plt.show()

#Initial parameters passed to current function
alfa = 0.5 # parameter related to magnetic field
k = 1.0 # parameter related to energy of a particle
suma = 1000 # number of elements to summarize
xmin=-100.5 # starting x axis value
xmax=100.5 # finishing x axis value
nx = 101 # number of divisions of x axis
ymin=-12.5 # starting y axis value
ymax=12.5 # finishing y axis value
ny = 101 # number of divisions of y axis
current(str(alfa),str(k),str(suma),xmin,xmax,nx,ymin,ymax,ny)

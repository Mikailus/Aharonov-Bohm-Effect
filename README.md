# Script for computing nonrelativistic probability density and nonrelativistic probability density current for Aharonov-Bohm Effect
1. Description
## The Aharonov-Bohm Effect
The Aharonov-Bohm Effect is quantum mechanical effect in which charged particle (eg. electron) moving in free field (without electric field **E** and magnetic field **B**) space deflects due to non-zero value of vector potential **A** which is very far from known behavior in classical mechanics or even quantum mechanics. Up to 1959, scalar potential V and vector potential **A** were considered as only mathematical objects which are useful for calculations but have **no** physical significance, ie. every physical problem could be solved and explained without use of potentials. As Aharonov and Bohm shown, one can consider an infinitely thin and infinitely long solenoid (very important property of this solenoid is that magnetic field is fully confined within) and particle moving in a space outside this solenoid. Particle cannot travel inside solenoid, so in a space possible for it, there is no magnetic field **B**. But... There is vector potential **A**! And it turned out that due to non zero value of vector potential **A**, particle deflects on this solenoid, despite it doesn't reach space with non-zero magnetic field **B** (fact of not reaching interior of solenoid is obtained due to zeroing of probability density).
## Probability density
Describes probability that in a measurement a particle will be located in determined place.
## Probability density current
Describes the flow of probability density. In classical limit it can be recognized as trajectory of a particle.
## Files
This project contains 4 files: 2 python files and 2 matlab files, which are optional to use and everyone needs only 2 python files.
### density.py
This script computes in parallel densities for particle in a space, creates pcolor plot of probability density for particle and saves results in files.
### current.py
This script computes in parallel density currents for particle in a space, creates streamline plot of probability density current for particle and saves results in files.
### draw_density.m
This script is **optional** and creates pcolor plot for given files (file with x coordinates, file with y coordinates, file with density array).
### draw_current.m
This script is **optional** and creates three different stream plots for given files (file with x coordinates, file with y coordinates, file with density array).
2. Prerequisities
User needs to have matplotlib, numpy and scipy libraries installed. If is also interested in using Matlab scripts, has to install Matlab.
3. Warning
Because of lot of calculations made during executing scripts, there is used parallel computing, and it is recommended to start with relatively small values (about 50-100) of number of divisions of x axis (nx) and number of divisions of y axis (ny) which are set in python scripts. The higher nx and ny, the more points have to be computed and the more time it takes!
4. Script parameters
alfa - parameter related to magnetic field
k - parameter related to energy of a particle
suma - number of elements to summarize for computing density or current
xmin - starting x axis value
xmax - finishing x axis value
nx - number of divisions of x axis
ymin - starting y axis value
ymax - finishing y axis value
ny - number of divisions of y axis
5. Link to Aharonov-Bohm paper
Paper can be downloaded from [here](https://journals.aps.org/pr/abstract/10.1103/PhysRev.115.485).
6. Authors
There is only one author - Michał Wajer.

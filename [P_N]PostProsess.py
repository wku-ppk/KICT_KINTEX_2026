get_ipython().magic('reset -sf') #command to initialize iPython environment

import itasca as it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
from scipy.integrate import cumtrapz

plt.close('all')


it.command("python-reset-state false") #the model new command will not affect python environment

print('start ==')

it.command("""
model new
model restore '[5]DynamicRun3D_T_Via4_1X30MPa.sav'
""")

## Construct Disp, Vel, Acc Matrix of Soil Layers
AccArray = np.array(it.history.get('freeFieldAcc-x', 'time'))
VelocityArray = np.array(it.history.get('freeFieldVel-x', 'time'))
DispArray = np.array(it.history.get('freeFieldDisp-x', 'time'))

def constArray(AccArray, historyName):
    tempCol = np.array(it.history.get(historyName))
    new_col = tempCol[:, 1].reshape(-1, 1)
    AccArray = np.hstack((AccArray, new_col))
    return AccArray

AccArray = constArray(AccArray, 'freeFieldAcc-y')
AccArray = constArray(AccArray, 'freeFieldAcc-z')

AccArray = constArray(AccArray, 'ticketHallMidSoilAcc-x')
AccArray = constArray(AccArray, 'ticketHallMidSoilAcc-y')
AccArray = constArray(AccArray, 'ticketHallMidSoilAcc-z')

AccArray = constArray(AccArray, 'upperShaftSoilAcc-x')
AccArray = constArray(AccArray, 'upperShaftSoilAcc-y')
AccArray = constArray(AccArray, 'upperShaftSoilAcc-z')

AccArray = constArray(AccArray, 'centerShaftSoilAcc-x')
AccArray = constArray(AccArray, 'centerShaftSoilAcc-y')
AccArray = constArray(AccArray, 'centerShaftSoilAcc-z')

AccArray = constArray(AccArray, 'lowerShaftSoilAcc-x')
AccArray = constArray(AccArray, 'lowerShaftSoilAcc-y')
AccArray = constArray(AccArray, 'lowerShaftSoilAcc-z')

VelocityArray = constArray(VelocityArray, 'freeFieldVel-y')
VelocityArray = constArray(VelocityArray, 'freeFieldVel-z')

VelocityArray = constArray(VelocityArray, 'ticketHallMidSoilVel-x')
VelocityArray = constArray(VelocityArray, 'ticketHallMidSoilVel-y')
VelocityArray = constArray(VelocityArray, 'ticketHallMidSoilVel-z')

VelocityArray = constArray(VelocityArray, 'upperShaftSoilVel-x')
VelocityArray = constArray(VelocityArray, 'upperShaftSoilVel-y')
VelocityArray = constArray(VelocityArray, 'upperShaftSoilVel-z')

VelocityArray = constArray(VelocityArray, 'centerShaftSoilVel-x')
VelocityArray = constArray(VelocityArray, 'centerShaftSoilVel-y')
VelocityArray = constArray(VelocityArray, 'centerShaftSoilVel-z')

VelocityArray = constArray(VelocityArray, 'lowerShaftSoilVel-x')
VelocityArray = constArray(VelocityArray, 'lowerShaftSoilVel-y')
VelocityArray = constArray(VelocityArray, 'lowerShaftSoilVel-z')

DispArray = constArray(DispArray, 'freeFieldDisp-y')
DispArray = constArray(DispArray, 'freeFieldDisp-z')

DispArray = constArray(DispArray, 'ticketHallMidSoilDisp-x')
DispArray = constArray(DispArray, 'ticketHallMidSoilDisp-y')
DispArray = constArray(DispArray, 'ticketHallMidSoilDisp-z')

DispArray = constArray(DispArray, 'upperShaftSoilDisp-x')
DispArray = constArray(DispArray, 'upperShaftSoilDisp-y')
DispArray = constArray(DispArray, 'upperShaftSoilDisp-z')

DispArray = constArray(DispArray, 'centerShaftSoilDisp-x')
DispArray = constArray(DispArray, 'centerShaftSoilDisp-y')
DispArray = constArray(DispArray, 'centerShaftSoilDisp-z')

DispArray = constArray(DispArray, 'lowerShaftSoilDisp-x')
DispArray = constArray(DispArray, 'lowerShaftSoilDisp-y')
DispArray = constArray(DispArray, 'lowerShaftSoilDisp-z')

time = AccArray[:, 0].reshape(-1, 1)  # Extract the time array # n x 1 column vector
acc_matrix = AccArray[:, 1:]  # Extract acceleration signals as a matrix (n x (m-1)) # n x (m-1) matrix
#vel_matrix = cumtrapz(acc_matrix, time, axis=0, initial=0) # Perform cumulative integration using matrix calculations # Each column of acc_matrix is integrated with respect to the time vector
#VelocityArray = np.hstack((time, vel_matrix)) # Combine the time column with the velocity matrix into a new array

#vel_matrix = VelocityArray[:, 1:]  # Extract acceleration signals as a matrix (n x (m-1)) # n x (m-1) matrix
#disp_matrix = cumtrapz(vel_matrix, time, axis=0, initial=0) # Perform cumulative integration using matrix calculations # Each column of acc_matrix is integrated with respect to the time vector
#DispArray = np.hstack((time, disp_matrix)) # Combine the time column with the velocity matrix into a new array

acc_g_matrix = acc_matrix/9.81
AccArray_g = np.hstack((time, acc_g_matrix))

## Construct Disp, Vel, Acc Matrix of Structural Members
strVelArray = np.array(it.history.get('ticketHallMidSlabVel-x', 'time')) # Velocity Array for Structures

strVelArray = constArray(strVelArray, 'ticketHallMidSlabVel-y')
strVelArray = constArray(strVelArray, 'ticketHallMidSlabVel-z')

strVelArray = constArray(strVelArray, 'upperShaftWallVel-x')
strVelArray = constArray(strVelArray, 'upperShaftWallVel-y')
strVelArray = constArray(strVelArray, 'upperShaftWallVel-z')

strVelArray = constArray(strVelArray, 'centerShaftWallVel-x')
strVelArray = constArray(strVelArray, 'centerShaftWallVel-y')
strVelArray = constArray(strVelArray, 'centerShaftWallVel-z')

strVelArray = constArray(strVelArray, 'lowerShaftWallVel-x')
strVelArray = constArray(strVelArray, 'lowerShaftWallVel-y')
strVelArray = constArray(strVelArray, 'lowerShaftWallVel-z')

strVelArray = constArray(strVelArray, 'mainPlatformWallVel-x')
strVelArray = constArray(strVelArray, 'mainPlatformWallVel-y')
strVelArray = constArray(strVelArray, 'mainPlatformWallVel-z')

str_time = strVelArray[:, 0] # Extract the time array
str_vel_matrix = strVelArray[:, 1:] # Extract the velocity signals (N x (M-1))
dt = np.diff(str_time).reshape(-1, 1)  # Compute the time differences (dt) for each time step # Reshape to make it an N-1 x 1 column vector

# Calculate the acceleration by differentiating the velocity
str_acc_matrix = np.diff(str_vel_matrix, axis=0) / dt # Since np.diff returns an array of length N-1, we need to divide by dt and pad the result
str_acc_matrix = np.vstack([np.zeros((1, str_vel_matrix.shape[1])), str_acc_matrix]) # To maintain the same shape as the original VelArray, we need to pad the first row with zeros (or some other value) # because np.diff reduces the dimension by 1
strAccArray = np.hstack((time.reshape(-1, 1), str_acc_matrix)) # Combine the time column with the acceleration matrix

strAccArray_g = np.hstack((time, str_acc_matrix/9.81))

# Calculate the disp of structural element
str_disp_matrix = cumtrapz(str_vel_matrix, time, axis=0, initial=0) # Perform cumulative integration using matrix calculations # Each column of acc_matrix is integrated with respect to the time vector
strDispArray = np.hstack((time, str_disp_matrix)) # Combine the time column with the velocity matrix into a new array

###############################################
### ARRAY EXPLANATION
### AccArray, VelArray, DispArray
### |0                      |1                      |2                      |3                       |4                      |5                      |6
### |time                   |freeField-x            |freeField-y            |freeField-z             |ticketHallMidSoilAcc-x |ticketHallMidSoilAcc-y |ticketHallMidSoilAcc-z
### |7                      |8                      |9                      |10                      |11                     |12                     |
### |upperShaftSoilAcc-x    |upperShaftSoilAcc-y    |upperShaftSoilAcc-z    |centerShaftSoilAcc-x    |centerShaftSoilAcc-y   |centerShaftSoilAcc-z   |
### |13                     |14                     |15                     |
### |lowerShaftSoilAcc-x    |lowerShaftSoilAcc-y    |lowerShaftSoilAcc-z    |
###############################################
### strAccArray, strVelArray, strDispArray
### |0                      |1                      |2                      |3                       |4                      |5                      |6
### |time                   |ticketHallMidSlabVel-x |ticketHallMidSlabVel-y |ticketHallMidSlabVel-z  |upperShaftWallVel-x    |upperShaftWallVel-y    |upperShaftWallVel-z
### |7                      |8                      |9                      |10                      |11                     |12                     |
### |centerShaftWallVel-x   |centerShaftWallVel-y   |centerShaftWallVel-z   |lowerShaftWallVel-x     |lowerShaftWallVel-y    |lowerShaftWallVel-z    |
### |13                     |14                     |15                     |
### |mainPlatformWallVel-x  |mainPlatformWallVel-y  |mainPlatformWallVel-z  |

def plotHistory(subPlotM, subPlotN, xLimit, xValue, yValues, legends, xTitle, yTitle, lineColors, opacities):
    ax1 = plt.subplot(gs[subPlotM, subPlotN])                                                      # This subplot takes first column of the first row

    overall_min_y = float('inf')
    overall_max_y = float('-inf')

    for yValue, legend, lineColor, opacity in zip(yValues, legends, lineColors, opacities):
        ax1.plot(xValue, yValue, label=legend, alpha=opacity, color=lineColor)

        overall_min_y = min(overall_min_y, np.min(yValue))
        overall_max_y = max(overall_max_y, np.max(yValue))

    ax1.set_xlim(0.0, xLimit)
    if xTitle == 'Y':
        ax1.set_xlabel('Time (s)', fontsize=20, fontweight='bold', family='Cambria', color='black', labelpad=20)
    if yTitle != 'N':
        ax1.set_ylabel(yTitle, fontsize=20, fontweight='bold', family='Cambria', color='black', labelpad=20)
    ax1.grid(False)
    ax1.legend(frameon=False, prop={'family': 'Verdana', 'size': 8, 'weight': 'bold'}, loc='upper right', shadow=True)
    ax1.tick_params(axis='both', direction='inout', length=10)

    ax1.set_ylim(ax1.get_yticks()[0], ax1.get_yticks()[-1]) # Update the Y-axis limits to coincide with major ticks

###############################################
### PLOT ITEMS
###############################################
### Acceleration Plot

lable_properties = FontProperties(family='Cambria', style='normal', size=18)

fig = plt.figure(figsize=(30, 12))                                                            # Set the figure size (width, height) in inches
gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1])        # Define the GridSpec with specific height and width ratios
gs.update(wspace=0.2, hspace=0.3)                                                             # Adjust the spacing between subplots # Adjust these values to increase/decrease spacing

plotHistory(0, 0, 10.0, AccArray[:,0], [AccArray_g[:,1]], ['freeField-X Acc.'], 'N', 'Acc. (g)', ['blue'], [1.0])
plotHistory(0, 1, 10.0, AccArray[:,0], [AccArray_g[:,2]], ['freeField-Y Acc.'], 'N', 'N', ['blue'], [1.0])
plotHistory(0, 2, 10.0, AccArray[:,0], [AccArray_g[:,3]], ['freeField-Z Acc.'], 'N', 'N', ['blue'], [1.0])

plotHistory(1, 0, 10.0, AccArray[:,0], [AccArray_g[:,4], strAccArray_g[:,1]], ['ticket Hall Mid. Soil-X Acc.', 'ticket Hall Mid. Structure-X Acc.'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 1, 10.0, AccArray[:,0], [AccArray_g[:,5], strAccArray_g[:,2]], ['ticket Hall Mid. Soil-Y Acc.', 'ticket Hall Mid. Structure-Y Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 2, 10.0, AccArray[:,0], [AccArray_g[:,6], strAccArray_g[:,3]], ['ticket Hall Mid. Soil-Z Acc.', 'ticket Hall Mid. Structure-Z Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(2, 0, 10.0, AccArray[:,0], [AccArray_g[:,7], strAccArray_g[:,4]], ['Shaft Upper Soil-X Acc.', 'Shaft Upper -X Acc.'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 1, 10.0, AccArray[:,0], [AccArray_g[:,8], strAccArray_g[:,5]], ['Shaft Upper Soil-Y Acc.', 'Shaft Upper -Y Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 2, 10.0, AccArray[:,0], [AccArray_g[:,9], strAccArray_g[:,6]], ['Shaft Upper Soil-Z Acc.', 'Shaft Upper -Z Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(3, 0, 10.0, AccArray[:,0], [AccArray_g[:,10], strAccArray_g[:,7]], ['Shaft Mid. Soil-X Acc.', 'Shaft Mid. -X Acc.'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 1, 10.0, AccArray[:,0], [AccArray_g[:,11], strAccArray_g[:,8]], ['Shaft Mid. Soil-Y Acc.', 'Shaft Mid. -Y Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 2, 10.0, AccArray[:,0], [AccArray_g[:,12], strAccArray_g[:,9]], ['Shaft Mid. Soil-Z Acc.', 'Shaft Mid. -Z Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(4, 0, 10.0, AccArray[:,0], [AccArray_g[:,13], strAccArray_g[:,10]], ['Shaft Lower Soil-X Acc.', 'Shaft Lower -X Acc.'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 1, 10.0, AccArray[:,0], [AccArray_g[:,14], strAccArray_g[:,11]], ['Shaft Lower Soil-Y Acc.', 'Shaft Lower -Y Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 2, 10.0, AccArray[:,0], [AccArray_g[:,15], strAccArray_g[:,12]], ['Shaft Lower Soil-Z Acc.', 'Shaft Lower -Z Acc.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(5, 0, 10.0, AccArray[:,0], [strAccArray_g[:,13]], ['Platform Wall -X Acc.'], 'Y', 'Acc. (g)', ['red'], [1.0])
plotHistory(5, 1, 10.0, AccArray[:,0], [strAccArray_g[:,14]], ['Platform Wall -Y Acc.'], 'Y', 'N', ['red'], [1.0])
plotHistory(5, 2, 10.0, AccArray[:,0], [strAccArray_g[:,15]], ['Platform Wall -Z Acc.'], 'Y', 'N', ['red'], [1.0])

plt.show()

### Velocity Plot

lable_properties = FontProperties(family='Cambria', style='normal', size=18)

fig = plt.figure(figsize=(30, 12))                                                            # Set the figure size (width, height) in inches
gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1])        # Define the GridSpec with specific height and width ratios
gs.update(wspace=0.2, hspace=0.3)                                                             # Adjust the spacing between subplots # Adjust these values to increase/decrease spacing

plotHistory(0, 0, 10.0, VelocityArray[:,0], [VelocityArray[:,1]], ['freeField-X Vel.'], 'N', 'Vel. (m/s)', ['blue'], [1.0])
plotHistory(0, 1, 10.0, VelocityArray[:,0], [VelocityArray[:,2]], ['freeField-Y Vel.'], 'N', 'N', ['blue'], [1.0])
plotHistory(0, 2, 10.0, VelocityArray[:,0], [VelocityArray[:,3]], ['freeField-Z Vel.'], 'N', 'N', ['blue'], [1.0])
 
plotHistory(1, 0, 10.0, VelocityArray[:,0], [VelocityArray[:,4], strVelArray[:,1]], ['ticket Hall Mid. Soil-X Vel.', 'ticket Hall Mid. Structure-X Vel.'], 'N', 'Vel. (m/s)', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 1, 10.0, VelocityArray[:,0], [VelocityArray[:,5], strVelArray[:,2]], ['ticket Hall Mid. Soil-Y Vel.', 'ticket Hall Mid. Structure-Y Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 2, 10.0, VelocityArray[:,0], [VelocityArray[:,6], strVelArray[:,3]], ['ticket Hall Mid. Soil-Z Vel.', 'ticket Hall Mid. Structure-Z Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(2, 0, 10.0, VelocityArray[:,0], [VelocityArray[:,7], strVelArray[:,4]], ['Shaft Upper Soil-X Vel.', 'Shaft Upper -X Vel.'], 'N', 'Vel. (m/s)', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 1, 10.0, VelocityArray[:,0], [VelocityArray[:,8], strVelArray[:,5]], ['Shaft Upper Soil-Y Vel.', 'Shaft Upper -Y Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 2, 10.0, VelocityArray[:,0], [VelocityArray[:,9], strVelArray[:,6]], ['Shaft Upper Soil-Z Vel.', 'Shaft Upper -Z Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(3, 0, 10.0, VelocityArray[:,0], [VelocityArray[:,10], strVelArray[:,7]], ['Shaft Mid. Soil-X Vel.', 'Shaft Mid. -X Vel.'], 'N', 'Vel. (m/s)', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 1, 10.0, VelocityArray[:,0], [VelocityArray[:,11], strVelArray[:,8]], ['Shaft Mid. Soil-Y Vel.', 'Shaft Mid. -Y Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 2, 10.0, VelocityArray[:,0], [VelocityArray[:,12], strVelArray[:,9]], ['Shaft Mid. Soil-Z Vel.', 'Shaft Mid. -Z Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(4, 0, 10.0, VelocityArray[:,0], [VelocityArray[:,13], strVelArray[:,10]], ['Shaft Lower Soil-X Vel.', 'Shaft Lower -X Vel.'], 'N', 'Vel. (m/s)', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 1, 10.0, VelocityArray[:,0], [VelocityArray[:,14], strVelArray[:,11]], ['Shaft Lower Soil-Y Vel.', 'Shaft Lower -Y Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 2, 10.0, VelocityArray[:,0], [VelocityArray[:,15], strVelArray[:,12]], ['Shaft Lower Soil-Z Vel.', 'Shaft Lower -Z Vel.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(5, 0, 10.0, VelocityArray[:,0], [strVelArray[:,13]], ['Platform Wall -X Vel.'], 'Y', 'Vel. (m/s)', ['red'], [1.0])
plotHistory(5, 1, 10.0, VelocityArray[:,0], [strVelArray[:,14]], ['Platform Wall -Y Vel.'], 'Y', 'N', ['red'], [1.0])
plotHistory(5, 2, 10.0, VelocityArray[:,0], [strVelArray[:,15]], ['Platform Wall -Z Vel.'], 'Y', 'N', ['red'], [1.0])

plt.show()

### Displacement Plot

lable_properties = FontProperties(family='Cambria', style='normal', size=18)

fig = plt.figure(figsize=(30, 12))                                                            # Set the figure size (width, height) in inches
gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1])        # Define the GridSpec with specific height and width ratios
gs.update(wspace=0.2, hspace=0.3)                                                             # Adjust the spacing between subplots # Adjust these values to increase/decrease spacing

plotHistory(0, 0, 10.0, DispArray[:,0], [DispArray[:,1]], ['freeField-X Disp.'], 'N', 'Disp. (m)', ['blue'], [1.0])
plotHistory(0, 1, 10.0, DispArray[:,0], [DispArray[:,2]], ['freeField-Y Disp.'], 'N', 'N', ['blue'], [1.0])
plotHistory(0, 2, 10.0, DispArray[:,0], [DispArray[:,3]], ['freeField-Z Disp.'], 'N', 'N', ['blue'], [1.0])

plotHistory(1, 0, 10.0, DispArray[:,0], [DispArray[:,4], strDispArray[:,1]], ['ticket Hall Mid. Soil-X Disp.', 'ticket Hall Mid. Structure-X Disp.'], 'N', 'Disp. (m)', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 1, 10.0, DispArray[:,0], [DispArray[:,5], strDispArray[:,2]], ['ticket Hall Mid. Soil-Y Disp.', 'ticket Hall Mid. Structure-Y Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 2, 10.0, DispArray[:,0], [DispArray[:,6], strDispArray[:,3]], ['ticket Hall Mid. Soil-Z Disp.', 'ticket Hall Mid. Structure-Z Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(2, 0, 10.0, DispArray[:,0], [DispArray[:,7], strDispArray[:,4]], ['Shaft Upper Soil-X Disp.', 'Shaft Upper -X Disp.'], 'N', 'Disp. (m)', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 1, 10.0, DispArray[:,0], [DispArray[:,8], strDispArray[:,5]], ['Shaft Upper Soil-Y Disp.', 'Shaft Upper -Y Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 2, 10.0, DispArray[:,0], [DispArray[:,9], strDispArray[:,6]], ['Shaft Upper Soil-Z Disp.', 'Shaft Upper -Z Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(3, 0, 10.0, DispArray[:,0], [DispArray[:,10], strDispArray[:,7]], ['Shaft Mid. Soil-X Disp.', 'Shaft Mid. -X Disp.'], 'N', 'Disp. (m)', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 1, 10.0, DispArray[:,0], [DispArray[:,11], strDispArray[:,8]], ['Shaft Mid. Soil-Y Disp.', 'Shaft Mid. -Y Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 2, 10.0, DispArray[:,0], [DispArray[:,12], strDispArray[:,9]], ['Shaft Mid. Soil-Z Disp.', 'Shaft Mid. -Z Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(4, 0, 10.0, DispArray[:,0], [DispArray[:,13], strDispArray[:,10]], ['Shaft Lower Soil-X Disp.', 'Shaft Lower -X Disp.'], 'N', 'Disp. (m)', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 1, 10.0, DispArray[:,0], [DispArray[:,14], strDispArray[:,11]], ['Shaft Lower Soil-Y Disp.', 'Shaft Lower -Y Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 2, 10.0, DispArray[:,0], [DispArray[:,15], strDispArray[:,12]], ['Shaft Lower Soil-Z Disp.', 'Shaft Lower -Z Disp.'], 'N', 'N', ['blue', 'red'], [1.0, 0.6])

plotHistory(5, 0, 10.0, DispArray[:,0], [strDispArray[:,13]], ['Platform Wall -X Disp.'], 'Y', 'Disp. (m)', ['red'], [1.0])
plotHistory(5, 1, 10.0, DispArray[:,0], [strDispArray[:,14]], ['Platform Wall -Y Disp.'], 'Y', 'N', ['red'], [1.0])
plotHistory(5, 2, 10.0, DispArray[:,0], [strDispArray[:,15]], ['Platform Wall -Z Disp.'], 'Y', 'N', ['red'], [1.0])

plt.show()


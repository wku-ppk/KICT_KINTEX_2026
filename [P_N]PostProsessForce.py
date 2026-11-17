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

## Construct ticketHallArray
ticketHallArray = np.array(it.history.get('ticketHallCeilMoment', 'time'))

def constArray(tempArray, historyName):
    tempCol = np.array(it.history.get(historyName))
    new_col = tempCol[:, 1].reshape(-1, 1)
    tempArray = np.hstack((tempArray, new_col))
    return tempArray

ticketHallArray = constArray(ticketHallArray, 'ticketHallWallMoment')
ticketHallArray = constArray(ticketHallArray, 'ticketHallCeilIPstress-XX')
ticketHallArray = constArray(ticketHallArray, 'ticketHallCeilIPstress-YY')
ticketHallArray = constArray(ticketHallArray, 'ticketHallWallIPstress-XX')
ticketHallArray = constArray(ticketHallArray, 'ticketHallWallIPstress-YY')

time = ticketHallArray[:, 0].reshape(-1, 1)                     # Extract the time array # n x 1 column vector
ticketHallForce_matrix = ticketHallArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
ticketHallForce_matrix_kPa = ticketHallForce_matrix/1000.0
ticketHallArray_kPa = np.hstack((time, ticketHallForce_matrix_kPa))

## Construct Force array for Structural Members
mainPlatFormArray = np.array(it.history.get('mainPlatFormCeilMoment', 'time'))
mainPlatFormArray  = constArray(mainPlatFormArray, 'mainPlatFormCeilIPstress-XX')
mainPlatFormArray  = constArray(mainPlatFormArray, 'mainPlatFormCeilIPstress-YY')

mainPlatFormForce_matrix = mainPlatFormArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
mainPlatFormForce_matrix_kPa = mainPlatFormForce_matrix/1000.0
mainPlatFormArray_kPa = np.hstack((time, mainPlatFormForce_matrix_kPa))

## Construct Force array for Structural Members : Shafe
upperShaftArray = np.array(it.history.get('upperShaftWallMoment', 'time'))
upperShaftArray  = constArray(upperShaftArray, 'upperShaftWallIPstress-XX')
upperShaftArray  = constArray(upperShaftArray, 'upperShaftWallIPstress-YY')

upperShaftForceForce_matrix = upperShaftArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
upperShaftForceForce_matrix_kPa = upperShaftForceForce_matrix/1000.0
upperShaftArray_kPa = np.hstack((time, upperShaftForceForce_matrix_kPa))

centerShaftArray = np.array(it.history.get('centerShaftWallMoment', 'time'))
centerShaftArray  = constArray(centerShaftArray, 'centerShaftWallIPstress-XX')
centerShaftArray  = constArray(centerShaftArray, 'centerShaftWallIPstress-YY')

centerShaftForceForce_matrix = centerShaftArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
centerShaftForceForce_matrix_kPa = centerShaftForceForce_matrix/1000.0
centerShaftArray_kPa = np.hstack((time, centerShaftForceForce_matrix_kPa))

lowerShaftArray = np.array(it.history.get('lowerShaftWallMoment', 'time'))
lowerShaftArray  = constArray(lowerShaftArray, 'lowerShaftWallIPstress-XX')
lowerShaftArray  = constArray(lowerShaftArray, 'lowerShaftWallIPstress-YY')

lowerShaftForceForce_matrix = lowerShaftArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
lowerShaftForceForce_matrix_kPa = lowerShaftForceForce_matrix/1000.0
lowerShaftArray_kPa = np.hstack((time, lowerShaftForceForce_matrix_kPa))

###############################################
### ARRAY EXPLANATION
### ticketHall
### |0                      |1                      |2                            |3                            |4                          |5                          |6
### |time                   |ticketHallCeilMoment   |ticketHallWallMoment         |ticketHallCeilPrincipalMax   |ticketHallCeilPrincipalMin |ticketHallWallPrincipalMax |ticketHallWallPrincipalMin

###############################################
### mainPlatForm
### |0                      |1                      |2                            |3                            |
### |time                   |mainPlatFormCeilMoment |mainPlatFormCeilPrincipalMax |mainPlatFormCeilPrincipalMin |
###############################################
### Shafts :// upper, center, lower
### |0                      |1                      |2                            |3                            |
### |time                   |----ShaftWallMoment    |----ShaftWallPrincipalMax    |----ShaftWallPrincipalMin   |

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
### Plot
lable_properties = FontProperties(family='Cambria', style='normal', size=18)

fig = plt.figure(figsize=(30, 12))                                                            # Set the figure size (width, height) in inches
gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1])                    # Define the GridSpec with specific height and width ratios
gs.update(wspace=0.2, hspace=0.3)                                                             # Adjust the spacing between subplots # Adjust these values to increase/decrease spacing

plotHistory(0, 0, 10.0, ticketHallArray[:,0], [ticketHallArray_kPa[:,1]], ['ticketHallCeilMoment'], 'N', 'Moment (kN-m)', ['blue'], [1.0])
plotHistory(0, 1, 10.0, ticketHallArray[:,0], [ticketHallArray_kPa[:,3]], ['ticketHallCeilIPStress-XX'], 'N', '$\sigma_xx$ (kPa)', ['blue'], [1.0])
plotHistory(0, 2, 10.0, ticketHallArray[:,0], [ticketHallArray_kPa[:,4]], ['ticketHallCeilIPStress-YY'], 'N', '$\sigma_yy$ (kPa)', ['blue'], [1.0])

plotHistory(1, 0, 10.0, ticketHallArray[:,0], [ticketHallArray_kPa[:,2]], ['ticketHallWallMoment'], 'N', 'Moment (kN-m)', ['blue'], [1.0])
plotHistory(1, 1, 10.0, ticketHallArray[:,0], [ticketHallArray_kPa[:,5]], ['ticketHallWallIPStress-XX'], 'N', '$\sigma_xx$ (kPa)', ['blue'], [1.0])
plotHistory(1, 2, 10.0, ticketHallArray[:,0], [ticketHallArray_kPa[:,6]], ['ticketHallWallIPStress-YY'], 'N', '$\sigma_yy$ (kPa)', ['blue'], [1.0])

plotHistory(2, 0, 10.0, mainPlatFormArray_kPa[:,0], [mainPlatFormArray_kPa[:,1]], ['mainPlatFormMoment'], 'N', 'Moment (kN-m)', ['blue'], [1.0])
plotHistory(2, 1, 10.0, mainPlatFormArray_kPa[:,0], [mainPlatFormArray_kPa[:,2]], ['mainPlatFormIPStress-XX'], 'N', '$\sigma_xx$ (kPa)', ['blue'], [1.0])
plotHistory(2, 2, 10.0, mainPlatFormArray_kPa[:,0], [mainPlatFormArray_kPa[:,3]], ['mainPlatFormIPStress-YY'], 'N', '$\sigma_yy$ (kPa)', ['blue'], [1.0])

plotHistory(3, 0, 10.0, upperShaftArray_kPa[:,0], [upperShaftArray_kPa[:,1]], ['upperShaftWallMoment'], 'N', 'Moment (kN-m)', ['blue'], [1.0])
plotHistory(3, 1, 10.0, upperShaftArray_kPa[:,0], [upperShaftArray_kPa[:,2]], ['upperShaftWallIPStress-XX'], 'N', '$\sigma_xx$ (kPa)', ['blue'], [1.0])
plotHistory(3, 2, 10.0, upperShaftArray_kPa[:,0], [upperShaftArray_kPa[:,3]], ['upperShaftWallIPStress-YY'], 'N', '$\sigma_yy$ (kPa)', ['blue'], [1.0])

plotHistory(4, 0, 10.0, centerShaftArray_kPa[:,0], [centerShaftArray_kPa[:,1]], ['centerShaftWallMoment'], 'N', 'Moment (kN-m)', ['blue'], [1.0])
plotHistory(4, 1, 10.0, centerShaftArray_kPa[:,0], [centerShaftArray_kPa[:,2]], ['centerShaftWallPrincipalMax'], 'N', '$\sigma_xx$ (kPa)', ['blue'], [1.0])
plotHistory(4, 2, 10.0, centerShaftArray_kPa[:,0], [centerShaftArray_kPa[:,3]], ['centerShaftWallIPStress-YY'], 'N', '$\sigma_yy$ (kPa)', ['blue'], [1.0])

plotHistory(5, 0, 10.0, lowerShaftArray_kPa[:,0], [lowerShaftArray_kPa[:,1]], ['lowerShaftWallMoment'], 'Y', 'Moment (kN-m)', ['blue'], [1.0])
plotHistory(5, 1, 10.0, lowerShaftArray_kPa[:,0], [lowerShaftArray_kPa[:,2]], ['lowerShaftWallIPStress-XX'], 'Y', '$\sigma_xx$ (kPa)', ['blue'], [1.0])
plotHistory(5, 2, 10.0, lowerShaftArray_kPa[:,0], [lowerShaftArray_kPa[:,3]], ['lowerShaftWallIPStress-YY'], 'Y', '$\sigma_yy$ (kPa)', ['blue'], [1.0])
plt.show()

######

## Construct Force array for Structural Members : Column
ticketHallColumnArray = np.array(it.history.get('column1-1AxialForce', 'time'))
ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-2AxialForce')
ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-3AxialForce')
ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-4AxialForce')

ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-1Moment')
ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-2Moment')
ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-3Moment')
ticketHallColumnArray  = constArray(ticketHallColumnArray, 'column1-4Moment')

time = ticketHallColumnArray[:, 0].reshape(-1, 1)                     # Extract the time array # n x 1 column vector
ticketHallColumnForce_matrix = ticketHallColumnArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
ticketHallColumnForce_matrix_kPa = ticketHallColumnForce_matrix/1000.0
ticketHallColumnArray_kPa = np.hstack((time, ticketHallColumnForce_matrix_kPa))

elShaftArray = np.array(it.history.get('elevatorShaft-1 axial force', 'time'))
elShaftArray  = constArray(elShaftArray, 'elevatorShaft-2 axial force')
elShaftArray  = constArray(elShaftArray, 'elevatorShaft-3 axial force')
elShaftArray  = constArray(elShaftArray, 'elevatorShaft-4 axial force')

elShaftArray  = constArray(elShaftArray, 'elevatorShaft-1 moment')
elShaftArray  = constArray(elShaftArray, 'elevatorShaft-2 moment')
elShaftArray  = constArray(elShaftArray, 'elevatorShaft-3 moment')
elShaftArray  = constArray(elShaftArray, 'elevatorShaft-4 moment')

time = elShaftArray[:, 0].reshape(-1, 1)                     # Extract the time array # n x 1 column vector
elShaftArrayForce_matrix = elShaftArray[:, 1:]                 # Extract Force signals as a matrix (n x (m-1)) # n x (m-1) matrix
elShaftArrayForce_matrix_kPa = elShaftArrayForce_matrix/1000.0
elShaftArray_kPa = np.hstack((time, elShaftArrayForce_matrix_kPa))

###############################################
### ARRAY EXPLANATION
### column / ticket Hall, elShaft
### |0                         |1                         |2                         |3                         |4                        |
### |time                      |ticketHallColumn-axial_1  |ticketHallColumn-axial_2  |ticketHallColumn-axial_3  |ticketHallColumn-axial_4 |
### |5                         |6                         |7                         |8                         |
### |ticketHallColumn-Moment_1 |ticketHallColumn-Moment_2 |ticketHallColumn-Moment_3 |ticketHallColumn-Moment_4 |

### Plot
lable_properties = FontProperties(family='Cambria', style='normal', size=18)

fig = plt.figure(figsize=(30, 12))                                                            # Set the figure size (width, height) in inches
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])                    # Define the GridSpec with specific height and width ratios
gs.update(wspace=0.2, hspace=0.3)                                                             # Adjust the spacing between subplots # Adjust these values to increase/decrease spacing

plotHistory(0, 0, 10.0, ticketHallColumnArray_kPa[:,0], [ticketHallColumnArray_kPa[:,1], ticketHallColumnArray_kPa[:,2], ticketHallColumnArray_kPa[:,3], ticketHallColumnArray_kPa[:,4]], \
            ['ticketHallColumn-Axial#1', 'ticketHallColumn-Axial#2', 'ticketHallColumn-Axial#3', 'ticketHallColumn-Axial#4'], 'N', 'Axial Force (kN)', ['blue', 'red', 'green', 'black'], [0.5, 0.5, 0.5, 0.5])

plotHistory(0, 1, 10.0, ticketHallColumnArray_kPa[:,0], [ticketHallColumnArray_kPa[:,5], ticketHallColumnArray_kPa[:,6], ticketHallColumnArray_kPa[:,7], ticketHallColumnArray_kPa[:,8]], \
            ['ticketHallColumn-Moment#1', 'ticketHallColumn-Moment#2', 'ticketHallColumn-Moment#3', 'ticketHallColumn-Moment#4'], 'N', 'Moment (kN-m)', ['blue', 'red', 'green', 'black'], [0.5, 0.5, 0.5, 0.5])

plotHistory(1, 0, 10.0, elShaftArray_kPa[:,0], [elShaftArray_kPa[:,1], elShaftArray_kPa[:,2], elShaftArray_kPa[:,3], elShaftArray_kPa[:,4]], \
            ['elShaft-Axial#1', 'elShaft-Axial#2', 'elShaft-Axial#3', 'elShaft-Axial#4'], 'Y', 'Axial Force (kN)', ['blue', 'red', 'green', 'black'], [0.5, 0.5, 0.5, 0.5])

plotHistory(1, 1, 10.0, elShaftArray_kPa[:,0], [elShaftArray_kPa[:,5], elShaftArray_kPa[:,6], elShaftArray_kPa[:,7], elShaftArray_kPa[:,8]], \
            ['elShaft-Moment#1', 'elShaft-Moment#2', 'elShaft-Moment#3', 'elShaft-Moment#4'], 'Y', 'Moment (kN-m)', ['blue', 'red', 'green', 'black'], [0.5, 0.5, 0.5, 0.5])
plt.show()


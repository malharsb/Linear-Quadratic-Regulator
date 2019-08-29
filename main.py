from BuggySimulator import *
import numpy as np
from controller import *
from util import *
import matplotlib.pyplot as plt
# from Evaluation_Task2 import *
from Evaluation import *


# get the trajectory
traj = get_trajectory('buggyTrace.csv')

#Get x_primt,y_prime,x_doubleprime and y_doubleprime
x_p = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,0],sigma=10,order=1)
x_pp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,0],sigma=10,order=2)
y_p = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,1],sigma=10,order=1)
y_pp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,1],sigma=10,order=2)

#Calculate curvature and put it into an array
curv=np.zeros(len(traj))
for i in range(len(x_p)):
    curv[i] = (x_p[i]*y_pp[i] - y_p[i]*x_pp[i])/(x_p[i]**2 + y_p[i]**2)**1.5

# initial the Buggy
vehicle = initail(traj, 0)

n = 5000
X = []
Y = []
delta = []
xd = []
yd = []
phi = []
phid = []
deltad = []
F = []
minDist =[]
'''
your code starts here
'''
# preprocess the trajectory
passMiddlePoint = False
nearGoal = False

cur_state = np.zeros(7)

for i in range(n):
    command,error = controller(traj,vehicle,curv,x_p,y_p)
    vehicle.update(command = command)
   
    # termination check
    disError,nearIdx = closest_node(vehicle.state.X, vehicle.state.Y, traj)
    stepToMiddle = nearIdx - len(traj)/2.0
    if abs(stepToMiddle) < 100.0:
        passMiddlePoint = True
        print('middle point passed')
    nearGoal = nearIdx >= len(traj)-50
    if nearGoal and passMiddlePoint:
        print('destination reached!')
        break
    

    # record states
    X.append(vehicle.state.X)
    Y.append(vehicle.state.Y)
    delta.append(vehicle.state.delta)
    xd.append(vehicle.state.xd)
    yd.append(vehicle.state.yd)
    phid.append(vehicle.state.phid)
    phi.append(vehicle.state.phi)
    deltad.append(command.deltad)
    F.append(command.F)
    minDist.append(disError)

    # to save the current states into a matrix
    cur_state = save_state(vehicle.state)

    if i == 0:
        state_saved = (np.matrix(cur_state)).reshape((1, 7))
    else:
        state_saved = np.concatenate((state_saved, (np.matrix(cur_state)).reshape((1, 7))), axis=0)


# save state
np.save('24-677_Project_4_BuggyStates_Malhar.npy', state_saved)

#Show Result
showResult(traj,X,Y,delta,xd,yd,F,phi,phid,minDist)

#Evaluate task
taskNum = 3
evaluation(minDist, traj, X, Y, taskNum)
#evaluation(minDist, traj, X, Y)
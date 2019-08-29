'''
This is a realization of the controller from Vehicle Dynamics and Control by Rajesh Rajamani.
Yaohui Guo
'''

'''
Team SpeedyGonzales Controller
'''
from BuggySimulator import *
import numpy as np
import scipy
import control
from scipy.ndimage import gaussian_filter1d
from util import *

def controller(traj,vehicle,curv,x_p,y_p):

	#Function for lqr
	def dlqr(A,B,Q,R):
		X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R)) 
		K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A)) 
		return K

	#Calculate the K matrix
	#A = np.asarray([[0,1,0,0],[0,-6,30,1.8],[0,0,0,1],[0,1.076,-5.383,-7.356]]) #error
	A = np.asarray([[0,1,0,0],[0,-6*(5/vehicle.state.xd),30,1.8*(5/vehicle.state.xd)],[0,0,0,1],[0,1.076*(5/vehicle.state.xd),-5.383,-7.356*(5/vehicle.state.xd)]])
	B = np.asarray([[0],[15],[0],[9.8684]]) #delta
	B_extra = np.asarray([[0],[-3.2],[0],[-7.356]]) # psi_dot_desired which is xdot/R
	C = np.asarray([1,0,0,0])
	D = np.asarray([0])

	Ad,Bd,Cd,Dd,dtd = scipy.signal.cont2discrete((A,B,C,D),0.05)
	Q = np.identity(4)
	R = 1

	#Final K (gains):
	K = dlqr(Ad,Bd,Q,R)

	#Extract the vehicle current state
	x_act = vehicle.state.X
	y_act = vehicle.state.Y

	curv_list=[]

	
	dist, index = closest_node(x_act, y_act, traj) #Get distance and closest point
	print(index)

	# la = Lookahead (Define the lookahead)
	if index<8100:
		la=100
	else:
		la=10

	#Get psi desired
	psi_des = np.arctan2((traj[index+la][1]-y_act),(traj[index+la][0]-x_act))
	#psi_des = np.arctan2((traj[index+50][1] - traj[index+30][1]),(traj[index+50][0] - traj[index+30][0]))
	#psi_des = np.arctan2(y_p[index+la],x_p[index+la])
	error = np.zeros(4)

	#Error e1:
	error[0] = (y_act - traj[index+la][1])*np.cos(psi_des) - (x_act - traj[index+la][0])*np.sin(psi_des)

	#Error e2:
	error[2] = wrap2pi((vehicle.state.phi) - psi_des)

	#Error e1_dot:
	error[1] = vehicle.state.yd + vehicle.state.xd*(error[2])

	#curv = np.absolute((np.average(x_p)*np.average(y_pp) - np.average(y_p)*np.average(x_pp))/((np.average(x_p**2) + np.average(y_p**2))**(1.5)))
	#psid_des = vehicle.state.xd*np.average(curv[index+la-20:index+la+20])
	psid_des = vehicle.state.xd*np.average(curv[index+la-1:index+la+1])

	#Take an average of the curves ahead
	curv_avg = np.average(curv[index+la-50:index+la+50])

	#error e2_dot:
	error[3] = vehicle.state.phid - psid_des

	#Covert to matrix form:
	error = np.transpose(np.matrix(error))

	#Calculate Delta
	delta = float(-K@error)

	#Compute delta_dot:
	deltad = (delta - vehicle.state.delta)/0.05

	#Set desired velocity based on the closest point on the trajectory
	if index<1200:
		v_des=35
	elif (index>2400) and (index<5000):
		v_des=16
	elif (index>3500) and (index<5400):
		v_des=16
	elif index>5800 and index<6000:
		v_des=12
	elif index>6000 and index<7700:
		v_des=10.5
	elif index>7800:
		v_des=35
	else:
		v_des=6.5

	#Bang-bang controller based on desired velocity
	if vehicle.state.xd>v_des:
		F= -10000
	else:
		F= 5000

	command1=vehicle.command(F,deltad)

	return command1, np.transpose(error)





	


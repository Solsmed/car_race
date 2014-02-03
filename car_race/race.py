from pylab import *

import car
import optimal_car
import track
import numpy as np

ion()

# This function trains a car in a track. 
# Your car.py class should be able to be accessed in this way.
# You may call this file by:

# import race
# final_car = race.train_car()
# race.show_race(final_car)

def train_car(maserati):
    
    close('all')

    print_log = False
    dynamicEpsilon = False
    doOptimalActionAnalysis = False;
    useOptimalCar = False

    n_trials = 1000
    if dynamicEpsilon or doOptimalActionAnalysis:
        numCars = 10
    else:
        numCars = 10
    epsilons = array([0.1]) #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    if dynamicEpsilon:
        epsilons = array([-1, -5, -9]) # parameters for epsilon function

    #######################################################################################################################################

    numEpsilons = epsilons.size
    n_time_steps = 1000  # maximum time steps for each trial
    timeLog = NaN * zeros((numCars,numEpsilons,n_trials))
    rewardLog = NaN * zeros((numCars,numEpsilons,n_trials))
    optimalActionsInterval = 20 # number of trials between each rendition of optimal actions
    optimalActionsResolution = 100; # density of evaluation coordinates
    optimalActions = NaN * zeros((ceil(n_trials / optimalActionsInterval),optimalActionsResolution,optimalActionsResolution))
    # create instances of a car and a track
    monaco = track.track()
    
    for c in arange(numCars):
        print "Starting sim of car ", c, "of", numCars
        for e in arange(numEpsilons):
            print "Epsilon ", e, "of", numEpsilons
            if maserati == None:
                if useOptimalCar:
                    print "Using optimal car"
                    ferrari = optimal_car.optimal_car()
                else:
                    print "Using standard car"
                    ferrari = car.car()
            else:
                print "Using loaded car"
                ferrari = maserati

            monaco.ferrari = ferrari
            #ferrari.set_epsilon(0.1)
            for j in arange(n_trials):

                # before every trial, reset the track and the car.
                # the track setup returns the initial position and velocity. 
                (position_0, velocity_0) = monaco.setup()	
                ferrari.reset()
                
                if not useOptimalCar:
                    if dynamicEpsilon:
                        ferrari.set_epsilon( exp(epsilons[e] * double(j)/n_trials) )# t = 0
                    else:
                        ferrari.set_epsilon( epsilons[e] )
                # choose a firsttion
                action = ferrari.choose_action(position_0, velocity_0, 0, print_results = print_log)
               
                # iterate over time
                for i in arange(n_time_steps) :	
                    #if dynamicEpsilon:
                    #    ferrari.set_epsilon( exp(epsilons[e] * ((double(i)/n_time_steps))) )
                    #else:
                    #    ferrari.set_epsilon( epsilons[e] )
                    # the track receives which action was taken and 
                    # returns the new position and velocity, and the reward value.
                    (position, velocity, R) = monaco.move(action)	
                   
                    # the car chooses a new action based on the new states and reward, and updates its parameters
                    action = ferrari.choose_action(position, velocity, R, print_results = print_log)
                   
                    # monaco.plot_world()
                   
                    # check if the race is over
                    if monaco.finished is True:
                       break

                    # if j == 30 and i >= 0:
                    #    print_log = True

                if monaco.finished:
                    timeLog[c][e][j] = i
                else:
                    timeLog[c][e][j] = NaN

                rewardLog[c][e][j] = monaco.total_reward

                if j%10 == 0:
                   monaco.plot_world()

                if (doOptimalActionAnalysis and (j%optimalActionsInterval == optimalActionsInterval - 1)):
                    coords = linspace(0+1/(2*optimalActionsResolution),1-1/(2*optimalActionsResolution),optimalActionsResolution)
                    for ix in arange(optimalActionsResolution):
                        for iy in arange(optimalActionsResolution):
                            px = coords[ix]
                            py = coords[iy]
                            (optAction, Q) = ferrari.actionMaxQ(ferrari.posCellsActivity((px, py)), ferrari.velCellsActivity((0,0)))
                            optimalActions[floor(j/optimalActionsInterval)][ix][iy] = optAction

                print 'Trial:', j

                #print 'Q policy stats: epsilon = ', ferrari.epsilon , 'epsilon ~=', (ferrari.pc_rand/(ferrari.pc_Qmax+ferrari.pc_rand)), 'e', e, ' epsilons[e]', epsilons[e]

            if (numEpsilons + numCars > 2):
                print "Car", c, ", epislon ",e ," times and rewards:"
                print timeLog[c][e]
                print rewardLog[c][e]

    if doOptimalActionAnalysis:
        np.savetxt("optimalActions.log",optimalActions.flatten())

    if (numEpsilons + numCars > 2):
        np.savetxt("timeLogs.log",timeLog.flatten())
        np.savetxt("rewardLogs.log",rewardLog.flatten())

    print ferrari.best_actions[0]
    print ferrari.best_actions[1]
    print ferrari.best_actions[2]

    return ferrari #returns a trained car
    
# This function shows a race of a trained car, with learning turned off
def show_race(ferrari):

    close('all')

    print_log = True

    # create instances of a track
    monaco = track.track()
    monaco.ferrari = ferrari

    n_time_steps = 1000  # maximum time steps
    
    # choose to plot every step and start from defined position
    (position_0, velocity_0) = monaco.setup(plotting=True)	
    ferrari.reset()

    # choose a first action
    action = ferrari.choose_action(position_0, velocity_0, 0, learn=False)

    # iterate over time
    for i in arange(n_time_steps) :	

        # informyour action
        (position, velocity, R) = monaco.move(action)	

        # choose new action, with learning turned off
        action = ferrari.choose_action(position, velocity, R, learn=False)

        # check if the race is over
        if monaco.finished is True:
            break


import race
import car
import optimal_car
import numpy as np

doTraining = True
doLoad = False
doSave = False
fileName = 'autosave_weights'



maserati = car.car()

if doLoad:
	#maserati.weights = np.loadtxt(fileName)
	#maserati = race.train_car(maserati)
    #maserati.best_actions.append([1, 1, 2, 2, 1, 2, 0, 6, 6, 3, 5, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 5]) # solsmed, Time: 21
    #maserati.best_actions.append([1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 3, 5, 5, 5, 5, 5, 5]) # perfectly positioned crash, Time: 22
    #maserati.best_actions.append([1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 5, 4, 4, 4, 5, 4, 4, 4, 4, 5, 5, 4]) # triangle crash, Time: 23
    #maserati.best_actions.append([1, 1, 3, 1, 6, 1, 3, 6, 3, 6, 3, 6, 1, 1, 3, 6, 6, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5]) # learned, crash free, Time: 27
    maserati.best_actions.append([1, 2, 2, 2, 7, 2, 7, 2, 2, 2, 7, 7, 3, 3, 3, 4, 4, 4, 5, 5, 5, 4, 5, 5]) # epsilon 0.01 ?

if doTraining:
	if doLoad:
		maserati = race.train_car(maserati)
	else:	
		maserati = race.train_car(None)
	print 'Training finished.'
	
	if doSave:
		np.savetxt(fileName, maserati.weights)

race.show_race(maserati)


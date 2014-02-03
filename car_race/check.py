from pylab import *
import track
import car

monaco = car.car()

p = [0.5,0.5]
v = [0.3,0.8]

pos = monaco.posCellsActivity(p)
vel = monaco.velCellsActivity(v)

figure(200)
clf()
im1 = imshow(vel)

draw()




            
        
        

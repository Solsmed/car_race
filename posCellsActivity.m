function [ A ] = posCellsActivity( p )

gridDist = 1/30;
sigmaP = 1/30;

[Ax, Ay] = meshgrid(0:gridDist:1,0:gridDist:1);

A = exp(-((p(1) - Ax).^2 + (p(2) - Ay).^2)/(2*sigmaP));

end


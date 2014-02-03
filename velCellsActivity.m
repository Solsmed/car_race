function [ A ] = velCellsActivity( v )

gridDist = 0.2;
sigmaP = 0.2;

[Ax, Ay] = meshgrid(-1:gridDist:1,-1:gridDist:1);

A = exp(-((v(1) - Ax).^2 + (v(2) - Ay).^2)/(2*sigmaP));

end


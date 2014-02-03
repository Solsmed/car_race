function [ Q ] = neuralQ( v, p, w )

numActions = size(w, 1);
Q = nan(numActions, 1);

posAct = posCellsActivity(p);
velAct = velCellsActivity(v);
Ap = posAct(:)';
numStatesV = numel(Ap);
Av = velAct(:)';
for a=1:numActions
    Q = sum(Av.*w(a,1:numStatesV)) + ...
        sum(Ap.*w(a,numStatesV+1:end));
end

end


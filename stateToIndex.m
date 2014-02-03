function [ ind, indx, indy ] = stateToIndex( state, activityFunction )

A = activityFunction(state);
[~, i] = max(A(:));
ind = i;
indx = floor((i - 1) / size(A,2)) + 1;
indy = (i - 1) - size(A,2)*floor((i - 1) / size(A,1)) + 1;

end


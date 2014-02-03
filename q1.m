%% SETUP
numActions = 9;
numStatesV = 11^2;
numStatesP = 31^2;

w = rand(numActions, numStatesV + numStatesP);

d = nan(numActions, 2);
d(2:numActions,:) = [cos(-2*pi*(1:numActions-1)/(numActions-1) + pi/2); sin(-2*pi*(1:numActions-1)/(numActions-1) + pi/2)]';
d(1,:) = [0, 0];

numEpisodes = 1000;
maxNumSteps = 300;
p_hist = cell(numEpisodes, 1);

epsilon = 0.1;
eta = 0.005;
gamma = 0.95;
lambda = 0.95;

goalPosition = [0.8, 0.8];
goalRadius = 0.05;

%% SIMULATION

a = 1;
Q = zeros(numStatesV, numStatesP, numActions);

for episode = 1:numEpisodes
    % Create history for episode
    p_hist{episode} = nan(maxNumSteps, length(p));
    
    % Initialise state for episode
    v = [0, 0];
    p = [0, 0];
    % Discretise
    vD = stateToIndex(v, velCellsActivity);
    pD = stateToIndex(p, posCellsActivity);
       
    % Initialise action according to policy
    if (rand >= 1 - epsilon)
        a = find(Q(vD,pD,:) == max(Q(vD,pD,:)));
    else
        a = randi(numActions);
    end

    % Initialise eligibility trace
    e = zeros(numStatesV, numStatesP, numActions);
    
    for t = 1:maxNumSteps
        % Take action a, get new state: vNew, pNew
        vNew = v + d(a);
        vNew(vNew < -1) = -1;
        vNew(vNew > 1) = 1;
        pNew = p + v;
        pNew(pNew < 0) = 0;
        pNew(pNew > 1) = 1;
        
        % Get reward
        if (norm(pNew - goalPosition) < goalRadius)
            r = 1000;
        else
            r = 0;
        end
        
        % Choose action in s' according to policy
        vDnew = stateToIndex(v, @velCellsActivity);
        pDnew = stateToIndex(p, @posCellsActivity);
        if (rand >= 1 - epsilon)
            aNew = find(Q(vDnew, pDnew, :) == max(Q(vDnew, pDnew, :)));
        else
            aNew = randi(numActions);
        end
                
        % Neural network Q update
        Qcur = neuralQ(v, p, w);
        Qnew = neuralQ(vNew, pNew, w);
        
        delta = r - (Qcur - gamma * Qnew);
        E = 1/2 * delta^2;

        e(v, p, a) = e(v, p, a) + 1;

        for vv = 1:numStatesV
            for pp = 1:numStatesP
                for aa = 1:numActions
                    Q(vv, pp, aa) = neuralQ(vv, pp, aa) + eta * delta * e(vv, pp, aa);
                    e(vv, pp, aa) = lambda * gamma * e(vv, pp, aa);
                end
            end
        end
        
        ApNew = posCellsActivity(pNew);
        AvNew = velCellsActivity(vNew);
        phiNew = [ApNew(:) AvNew(:)];
        Ap = posCellsActivity(p);
        Av = velCellsActivity(v);
        phi = [Ap(:) Av(:)];
        
        dEdw(v, p, ) = delta * (gamma*(i == aNew)*phiNew + (i == a)*phi);
        
        w(:,:) = eta * delta * e?;
        
        % Set current state to new state
        p = pNew;
        v = nVew;
        p_hist(t,:) = p; % record position
    end
end
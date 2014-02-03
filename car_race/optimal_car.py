from pylab import *

# this is a dummy class, use it as template inserting your algorithm.

class optimal_car:
    
    def __init__(self):

        # setup your parameters here.
        
        self.n_actions = 9
        self.n_vel_size = 11
        self.n_pos_size = 31
        self.n_states_vel = self.n_vel_size**2
        self.n_states_pos = self.n_pos_size**2
        self.n_neurons = self.n_states_vel + self.n_states_pos
        
        self.previous_Q = 0
        self.previous_action = 0

        self.epsilon = 0.1
        self.eta = 0.005
        self.gamma = 0.95
        self.llambda = 0.95

        self.weights = zeros((self.n_actions, self.n_neurons))

        self.velAct = zeros(self.n_states_vel)
        self.posAct = zeros(self.n_states_pos)

        self.previous_velAct = zeros(self.n_states_vel)
        self.previous_posAct = zeros(self.n_states_pos)

        self.eligibility_trace = zeros((self.n_actions, self.n_neurons))
        self.time = 0
        self.trial = 0

        self.current_action_history = list()
        self.best_actions = list()
        self.record_pool_size = 10

    def reset(self) :
        self.current_action_history = list()
        self.eligibility_trace = zeros((self.n_actions, self.n_neurons))
        self.time = 0
        self.trial = self.trial + 1.0

        baseRandom = 0.03
        self.epsilon = baseRandom + (1-baseRandom)*exp(-9 * self.trial/1000)

    def set_epsilon(self, newEpsilon) :
        self.epsilon = newEpsilon

    def velCellsActivity(self,v) :
        gridCount = self.n_vel_size
        sigmaP = 2./(self.n_vel_size - 1)
        [Ax, Ay] = meshgrid(linspace(-1,1,gridCount),linspace(-1,1,gridCount))
        A = exp(-((v[0] - Ax)**2 + (v[1] - Ay)**2)/(2*sigmaP**2))
        return A

    def posCellsActivity(self,p) :
        gridCount = self.n_pos_size
        sigmaP = 1./(self.n_pos_size - 1)
        [Ax, Ay] = meshgrid(linspace(0,1,gridCount),linspace(0,1,gridCount))
        A = exp(-((p[0] - Ax)**2 + (p[1] - Ay)**2)/(2*sigmaP**2))
        return A

    def calculate_Q(self, posAct, velAct, action) :
        return sum(self.weights[action,:self.n_states_vel] * velAct.flatten()) + sum(self.weights[action,self.n_states_vel:] * posAct.flatten())

    def actionMaxQ(self, posAct, velAct) :
        Q = zeros((self.n_actions,1))
        for a in arange(self.n_actions):
            Q[a] = self.calculate_Q(posAct, velAct, a) #sum(self.weights[a,:self.n_states_vel] * self.velAct.flatten()) + sum(self.weights[a,self.n_states_vel:] * self.posAct.flatten())
        action = Q.argmax(0)[0]
        Q = Q[action][0]

        return action, Q

    def decideAction(self, posAct, velAct, epsilon) :
        if (rand() <= 1 - epsilon):
            (action, Q) = self.actionMaxQ(posAct, velAct)
        else:
            action = floor(rand()*self.n_actions)
            # if learn:
            Q = self.calculate_Q(posAct, velAct, action) #sum(self.weights[action,:self.n_states_vel] * self.velAct.flatten()) + sum(self.weights[action,self.n_states_vel:] * self.posAct.flatten())

        return action, Q

    def update_eligibility_trace(self) :
        self.eligibility_trace = self.gamma * self.llambda * self.eligibility_trace
        self.eligibility_trace[self.previous_action,:self.n_states_vel] = self.eligibility_trace[self.previous_action,:self.n_states_vel] + self.previous_velAct.flatten()
        self.eligibility_trace[self.previous_action,self.n_states_vel:] = self.eligibility_trace[self.previous_action,self.n_states_vel:] + self.previous_posAct.flatten()

    def update_states(self, action) :
        self.previous_action = action
        self.previous_velAct = self.velAct
        self.previous_posAct = self.posAct

    def choose_action(self, position, velocity, R, learn = True, debug = False):

        if debug:
            print '\n'
            print 'Reward:', R, "   p':", position, "   v':", velocity

        Q = None

        if learn:
            t = self.time / 1000.0
            stress_factor = exp(-5 * t)

            if R > 0 :
                self.best_actions.append(self.current_action_history)
                self.current_action_history = list()

                self.best_actions = sorted(self.best_actions, key=lambda actlist: len(actlist)) # sorts action sequence length, increasing

                if len(self.best_actions) > self.record_pool_size:
                    self.best_actions.pop() # removes last element in list
                
                if debug:
                    print "number of records: ", len(self.best_actions), " best=", len(self.best_actions[0]), "worst=", len(self.best_actions[len(self.best_actions) - 1])

                R = R + (4*R)*stress_factor
            if R < 0:
                R = R

        self.velAct = self.velCellsActivity(velocity)
        self.posAct = self.posCellsActivity(position)

        if learn:
            (action, Q) = self.decideAction(self.posAct, self.velAct, self.epsilon)
        else:
            if self.time < len(self.best_actions[0]):
                action = self.best_actions[0][self.time]
            else:
                (action, Q) = self.decideAction(self.posAct, self.velAct, 0)

        if learn:
            delta = R - self.previous_Q + (self.gamma * Q)
            self.update_eligibility_trace()            
            self.weights += self.eta * delta * self.eligibility_trace
            self.update_states(action)         
            self.previous_Q = sum(self.weights[self.previous_action,:self.n_states_vel] * self.previous_velAct.flatten()) + sum(self.weights[self.previous_action,self.n_states_vel:] * self.previous_posAct.flatten())

        self.time += 1
        self.current_action_history.append(action)

        return action




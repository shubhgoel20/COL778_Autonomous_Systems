import heapq
import os
import pickle
import time
import numpy as  np
from utils import Domain, PriorityQueue
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import csv

################################## Utilities ####################################################

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

class Domain:
    def __init__(self, grid, p = 0.8, gamma = 0.9, living_reward = 0, hole_reward = 0, goal_reward = 1, snow = False):
        self.p = p
        self.gamma = gamma
        self.living_reward = living_reward
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.grid = grid
        self.snow = snow
    
    def transition_prob(self, curr_state, action, next_state):
        d = {'r':(0,1), 'd':(1,0), 'l':(0,-1), 't':(-1,0)}
        x_curr,y_curr = curr_state
        r,c = self.grid.shape
        x_int = max(min(x_curr+d[action][0],r-1),0)
        y_int = max(min(y_curr+d[action][1],c-1),0)
        pr = {}
        if(self.snow):
            opp = {'r':'l','d':'t','l':'r','t':'d'}
            pr[(x_int,y_int)] = 1/3
            for key,value in d.items():
                if(key == action):
                    continue
                x = max(min(x_curr+value[0],r-1),0)
                y = max(min(y_curr+value[1],c-1),0)
                if pr.get((x,y)) is None:
                    if(key == opp[action]):
                        pr[(x,y)] = 0
                    else:
                        pr[(x,y)] = 1/3
                else:
                    if(key == opp[action]):
                        pr[(x,y)] += 0
                    else:
                        pr[(x,y)] += 1/3
            return pr[next_state]

        pr[(x_int,y_int)] = self.p

        for key,value in d.items():
            if(key == action):
                continue
            x = max(min(x_curr+value[0],r-1),0)
            y = max(min(y_curr+value[1],c-1),0)
            if pr.get((x,y)) is None:
                pr[(x,y)] = (1 - self.p)/3
            else:
                pr[(x,y)] += (1 - self.p)/3

        return pr[next_state]
    
    def get_reward(self,state):
        x,y = state

        if(self.grid[x][y] == 'F' or self.grid[x][y] == 'S'):
            return self.living_reward
        elif (self.grid[x][y] == 'H'):
            return self.hole_reward
        else:
            return self.goal_reward
    
    def get_state_actions(self,state):
        x,y = state
        r,c = self.grid.shape

        states = list(set([(x,min(y+1,c-1)),(min(x+1,r-1),y),(x,max(y-1,0)),(max(x-1,0),y)]))
        actions = ['r','d','l','t']

        return states, actions
    

def run(algo, large_domain, large_grid, small_domain, small_grid, eps = 1e-6):
    name = os.path.join("partA",algo.__name__)
    if not os.path.exists(name):
        os.makedirs(name)
    save_policy_large = os.path.join(name,'policy_large.pkl')
    save_util_large = os.path.join(name,'util_large.pkl')
    save_policy_small = os.path.join(name,'policy_small.pkl')
    save_util_small = os.path.join(name,'util_small.pkl')
    save_ss_util_large = os.path.join(name,'ss_util_large.pkl')
    save_ss_util_small = os.path.join(name,'ss_util_small.pkl')
    #Large Map
    start = time.time()
    p_large,V_large, num_cycles_large, ss_util_large = algo(large_domain, large_grid, eps)
    end = time.time()
    time_taken_large = (end - start)*1000
    #Small Map
    start = time.time()
    p_small,V_small, num_cycles_small, ss_util_small = algo(small_domain, small_grid, eps)
    end = time.time()
    time_taken_small= (end - start)*1000
    print(name,':')
    print('Wall Time Large:', time_taken_large)
    print('Num Iterations Large:', num_cycles_large)
    print('Wall Time small:', time_taken_small)
    print('Num Iterations small:', num_cycles_small)

    with open(save_policy_large, 'wb') as f:
        pickle.dump(p_large, f)

    with open(save_util_large, 'wb') as f:
        pickle.dump(V_large, f)

    with open(save_policy_small, 'wb') as f:
        pickle.dump(p_small, f)

    with open(save_util_small, 'wb') as f:
        pickle.dump(V_small, f) 

    with open(save_ss_util_large, 'wb') as f:
        pickle.dump(ss_util_large, f)
    with open(save_ss_util_small, 'wb') as f:
        pickle.dump(ss_util_small, f)


def run2(algo, domain,grid,name,eps = 1e-6, dir = "./"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    policy_save = os.path.join(dir,'policy_'+ name +'.pkl')
    util_save = os.path.join(dir,'util_'+ name +'.pkl')
    p,V, num_cycles, ss_util = algo(domain, grid, eps)
    with open(policy_save, 'wb') as f:
        pickle.dump(p, f)

    with open(util_save, 'wb') as f:
        pickle.dump(V, f)

################################## Utilities ####################################################

################################## MDP Functions####################################################
        
def ValueIteration(domain:Domain, grid, eps):
    curr_V = np.zeros(grid.shape)
    V = np.zeros(grid.shape)
    r,c = grid.shape
    ss_util = []
    ss_util.append(curr_V[grid == 'S'])
    num_cycles = 0
    while True:
        delta = 0.0
        for i in range(r):
            for j in range(c):
                if(grid[i][j] == 'G'):
                    V[i][j] = domain.goal_reward
                    continue
                if(grid[i][j] == 'H'):
                    V[i][j] = domain.hole_reward
                    continue
                next_states, actions = domain.get_state_actions((i,j))

                best_utility = -1.0e9
                for a in actions:
                    temp = 0.0
                    for s in next_states:
                        if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                            temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*curr_V[s[0]][s[1]])
                        else:
                            temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                    best_utility = max(best_utility,temp)
                V[i][j] = best_utility
                delta = max(delta,abs(V[i][j] - curr_V[i][j]))
        
        curr_V = deepcopy(V)
        ss_util.append(curr_V[grid == 'S'])
        num_cycles+=1
        if((num_cycles%50) == 0):
            print("completed",num_cycles,"cycles, delta =",delta)
        if(delta < eps*((1-domain.gamma)/domain.gamma)):
            print("completed",num_cycles,"cycles, delta =",delta)
            break
    
    policy = [['_' for j in range(c)] for i in range(r)]

    for i in range(r):
        for j in range(c):
            if(grid[i][j] == 'G'):
                continue
            if(grid[i][j] == 'H'):
                continue
            next_states, actions = domain.get_state_actions((i,j))
            best_action = 'r'
            best_utility = -1

            for a in actions:
                temp = 0.0
                for s in next_states:
                    if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*curr_V[s[0]][s[1]])
                    else:
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                if(temp > best_utility):
                    best_utility = temp
                    best_action = a
            
            policy[i][j] = best_action

    return policy, curr_V, num_cycles, ss_util

def policy_evaluation(domain:Domain, policy, grid,eps):
    curr_V = np.zeros(grid.shape)
    V = np.zeros(grid.shape)
    r,c = grid.shape

    while True:
        delta = 0.0
        for i in range(r):
            for j in range(c):
                if(grid[i][j] == 'G'):
                    V[i][j] = domain.goal_reward
                    continue
                if(grid[i][j] == 'H'):
                    V[i][j] = domain.hole_reward
                    continue
                next_states,_= domain.get_state_actions((i,j))

                temp = 0.0
                for s in next_states:
                    if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                        temp+= domain.transition_prob((i,j),policy[i][j],s)*(domain.get_reward(s) + domain.gamma*curr_V[s[0]][s[1]])
                    else:
                        temp+= domain.transition_prob((i,j),policy[i][j],s)*(domain.get_reward(s))
                V[i][j] = temp
                delta = max(delta,abs(V[i][j] - curr_V[i][j]))
        
        curr_V = deepcopy(V)
        if(delta < eps):
            break
    return curr_V

def PolicyIteration(domain:Domain, grid, eps):
    r,c = grid.shape
    policy = [['_' for j in range(c)] for i in range(r)]

    for i in range(r):
        for j in range(c):
            if(grid[i][j] == 'G'):
                continue
            if(grid[i][j] == 'H'):
                continue
            _,actions = domain.get_state_actions((i,j))
            n = len(actions) - 1
            idx = random.randint(0,n)
            policy[i][j] = actions[idx]
    
    curr_V = policy_evaluation(domain,policy,grid,eps)
    ss_util = []
    ss_util.append(curr_V[grid == 'S'])

    num_cycles = 0
    while True:
        #update the current policy
        for i in range(r):
            for j in range(c):
                if(grid[i][j] == 'G'):
                    continue
                if(grid[i][j] == 'H'):
                    continue
                next_states, actions = domain.get_state_actions((i,j))
                best_action = 'r'
                best_utility = -1

                for a in actions:
                    temp = 0.0
                    for s in next_states:
                        if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                            temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*curr_V[s[0]][s[1]])
                        else:
                            temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                    if(temp > best_utility):
                        best_utility = temp
                        best_action = a
                
                policy[i][j] = best_action
        
        V = policy_evaluation(domain,policy,grid,eps)

        delta = 0.0
        for i in range(r):
            for j in range(c):
                delta+= abs(V[i][j] - curr_V[i][j])
        
        curr_V = deepcopy(V)
        ss_util.append(curr_V[grid == 'S'])
        num_cycles+=1
        if((num_cycles%50) == 0):
            print("completed",num_cycles,"cycles, delta =",delta)
        
        if(delta < eps):
            break

    return policy, curr_V, num_cycles, ss_util

def RowMajorSweep(domain:Domain, grid, eps):
    curr_V = np.zeros(grid.shape)
    r,c = grid.shape
    ss_util = []
    ss_util.append(curr_V[grid == 'S'])

    num_cycles = 0
    while True:
        delta = 0.0
        for i in range(r - 1, -1, -1):
            for j in range(c-1,-1,-1):
                if(grid[i][j] == 'G'):
                    curr_V[i][j] = domain.goal_reward
                    continue
                if(grid[i][j] == 'H'):
                    curr_V[i][j] = domain.hole_reward
                    continue
                next_states, actions = domain.get_state_actions((i,j))

                best_utility = -1e9
                for a in actions:
                    temp = 0.0
                    for s in next_states:
                        if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                            temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*curr_V[s[0]][s[1]])
                        else:
                            temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                    best_utility = max(best_utility,temp)
                delta = max(delta,abs(best_utility - curr_V[i][j]))
                curr_V[i][j] = best_utility

        ss_util.append(curr_V[grid == 'S'])        
        num_cycles+=1
        if((num_cycles%50) == 0):
            print("completed",num_cycles,"cycles, delta =",delta)
        if(delta < eps*((1-domain.gamma)/domain.gamma)):
            break
    
    policy = [['_' for j in range(c)] for i in range(r)]

    for i in range(r):
        for j in range(c):
            if(grid[i][j] == 'G'):
                continue
            if(grid[i][j] == 'H'):
                continue
            next_states, actions = domain.get_state_actions((i,j))
            best_action = 'r'
            best_utility = -1

            for a in actions:
                temp = 0.0
                for s in next_states:
                    if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*curr_V[s[0]][s[1]])
                    else:
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                if(temp > best_utility):
                    best_utility = temp
                    best_action = a
            
            policy[i][j] = best_action

    return policy, curr_V, num_cycles, ss_util

def PrioritySweep(domain:Domain, grid, eps):
    V = np.zeros(grid.shape)
    r,c = grid.shape
    ss_util = []
    ss_util.append(V[grid == 'S'])
    
    pq = PriorityQueue()
    #initialize priority queue
    # pr = [[0.0 for j in range(c)] for i in range(r)]
    for i in range(r):
        for j in range(c):
            curr = V[i][j]
            new_util = -1e9
            if(grid[i][j] == 'G'):
                V[i][j] = domain.goal_reward
                continue
            if(grid[i][j] == 'H'):
                V[i][j] = domain.hole_reward
                continue

            next_states, actions = domain.get_state_actions((i,j))

            for a in actions:
                temp = 0.0
                for s in next_states:
                    if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*V[s[0]][s[1]])
                    else:
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                new_util = max(new_util,temp)
            pq.push(((i,j),1+ abs(new_util - curr)), 1+ abs(new_util - curr))
            # pr[i][j] = 1+ abs(new_util - curr)
            V[i][j] = new_util

    
    num_cycles = 0
    num_updates = 0
    while True:
        start_state, delta = pq.pop()
        prev_states, _ = domain.get_state_actions(start_state)

        for prev_s in prev_states:
            i,j = prev_s
            curr = V[i][j]
            new_util = -1e9
            if(grid[i][j] == 'G'):
                V[i][j] = domain.goal_reward
                continue
            if(grid[i][j] == 'H'):
                V[i][j] = domain.hole_reward
                continue

            next_states, actions = domain.get_state_actions((i,j))

            for a in actions:
                temp = 0.0
                for s in next_states:
                    if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*V[s[0]][s[1]])
                    else:
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                new_util = max(new_util,temp)
            pq.push(((i,j),1+ abs(new_util - curr)), 1+ abs(new_util - curr))
            # if(new_util == curr):
            #     test = (i,j)
            V[i][j] = new_util
        
        num_updates+=1
        if(num_updates%(r*c) == 0):
            ss_util.append(V[grid == 'S'])
            num_cycles+=1
        # if((num_cycles%1000) == 0):
        #     print("completed",num_cycles,"cycles, delta =",delta - 1)
        if((delta - 1) < eps*((1-domain.gamma)/domain.gamma)):
            print("completed",num_cycles,"cycles, delta =",delta - 1)
            break
    
    policy = [['_' for j in range(c)] for i in range(r)]

    for i in range(r):
        for j in range(c):
            if(grid[i][j] == 'G'):
                continue
            if(grid[i][j] == 'H'):
                continue
            next_states, actions = domain.get_state_actions((i,j))
            best_action = 'r'
            best_utility = -1

            for a in actions:
                temp = 0.0
                for s in next_states:
                    if(grid[s[0]][s[1]] == 'F' or grid[s[0]][s[1]] == 'S'):
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s) + domain.gamma*V[s[0]][s[1]])
                    else:
                        temp+= domain.transition_prob((i,j),a,s)*(domain.get_reward(s))
                if(temp > best_utility):
                    best_utility = temp
                    best_action = a
            
            policy[i][j] = best_action

    return policy,V, num_cycles, ss_util
        
################################## MDP Functions####################################################

################################## Plotting Functions####################################################
CR = 'k'

def plot_heatmap_with_arrows(algo, map, save_path, value_path, policy_path):
    grid = []
    with open(map+'_map.csv', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            grid.append(row)

    grid = np.array(grid)

    mask = np.ones_like(grid, dtype='float64')
    mask[np.where(grid == 'H')] = 0

    with open(value_path, 'rb') as f:
        values = pickle.load(f)
    
    with open(policy_path, 'rb') as f:
        policy = pickle.load(f)


    # values+=mask
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot heatmap
    cax = ax.matshow(values, cmap='autumn_r')
    
    # Add color bar
    # fig.colorbar(cax)
    
    # Add arrows for policy
    for i in range(len(policy)):
        for j in range(len(policy[0])):
            if policy[i][j] == 'r':
                ax.arrow(j - 0.4, i, 0.4, 0, head_width=0.3, head_length=0.3, fc=CR, ec=CR)
            elif policy[i][j] == 'd':
                ax.arrow(j, i - 0.4, 0, 0.4, head_width=0.3, head_length=0.3, fc=CR, ec=CR)
            elif policy[i][j] == 'l':
                ax.arrow(j+0.4, i, -0.4, 0, head_width=0.3, head_length=0.3, fc=CR, ec=CR)
            elif policy[i][j] == 't':
                ax.arrow(j, i+0.4, 0, -0.4, head_width=0.3, head_length=0.3, fc=CR, ec=CR)
            if(mask[i][j] == 0):
                ax.add_patch(plt.Rectangle((j -0.5, i-0.5), 1, 1, color='magenta', fill=True))

    ax.set_aspect('auto')
    plt.tight_layout(pad = 2)
    ax.set_title(algo)
    plt.savefig(save_path)
    plt.close()
    # plt.show()

def line_plot(map, dir = "./"):
    ss_util = 'ss_util_'+map+'.pkl'
    point_path = []
    points = []
    algos = algos = ['ValueIteration','RowMajorSweep','PrioritySweep','PolicyIteration']
    for algo in algos:
        point_path.append(os.path.join(dir,algo,ss_util))
    
    for i,path in enumerate(point_path):
        with open(path, 'rb') as f:
            points.append(pickle.load(f))
            # if(i == 0): print(points[i])
    
    for i,algo in enumerate(algos):
        plt.plot(points[i],label=algo, marker = 'o', markersize = 3)
    
    plt.xlabel('Number of Iterations')
    plt.ylabel('Value')
    plt.title(map+' map')

    # Adding legend
    plt.legend()

    # Displaying the plot
    plt.grid(True, ls = '--')
    # plt.xscale('log', base = 2)
    # if(map == 'large'): plt.yscale('log')
    plt.savefig(os.path.join(dir,'line_plot_'+map+'.png'))
    plt.close()

def plot_partA():
    line_plot('small',dir = "partA")
    line_plot('large',dir = "partA")

    algos = ['ValueIteration','RowMajorSweep','PrioritySweep','PolicyIteration']
    # algos = ['ValueIteration']
    maps = ['large','small']

    for algo in algos:
        for map in maps:
            save_path = os.path.join("partA",algo,map+'.png')
            value_path = os.path.join("partA",algo,"util_"+map+".pkl")
            policy_path = os.path.join("partA",algo,"policy_"+map+".pkl")
            plot_heatmap_with_arrows(algo,map,save_path, value_path, policy_path)

def plot_partB1():
    names = ["neg01","neg09","001"]
    for name in names:
        save_path = os.path.join("partB1",name+'.png')
        value_path = os.path.join("partB1","util_"+name+".pkl")
        policy_path = os.path.join("partB1","policy_"+name+".pkl")
        plot_heatmap_with_arrows('ValueIteration','small',save_path, value_path, policy_path)

def plot_partB2():
    names = ["snow"]
    for name in names:
        save_path = os.path.join("partB2",name+'.png')
        value_path = os.path.join("partB2","util_"+name+".pkl")
        policy_path = os.path.join("partB2","policy_"+name+".pkl")
        plot_heatmap_with_arrows('ValueIteration','small',save_path, value_path, policy_path)
################################## Plotting Functions####################################################

################################## main ####################################################
large_grid = []
with open('large_map.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        large_grid.append(row)

large_grid = np.array(large_grid)
large_domain = Domain(large_grid)

small_grid = []
with open('small_map.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        small_grid.append(row)

small_grid = np.array(small_grid)
small_domain = Domain(small_grid)

def partA():
    ###################### Value Iteration #########################
    run(ValueIteration,large_domain,large_grid,small_domain,small_grid, eps=1e-8)
    print()

    ###################### Row major Sweep #########################
    run(RowMajorSweep,large_domain,large_grid,small_domain,small_grid, eps=1e-8)
    print()

    ###################### Priortized Sweep #########################
    run(PrioritySweep,large_domain,large_grid,small_domain,small_grid, eps=1e-8)
    print()

    ###################### Policy Iteration #########################
    run(PolicyIteration,large_domain,large_grid,small_domain,small_grid, eps=1e-8)
    print()

def partB1():
    domain = Domain(small_grid,living_reward=-0.1, p = 1.0)
    run2(ValueIteration,domain,small_grid,"neg01", dir = "partB1")

    domain = Domain(small_grid,living_reward=-0.9, p = 1.0)
    run2(ValueIteration,domain,small_grid,"neg09", dir = "partB1")

    domain = Domain(small_grid,living_reward=0.001, gamma=0.999, p = 1.0)
    run2(ValueIteration,domain,small_grid,"001",dir = "partB1")

def partB2():
    domain = Domain(small_grid,snow = True)
    run2(ValueIteration,domain,small_grid,"snow",eps = 1e-8, dir = "partB2")


if __name__ == '__main__':
    partA()
    partB1()
    partB2()

    plot_partA()
    plot_partB1()
    plot_partB2()

import numpy as np
import plotly.graph_objs as go
from itertools import permutations

class Plane:
    def __init__(self, sig_s, sig_rx, sig_ry, sig_rz, sig_rx_, sig_ry_, sig_rz_, u, sig_dist = 0, init_bel_sig = 0, init_state = None, T = 500):
        self.T = T
        self.sig_s = sig_s #noise in measurements
        self.sig_rx = sig_rx #noise in position update
        self.sig_ry = sig_ry #noise in position update
        self.sig_rz = sig_rz #noise in position update
        self.sig_rx_ = sig_rx_  #noise in velocity update
        self.sig_ry_ = sig_ry_  #noise in velocity update
        self.sig_rz_ = sig_rz_  #noise in velocity update
        self.sig_dist = sig_dist #noise in landmark observation
        self.Q = np.diag([sig_rx*sig_rx, sig_ry*sig_ry, sig_rz*sig_rz, sig_rx_*sig_rx_, sig_ry_*sig_ry_, sig_rz_*sig_rz_])
        self.R = np.diag([sig_s*sig_s, sig_s*sig_s, sig_s*sig_s])
        self.S = np.diag([sig_dist*sig_dist])
        self.X = np.zeros((6, self.T)) #actual trajectory
        self.Z = np.zeros((3, self.T)) #observations
        self.est = np.zeros((6, self.T)) #estimates
        self.curr_cov = np.diag([init_bel_sig*init_bel_sig]*6)
        self.curr_mu_est = np.zeros((6, self.T))
        self.cov_list = []
        self.cov_list.append(self.curr_cov[:2,:2])
        if(init_state is not None):
            self.X[:,0] = init_state
            self.est[:,0] = init_state
            self.Z[:,0] = init_state[:3]
            self.curr_mu_est = init_state
        self.A = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        self.B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.C = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        
        self.eps = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0, 0], cov=self.Q, size=self.T) #noise in motion
        
        self.delta = np.random.multivariate_normal(mean=[0, 0, 0], cov= self.R, size=self.T) #noise in measurement
        self.gamma = np.random.normal(0, sig_dist*sig_dist, self.T) #noise in landmark observation
    
        #actions
        self.u = u

    
    def update_cov_list(self):
        self.cov_list.append(self.curr_cov[:2,:2])

    def grad(self,t,landmark_pos):
        return (np.dot(self.C.T,np.dot(self.C,self.X[:,t]) - landmark_pos)/self.dist(self.X[:,t], landmark_pos)).reshape(1,-1)
    
    def dist(self,plane_est,landmark_pos):
        temp = np.dot(self.C,plane_est) - landmark_pos
        return np.sqrt(np.dot(temp,temp.T))
    
    def landmark_update(self,t,landmark_pos):
        observation = self.dist(self.X[:,t],landmark_pos) + self.gamma[t]
        H = self.grad(t,landmark_pos)
        K = np.dot(np.dot(self.curr_cov,H.T),np.linalg.inv(np.dot(np.dot(H,self.curr_cov),H.T) + self.S))
        self.curr_mu_est = self.curr_mu_est + np.dot(K, (observation - self.dist(self.curr_mu_est,landmark_pos))).reshape(6)
        self.curr_cov = np.dot((np.eye(self.curr_cov.shape[0]) - np.dot(K,H)),self.curr_cov)
        self.est[:, t] = self.curr_mu_est
        return self.est[:,t]

    
    def kalman_filter(self,curr_mu, curr_cov, action, observation, flag = 'estimate'):
        if flag == 'predict':
            mu = np.dot(self.A,curr_mu) + np.dot(self.B,action)
            cov = np.dot(np.dot(self.A,curr_cov),self.A.T) + self.Q
            return mu, cov
        if flag == 'correct':
            K = np.dot(np.dot(curr_cov,self.C.T),np.linalg.inv(np.dot(np.dot(self.C,curr_cov),self.C.T) + self.R))
            mu = curr_mu + np.dot(K, (observation - np.dot(self.C,curr_mu)))
            cov = np.dot((np.eye(curr_cov.shape[0]) - np.dot(K,self.C)),curr_cov)
            return mu,cov
        mu_action = np.dot(self.A,curr_mu) + np.dot(self.B,action)
        cov_action = np.dot(np.dot(self.A,curr_cov),self.A.T) + self.Q
        K = np.dot(np.dot(cov_action,self.C.T),np.linalg.inv(np.dot(np.dot(self.C,cov_action),self.C.T) + self.R))
        mu = mu_action + np.dot(K, (observation - np.dot(self.C,mu_action)))
        cov = np.dot((np.eye(curr_cov.shape[0]) - np.dot(K,self.C)),cov_action)

        return mu, cov
    
    def observe(self,t):
        self.Z[:, t] = np.dot(self.C,self.X[:,t]) + self.delta[t]
        return self.Z[:,t]

    def simulate(self,t):
        self.X[:, t] = np.dot(self.A, self.X[:, t-1]) + np.dot(self.B, self.u[t-1]) + self.eps[t-1]
        return self.X[:,t]

    def estimate(self, t, flag = 'estimate', observation = None):
        if observation is None:
            observation = self.Z[:, t]
        self.est[:, t], self.curr_cov = self.kalman_filter(self.curr_mu_est, self.curr_cov, self.u[t-1], observation, flag)
        self.curr_mu_est = self.est[:,t]
        return self.est[:,t]

    def get_trajectory(self):
        return self.X
    def get_observations(self):
        return self.Z
    def get_estimates(self):
        return self.est
    def get_cov_list(self):
        return self.cov_list
    
    def plot(self, flags, mask = None, title = 'title'):
        data = []
        for flag in flags:
            if(flag == 'observations'):
                data.append(self.Z)
            elif(flag == 'actual'):
                data.append(self.X)
            elif(flag == 'estimates'):
                data.append(self.est)
            elif(flag == 'true_velocity'):
                data.append(self.X[3:,:])
            elif(flag == 'estimated_velocity'):
                data.append(self.est[3:,:])
            else:
                print("Incorrect Flag")
                return
        col = ['green', 'purple', 'orange']
        fig = go.Figure()
        for i,points in enumerate(data):
            colors = np.full(self.T, col[i])
            if(mask is not None):
                colors = np.where(mask == 0, col[i], 'black')
            colors[0] = 'red'
                
            fig.add_trace(go.Scatter3d(
                x=points[0,:], y=points[1,:], z=points[2,:],
                marker=dict(
                    size=2,
                    color=colors,
                    colorscale='Viridis',
                ),
                line=dict(width=2, color = col[i]),
                name = flags[i]
            ))

        fig.update_layout(
            title = title,
            width=800,
            height=700,
            autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=0,
                        y=1.0707,
                        z=1,
                    )
                ),
                aspectratio = dict( x=1, y=1, z=0.7 ),
                aspectmode = 'manual'
            ),
        )
        fig.write_html(title + '.html')
        # fig.show()


    def get_ellipse_points(self,mu, sigma, num_points=100):
        
        t = np.linspace(0, 2*np.pi, num_points)
        circle_points = np.column_stack((np.cos(t), np.sin(t)))
        eigvals, eigvecs = np.linalg.eig(sigma)
        transform_matrix = eigvecs @ np.diag(np.sqrt(eigvals))
        ellipse_points = mu + np.dot(transform_matrix, circle_points.T).T

        return ellipse_points
        
    def plot_uncertainty_ellipses(self, mask = None, title = 'title'):
        mean_list = list(self.est[:2,:].T)
        cov_list = self.cov_list

        fig = go.Figure()

        t = 0
        for mean, cov in zip(mean_list, cov_list):
            ellipse_points = self.get_ellipse_points(mean, cov)
            x = ellipse_points[:, 0]
            y = ellipse_points[:, 1]
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(width=2),
                showlegend=False
            ))
            if mask is not None:
                if mask[t] == 0:
                    color = 'red'
                else:
                    color = 'black'
            

                fig.add_trace(go.Scatter(
                    x=[mean[0]],
                    y=[mean[1]],
                    mode='markers',
                    marker=dict(size=4, color = color),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[mean[0]],
                    y=[mean[1]],
                    mode='markers',
                    marker=dict(size=8),
                    showlegend=False
                ))
            t+=1
        fig.update_layout(
            title=title+ ': Uncertainty Ellipses',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
        )

        fig.write_html(title + '.html')
        # fig.show()

def get_permutations(input_list):
    all_permutations = list(permutations(input_list))
    
    return all_permutations

def mahalanobis_dist(mu,cov,x):
    return np.dot(np.dot((x - mu).T,np.linalg.inv(cov)),(x- mu))

def metric(association, curr_mean, curr_cov):
    score = 0.0
    for i in range(len(association)):
        score+= (mahalanobis_dist(curr_mean[i],curr_cov[i],association[i][0]))**2
    return score

def min_covered(cost_matrix):
    n = cost_matrix.shape[0]
    k = 2*n
    mask = (2**n) - 1
    rows = np.zeros(n)
    cols = np.zeros(n)
    covered_rows = 0
    covered_cols = 0
    num_lines = 2*n
    for p in range(1,2**k):
        r = p&mask
        c = (p&(mask<<4))>>4
        zeros_covered = True
        for i in range(n):
            for j in range(n):
                if(cost_matrix[i,j] == 0):
                    if((not((r&(1<<i))>>i)) and (not((c&(1<<j))>>j))):
                        zeros_covered = False
                        break
            if(not zeros_covered):
                break
        if(zeros_covered):
            tr = bin(r)
            tc = bin(c)
            curr_lines = tr.count('1') + tc.count('1')
            if(curr_lines < num_lines):
                covered_rows = r
                covered_cols = c   
                num_lines = curr_lines 
    for i in range(n):
        if(covered_rows&(1<<i)):
            rows[i] = 1
        if(covered_cols&(1<<i)):
            cols[i] = 1

    return rows,cols

def hungarian_algorithm(cost_matrix):
    n = cost_matrix.shape[0]
    cost_matrix = cost_matrix - np.min(cost_matrix, axis=1, keepdims=True) #Step 1
    cost_matrix = cost_matrix - np.min(cost_matrix, axis=0, keepdims=True) #Step 2
    rows,cols = min_covered(cost_matrix) #Step 3
    num_covered = np.sum(rows) + np.sum(cols)
    while int(num_covered) < n: #Step 4
        min_num = 1e16
        for i in range(n):
            for j in range(n):
                if ((not rows[i]) and (not cols[j])):
                    min_num = min(min_num, cost_matrix[i,j])
        for i in range(n):
            if(int(rows[i]) == 1):
                continue
            for j in range(n):
                cost_matrix[i,j] -= min_num
        for j in range(n):
            if(int(cols[j]) == 0):
                continue
            for i in range(n):
                cost_matrix[i,j] += min_num
        rows,cols = min_covered(cost_matrix)
        num_covered = np.sum(rows) + np.sum(cols)
    assignment = np.zeros(n)
    for k in range(n):
        idx = 0
        for i in range(n):
            cnt = 0
            for j in range(n):
                cnt+=(cost_matrix[i,j] == 0)
            if(cnt == 1):
                idx = i
                break
        idx2 = 0
        for j in range(n):
            if(cost_matrix[idx,j] == 0):
                idx2 = j
                break
        assignment[idx] = idx2
        for i in range(n):
            cost_matrix[i,idx2]+=1
    return assignment

#1 a)
T = 500
sig_s = 7
sig_rx = 1.2
sig_ry = 1.2
sig_rz = 1.2
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01

t_values = np.arange(T)
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T

plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,init_state = np.array([0,0,0,0,0,0]), T = T)

for t in range(T-1):
    plane.simulate(t+1)
    plane.observe(t+1)

plane.plot(['actual','observations'], title='Figure 2')

#1 c)
T = 500
sig_s = 7
sig_rx = 1.2
sig_ry = 1.2
sig_rz = 1.2
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01
init_bel_sig = 0.01

t_values = np.arange(T)
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T

plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,init_bel_sig = init_bel_sig ,init_state = np.array([0,0,0,0,0,0]), T = T)

for t in range(T-1):
    plane.simulate(t+1)
    plane.observe(t+1)
    plane.estimate(t+1)
    plane.update_cov_list()

plane.plot(['actual','estimates','observations'], title='Figure 4')
plane.plot_uncertainty_ellipses(title= 'Figure 5')

#1d)
T = 500
sig_s = [11,7,7]
sig_rx = [1.2,2,1.2]
sig_ry = [1.2,2,1.2]
sig_rz = [1.2,2,1.2]
sig_rx_ = [0.01,0.01,0.1]
sig_ry_ = [0.01,0.01,0.1]
sig_rz_ = [0.01,0.01,0.1]
init_bel_sig = 0.01

t_values = np.arange(T)
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T

planes = []
for i in range(3):
    plane = Plane(sig_s[i],sig_rx[i],sig_ry[i],sig_rz[i],
                  sig_rx_[i],sig_ry_[i],sig_rz_[i],u,init_bel_sig = init_bel_sig ,init_state = np.array([0,0,0,0,0,0]), T = T)
    planes.append(plane)
for t in range(T-1):
    for i,plane in enumerate(planes):
        planes[i].simulate(t+1)
        planes[i].observe(t+1)
        planes[i].estimate(t+1)
        planes[i].update_cov_list()

planes[0].plot(['actual','estimates','observations'], title = 'Figure 7(a)')
planes[1].plot(['actual','estimates','observations'], title = 'Figure 8(a)')
planes[2].plot(['actual','estimates','observations'], title = 'Figure 9(a)')

planes[0].plot_uncertainty_ellipses(title = 'Figure 7(b)')
planes[1].plot_uncertainty_ellipses(title = 'Figure 8(b)')
planes[2].plot_uncertainty_ellipses(title = 'Figure 9(b)')

#1e)
T = 500
sig_s = 7
sig_rx = 1.2
sig_ry = 1.2
sig_rz = 1.2
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01
init_bel_sig = 0.01

t_values = np.arange(T)
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T

plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,init_bel_sig = init_bel_sig ,init_state = np.array([0,0,0,0,0,0]), T = T)

for t in range(T-1):
    plane.simulate(t+1)
    plane.estimate(t+1,'predict')
    if(((t//50)%2) == 0):
        plane.observe(t+1)
        plane.estimate(t+1,'correct')
    plane.update_cov_list()


plane.plot(['actual','estimates'], title = 'Figure 10')
plane.plot_uncertainty_ellipses(title = 'Figure 11')


#1f)
T = 500
sig_s = 7
sig_rx = 1.2
sig_ry = 1.2
sig_rz = 1.2
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01
init_bel_sig = 0.01

t_values = np.arange(T)
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T

plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,init_bel_sig = init_bel_sig ,init_state = np.array([0,0,0,0,0,0]), T = T)

for t in range(T-1):
    plane.simulate(t+1)
    plane.observe(t+1)
    plane.estimate(t+1)
    plane.update_cov_list()

plane.plot(['true_velocity','estimated_velocity'], title='Figure 12')

#1(g) Different Settings
T = 500
sig_s = [7,8,9, 10]
sig_rx = [1.2, 1.4, 1.6, 2]
sig_ry = [1.2, 1.4, 1.6, 2]
sig_rz = [1.2, 1.4, 1.6, 2]
sig_rx_ = [0.01, 0.03, 0.05, 0.07]
sig_ry_ = [0.01, 0.03, 0.05, 0.07]
sig_rz_ = [0.01, 0.03, 0.05, 0.07]
init_bel_sig = [0.01, 0.03, 0.05, 0.07]
init_state = [np.array([0,0,0,0,0,0]), np.array([5,5,5,0,5,0]), np.array([10,10,10,0,0,10]), np.array([15,15,15,0,0,15])]
t_values = np.arange(T)

#actions
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T
planes = []
for i in range(4):
    plane = Plane(sig_s[i], sig_rx[i], sig_ry[i], sig_rz[i], sig_rx_[i],sig_ry_[i], sig_rz_[i], u, init_bel_sig=init_bel_sig[i], init_state=init_state[i], T = T)
    planes.append(plane)

cnt = 0 #For checking the accuracy of the strategy
for t in range(T-1):
    observations = []
    predictions = []
    pred_cov = []
    cnt+=1
    for i in range(4):
        planes[i].simulate(t+1)
        observations.append((planes[i].observe(t+1),i))
        predictions.append(planes[i].estimate(t+1,'predict')[:3])
        pred_cov.append(planes[i].curr_cov[:3,:3])
    np.random.shuffle(observations) #Randomize the obtained observations
    perms = get_permutations(observations) #get all the permutations
    best_score = 1e16
    best_perm = []
    for i in range(len(perms)):
        curr_score = metric(perms[i],predictions,pred_cov) 
        if(curr_score < best_score):
            best_score = curr_score
            best_perm = perms[i]

    flag = True
    for i,plane in enumerate(planes):
        plane.estimate(t+1, 'correct', best_perm[i][0]) #Correct the believes using the best permutations
        if(i != best_perm[i][1] and flag):
            cnt-=1
            flag = False
print(cnt)

planes[0].plot(['actual', 'estimates'], title = 'Figure 14(a)')
planes[1].plot(['actual', 'estimates'], title = 'Figure 14(b)')
planes[2].plot(['actual', 'estimates'], title = 'Figure 14(c)')
planes[3].plot(['actual', 'estimates'], title = 'Figure 14(d)')

#1(g) Identical Setting
T = 500
sig_s = [7,7,7, 7]
sig_rx = [1.2, 1.2, 1.2, 2]
sig_ry = [1.2, 1.2, 1.2, 2]
sig_rz = [1.2, 1.2, 1.2, 2]
sig_rx_ = [0.01, 0.01, 0.01, 0.01]
sig_ry_ = [0.01, 0.01, 0.01, 0.01]
sig_rz_ = [0.01, 0.01, 0.01, 0.01]
init_bel_sig = [0.01, 0.01, 0.01, 0.01]
init_state = [np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0])]
t_values = np.arange(T)

#actions
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T
planes = []
for i in range(4):
    plane = Plane(sig_s[i], sig_rx[i], sig_ry[i], sig_rz[i], sig_rx_[i],sig_ry_[i], sig_rz_[i], u, init_bel_sig=init_bel_sig[i], init_state=init_state[i], T = T)
    planes.append(plane)

cnt = 0 #For checking the accuracy of the strategy
for t in range(T-1):
    observations = []
    predictions = []
    pred_cov = []
    cnt+=1
    for i in range(4):
        planes[i].simulate(t+1)
        observations.append((planes[i].observe(t+1),i))
        predictions.append(planes[i].estimate(t+1,'predict')[:3])
        pred_cov.append(planes[i].curr_cov[:3,:3])
    np.random.shuffle(observations) #Randomize the obtained observations
    perms = get_permutations(observations) #get all the permutations
    best_score = 1e16
    best_perm = []
    for i in range(len(perms)):
        curr_score = metric(perms[i],predictions,pred_cov) 
        if(curr_score < best_score):
            best_score = curr_score
            best_perm = perms[i]

    flag = True
    for i,plane in enumerate(planes):
        plane.estimate(t+1, 'correct', best_perm[i][0]) #Correct the believes using the best permutations
        if(i != best_perm[i][1] and flag):
            cnt-=1
            flag = False
print(cnt)

planes[0].plot(['actual', 'estimates'], title = 'Figure 15(a)')
planes[1].plot(['actual', 'estimates'], title = 'Figure 15(b)')
planes[2].plot(['actual', 'estimates'], title = 'Figure 15(c)')
planes[3].plot(['actual', 'estimates'], title = 'Figure 15(d)')

#1(h) Different Settings
T = 500
sig_s = [7,8,9, 10]
sig_rx = [1.2, 1.4, 1.6, 2]
sig_ry = [1.2, 1.4, 1.6, 2]
sig_rz = [1.2, 1.4, 1.6, 2]
sig_rx_ = [0.01, 0.03, 0.05, 0.07]
sig_ry_ = [0.01, 0.03, 0.05, 0.07]
sig_rz_ = [0.01, 0.03, 0.05, 0.07]
init_bel_sig = [0.01, 0.03, 0.05, 0.07]
init_state = [np.array([0,0,0,0,0,0]), np.array([5,5,5,0,5,0]), np.array([10,10,10,0,0,10]), np.array([15,15,15,0,0,15])]

t_values = np.arange(T)

#actions
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T
planes = []
for i in range(4):
    plane = Plane(sig_s[i], sig_rx[i], sig_ry[i], sig_rz[i], sig_rx_[i],sig_ry_[i], sig_rz_[i], u, init_bel_sig=init_bel_sig[i], init_state=init_state[i], T = T)
    planes.append(plane)


for t in range(T-1):
    observations = []
    predictions = []
    pred_cov = []
    cnt+=1
    for i in range(4):
        planes[i].simulate(t+1)
        observations.append(planes[i].observe(t+1))
        predictions.append(planes[i].estimate(t+1,'predict')[:3])
        pred_cov.append(planes[i].curr_cov[:3,:3])
    np.random.shuffle(observations) #Randomize the obtained observations
    cost_matrix = np.zeros((4,4))

    for i in range(4):
        for j in range(4):
            cost_matrix[i,j] = (mahalanobis_dist(predictions[i], pred_cov[i], observations[j]))**2

    assignment = hungarian_algorithm(cost_matrix)
    flag = True
    for i,plane in enumerate(planes):
        plane.estimate(t+1, 'correct', observations[int(assignment[i])]) #Correct the believes using the best permutations

planes[0].plot(['actual', 'estimates'], title = 'Figure 16(a)')
planes[1].plot(['actual', 'estimates'], title = 'Figure 16(b)')
planes[2].plot(['actual', 'estimates'], title = 'Figure 16(c)')
planes[3].plot(['actual', 'estimates'], title = 'Figure 16(d)')

#1(h) Identical Setting
T = 500
sig_s = [7,7,7, 7]
sig_rx = [1.2, 1.2, 1.2, 2]
sig_ry = [1.2, 1.2, 1.2, 2]
sig_rz = [1.2, 1.2, 1.2, 2]
sig_rx_ = [0.01, 0.01, 0.01, 0.01]
sig_ry_ = [0.01, 0.01, 0.01, 0.01]
sig_rz_ = [0.01, 0.01, 0.01, 0.01]
init_bel_sig = [0.01, 0.01, 0.01, 0.01]
init_state = [np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0])]
t_values = np.arange(T)

#actions
u = np.vstack([np.sin(t_values), np.cos(t_values), np.sin(t_values)]).T
planes = []
for i in range(4):
    plane = Plane(sig_s[i], sig_rx[i], sig_ry[i], sig_rz[i], sig_rx_[i],sig_ry_[i], sig_rz_[i], u, init_bel_sig=init_bel_sig[i], init_state=init_state[i], T = T)
    planes.append(plane)

for t in range(T-1):
    observations = []
    predictions = []
    pred_cov = []
    cnt+=1
    for i in range(4):
        planes[i].simulate(t+1)
        observations.append(planes[i].observe(t+1))
        predictions.append(planes[i].estimate(t+1,'predict')[:3])
        pred_cov.append(planes[i].curr_cov[:3,:3])
    np.random.shuffle(observations) #Randomize the obtained observations
    cost_matrix = np.zeros((4,4))

    for i in range(4):
        for j in range(4):
            cost_matrix[i,j] = (mahalanobis_dist(predictions[i], pred_cov[i], observations[j]))**2

    assignment = hungarian_algorithm(cost_matrix)
    flag = True
    for i,plane in enumerate(planes):
        plane.estimate(t+1, 'correct', observations[int(assignment[i])]) #Correct the believes using the best permutations

    
planes[0].plot(['actual', 'estimates'], title = 'Figure 17(a)')
planes[1].plot(['actual', 'estimates'], title = 'Figure 17(b)')
planes[2].plot(['actual', 'estimates'], title = 'Figure 17(c)')
planes[3].plot(['actual', 'estimates'], title = 'Figure 17(d)')

#2(b)(c)
T = 200
sig_s = 10
sig_rx = 0.01
sig_ry = 0.01
sig_rz = 0.01
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01
sig_dist = 1
init_bel_sig = 0.01
init_state = np.array([100,0,0,0,4,0])
landmarks = [np.array([150,0,100]), np.array([-150,0,100]), np.array([0,150,100]), np.array([0,-150,100])]
rng = 90
t_values = np.arange(T)

#actions
u = np.vstack([-0.128*np.cos(0.032*t_values), -0.128*np.sin(0.032*t_values), np.array([0.01]*t_values)]).T
plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,sig_dist, init_bel_sig, init_state, T)

cnt = 0
mask = np.zeros(T)
for t in range(T-1):
    X = plane.simulate(t+1)
    plane.observe(t+1)
    plane.estimate(t+1)

    best_dist = 1e16
    nearest_landmark = np.zeros(3)
    for landmark in landmarks:
        curr_dist = plane.dist(X,landmark)
        if(curr_dist < best_dist):
            best_dist = curr_dist
            nearest_landmark = landmark
    if(best_dist <= rng):
        plane.landmark_update(t+1,nearest_landmark)
        cnt+=1
        mask[t] = 1
    plane.update_cov_list()
print(cnt)

plane.plot(['actual', 'estimates'], mask = mask, title='Figure 20')
plane.plot_uncertainty_ellipses(mask, title = 'Figure 21')

#2(d) S = 0.1
T = 200
sig_s = 10
sig_rx = 0.01
sig_ry = 0.01
sig_rz = 0.01
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01
sig_dist = 0.1
init_bel_sig = 0.01
init_state = np.array([100,0,0,0,4,0])
landmarks = [np.array([150,0,100]), np.array([-150,0,100]), np.array([0,150,100]), np.array([0,-150,100])]
rng = 90
t_values = np.arange(T)

#actions
u = np.vstack([-0.128*np.cos(0.032*t_values), -0.128*np.sin(0.032*t_values), np.array([0.01]*t_values)]).T
plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,sig_dist, init_bel_sig, init_state, T)

cnt = 0
mask = np.zeros(T)
for t in range(T-1):
    X = plane.simulate(t+1)
    plane.observe(t+1)
    plane.estimate(t+1)

    best_dist = 1e16
    nearest_landmark = np.zeros(3)
    for landmark in landmarks:
        curr_dist = plane.dist(X,landmark)
        if(curr_dist < best_dist):
            best_dist = curr_dist
            nearest_landmark = landmark
    if(best_dist <= rng):
        plane.landmark_update(t+1,nearest_landmark)
        cnt+=1
        mask[t] = 1
    plane.update_cov_list()
print(cnt)
plane.plot(['actual', 'estimates'], mask, title='Figure 23')
plane.plot_uncertainty_ellipses(mask, title= 'Figure 24')

#2(d) S = 20
T = 200
sig_s = 10
sig_rx = 0.01
sig_ry = 0.01
sig_rz = 0.01
sig_rx_ = 0.01
sig_ry_ = 0.01
sig_rz_ = 0.01
sig_dist = 20
init_bel_sig = 0.01
init_state = np.array([100,0,0,0,4,0])
landmarks = [np.array([150,0,100]), np.array([-150,0,100]), np.array([0,150,100]), np.array([0,-150,100])]
rng = 90
t_values = np.arange(T)

#actions
u = np.vstack([-0.128*np.cos(0.032*t_values), -0.128*np.sin(0.032*t_values), np.array([0.01]*t_values)]).T
plane = Plane(sig_s,sig_rx,sig_ry,sig_rz,sig_rx_,sig_ry_,sig_rz_,u,sig_dist, init_bel_sig, init_state, T)

cnt = 0
mask = np.zeros(T)
for t in range(T-1):
    X = plane.simulate(t+1)
    plane.observe(t+1)
    plane.estimate(t+1)

    #Find the nearest landmark
    best_dist = 1e16
    nearest_landmark = np.zeros(3)
    for landmark in landmarks:
        curr_dist = plane.dist(X,landmark)
        if(curr_dist < best_dist):
            best_dist = curr_dist
            nearest_landmark = landmark
    #Check whether the landmark is within range
    if(best_dist <= rng):
        plane.landmark_update(t+1,nearest_landmark)
        cnt+=1
        mask[t] = 1
    plane.update_cov_list()
print(cnt)
plane.plot(['actual', 'estimates'], mask, title = 'Figure 25')
plane.plot_uncertainty_ellipses(mask, title= 'Figure 26')





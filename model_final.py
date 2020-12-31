import numpy as np
import sympy as sym
from operator import add
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#define pandemic simulator function depending on X, Y, Z returning Casualties
def model_opt(budget, t_limit, dt):
    print('model_opt budget:', budget)
    X = budget[0]
    Y = budget[1]
    Z = budget[2]
    #initialize time at day 0
    t = 0

    #define initial pandemic state
    state = {
        # 'S':[100000.0],
        'S':[38000000.0],
        'I0':[1.0],
        'Is':[0.0],
        'Ia':[0.0],
        'C':[0.0],
        'H':[0.0],
        'V':[0.0]
    }

    #calculate total population and add it to the state_0
    N = [sum(i) for i in zip(*list(state.values()))]
    N_0 = {'N':N}
    state.update(N_0)

    #add iteration limit (aid in troubleshooting)
    t_limit = {'t_limit':t_limit}
    state.update(t_limit)

    #initialize weights (empty so first iteration takes new weights)
    weights = {
        'f_0': [],
        'v': [],
        'f_s': [],
        'f_a': [],
        'i_s': [],
        'i_a': [],
        'c': [],
        'r_1': [],
        'r_2': [],
        'z': []
    }

    #enter while loop to calculate iterations until no sick individuals are left or t_limit is hit
    while(t != state.get('t_limit')): #state.get('I0')[t] + state.get('Is')[t] + state.get('Ia')[t] != 0 or

        #calculates new weight values and append to wieght lists
        weights.get('f_0').append(new_f_0(X))
        weights.get('v').append(new_v(t, Y))
        weights.get('f_s').append(new_f_s(t, X, Z))
        weights.get('f_a').append(new_f_a(t, X, Z))
        weights.get('i_s').append(new_i_s())
        weights.get('i_a').append(new_i_a())
        weights.get('c').append(new_c(t, Z))
        weights.get('r_1').append(new_r1(t, Z))
        weights.get('r_2').append(new_r2(t, Z))
        weights.get('z').append(new_z(t, X, state.get('Is')))

        #calculate new state values and append to state lists
        state.get('S').append(new_S(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('I0').append(new_I0(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('Is').append(new_Is(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('Ia').append(new_Ia(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('C').append(new_C(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('H').append(new_H(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('V').append(new_V(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))

        #iterate time period by 1 day
        t += dt

    #return state.get('S'), state.get('I0'), state.get('Is'), state.get('Ia'), state.get('C'), state.get('H'), state.get('V'), weights.get('f_0'), weights.get('v'), weights.get('f_s'), weights.get('f_a'), weights.get('i_s'), weights.get('i_a'), weights.get('c'), weights.get('r_1'), weights.get('r_2'), weights.get('z')

    print('(used for scipy optimization) Casualties: {:.2f} '.format(max(state.get('C'))))
    return max(state.get('C'))


#define pandemic simulator function depending on X, Y, Z
def model(budget, t_limit, dt):
    print('model budget:', budget)
    X = budget[0]
    Y = budget[1]
    Z = budget[2]
    #initialize time at day 0
    t = 0

    #define initial pandemic state
    state = {
        # 'S':[100000.0],
        'S':[38000000.0],
        'I0':[1.0],
        'Is':[0.0],
        'Ia':[0.0],
        'C':[0.0],
        'H':[0.0],
        'V':[0.0]
    }

    #calculate total population and add it to the state_0
    N = [sum(i) for i in zip(*list(state.values()))]
    N_0 = {'N':N}
    state.update(N_0)

    #add iteration limit (aid in troubleshooting)
    t_limit = {'t_limit':t_limit}
    state.update(t_limit)

    #initialize weights (empty so first iteration takes new weights)
    weights = {
        'f_0': [],
        'v': [],
        'f_s': [],
        'f_a': [],
        'i_s': [],
        'i_a': [],
        'c': [],
        'r_1': [],
        'r_2': [],
        'z': []
    }

    #enter while loop to calculate iterations until no sick individuals are left or t_limit is hit
    while(t != state.get('t_limit')): #state.get('I0')[t] + state.get('Is')[t] + state.get('Ia')[t] != 0 or

        #calculates new weight values and append to wieght lists
        weights.get('f_0').append(new_f_0(X))
        weights.get('v').append(new_v(t, Y))
        weights.get('f_s').append(new_f_s(t, X, Z))
        weights.get('f_a').append(new_f_a(t, X, Z))
        weights.get('i_s').append(new_i_s())
        weights.get('i_a').append(new_i_a())
        weights.get('c').append(new_c(t, Z))
        weights.get('r_1').append(new_r1(t, Z))
        weights.get('r_2').append(new_r2(t, Z))
        weights.get('z').append(new_z(t, X, state.get('Is')))

        #calculate new state values and append to state lists
        state.get('S').append(new_S(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('I0').append(new_I0(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('Is').append(new_Is(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('Ia').append(new_Ia(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('C').append(new_C(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('H').append(new_H(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))
        state.get('V').append(new_V(state.get('N')[0], state.get('S')[t], state.get('I0')[t], state.get('Is')[t], state.get('Ia')[t], state.get('C')[t], state.get('H')[t], state.get('V')[t], weights.get('f_0')[t], weights.get('v')[t], weights.get('f_s')[t], weights.get('f_a')[t], weights.get('i_s')[t], weights.get('i_a')[t], weights.get('c')[t], weights.get('r_1')[t], weights.get('r_2')[t], weights.get('z')[t]))

        #iterate time period by 1 day
        t += dt

    return state.get('S'), state.get('I0'), state.get('Is'), state.get('Ia'), state.get('C'), state.get('H'), state.get('V'), weights.get('f_0'), weights.get('v'), weights.get('f_s'), weights.get('f_a'), weights.get('i_s'), weights.get('i_a'), weights.get('c'), weights.get('r_1'), weights.get('r_2'), weights.get('z')


# define state functions
def new_S(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    S_delta = -1 * z * (S/N) * (I0*f_0 + Is*f_s + Ia*f_a) - S*v
    new_S = S + S_delta
    return new_S

def new_I0(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    I0_delta = z * (S/N) * (I0*f_0 + Is*f_s + Ia*f_a) - I0*i_s - I0*i_a
    new_I0 = I0 + I0_delta
    return new_I0

def new_Is(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    Is_delta = I0*i_s - Is*c - Is*r_1
    new_Is = Is + Is_delta
    return new_Is

def new_Ia(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    Ia_delta = I0*i_a - Ia*r_2
    new_Ia = Ia + Ia_delta
    return new_Ia

def new_C(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    C_delta = Is*c
    new_C = C + C_delta
    return new_C

def new_H(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    H_delta = Is*r_1 + Ia*r_2
    new_H = H + H_delta
    return new_H

def new_V(N, S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z):
    V_delta = S*v
    new_V = V + V_delta
    return new_V

# define weight functions
def new_i_s():
    i_s = 0.8 / 6 #80% of people develop symptomes after on average 6 days
    return i_s

def new_i_a():
    i_a = 0.2 / 6 #80% of people develop symptomes after on average 6 days
    return i_a

def new_v(t, Y):
    # if (Y>200): #Added for test with other solver
    #    X=200

    if(Y>200): #Spending over a certain limit no longer speeds up the process due to time taken for clinical trials
        Y = 200
    spending_shift = -1*(18/5*Y) + 1080
    v = 0.5196*np.exp(0.0146*(t - spending_shift))/100 #divide by 100 to convert percentage to decimal
    if(v>1):
        v = 1
    if(t < spending_shift):
        v = 0
    return v

def new_z(t, X, Is):
    if (X>50): #Added for test with other solver
       X=50

    #parameters
    num_of_cases_trigger = 1000
    population_reaction_speed = 25; #(90% of the change will be done in population_reaction_speed*2 days)

    alert_day = next((x[0] for x in enumerate(Is) if x[1] > num_of_cases_trigger), 1200)
    alert_time_shift = alert_day/population_reaction_speed #based on the following defined exponential, every shift of 1 = 13 days therefore, shift 1/13 for a one day delay
    X_factor = (X/50)*(13.4-1.4)

    z = 13.4 + X_factor*np.exp((-1*t/population_reaction_speed) + alert_time_shift) - X_factor

    if(z > 13.4):
        z = 13.4

    return z

def new_f_0(X):
    if (X>50): #Added for test with other solver
       X=50

    x = sym.Symbol('x')
    budget1 = 0 # in millions
    f0_prob1 = 15/100 # upper bound of probability in decimal
    budget2 = 50 # change to max X budget for flexbility
    f0_prob2 = 0.01/100 # lower bound of probability in decimal
    m = (f0_prob2 - f0_prob1) / (budget2 - budget1)
    f0 = m*(x-budget1)+f0_prob1
    return f0.subs({x:X})

def new_f_s(t, X, Z):
    if (Z>1000): #Added for test with other solver
       Z=1000
    if (X>50): #Added for test with other solver
       X=50

    x = sym.Symbol('x')
    z = sym.Symbol('z')

    avgSpreadRate = 0.15

    xBudget1 = 0 # change to low X budget for flexbility
    zBudget1 = 770 # change to low Z budget for flexbility
    fs_prob1 = (avgSpreadRate+0.1) # upper bound of spread probability when spending lower (in decimal)

    xBudget2 = 50 # change to high X budget for flexbility
    zBudget2 = 1000 # change to high Z budget for flexbility
    fs_prob2 = (avgSpreadRate-0.1) # lower bound of spread probability

    # Symbolic equations below that adapt in case upper/lower bounds change
    m_x = (fs_prob2 - fs_prob1) / (xBudget2 - xBudget1)
    fs_x =  m_x*(x-xBudget1)+fs_prob1

    m_z = (fs_prob2 - fs_prob1) / (zBudget2 - zBudget1)
    fs_z =  m_z*(z-zBudget1)+fs_prob1

    if t <= 180: # Assuming weight of awareness spending is significant in first 6 months
        wx = 0.6
    else:
        wx = 0.1

    fs = wx*fs_x.subs({x:X}) + (1-wx)*fs_z.subs({z:Z})
    return fs

def new_f_a(t, X, Z):
    if (Z>1000): #Added for test with other solver
       Z=1000
    if (X>50): #Added for test with other solver
       X=50
    fa = (1-0.42)*new_f_s(t, X, Z) # asymptomatic probabiliy is 42% less than symptomatic
    return fa

def new_r1(t, Z):
    if (Z>1000):
        Z=1000
    C = Z*0.13/1000
    new_c = 0.2 + C*np.exp((-t/5)) - C
    new_r1 = (1 - new_c) / 14 #base recovery chance plus extra chance from budgeted care
    return new_r1

def new_r2(t, Z):
    if (Z>1000): #Added for test with other solver
        Z=1000
    new_r2 = 1/14  #base recovery chance but in this stage for an expected 14 days
    return new_r2

def new_c(t, Z):
    if (Z>1000):
        Z=1000
    C = Z*0.13/1000
    new_c = (0.2 + C*np.exp((-t/5)) - C) / 14
    return new_c

# optimization function
def constraint1(budget):
    X = budget[0]
    Y = budget[1]
    Z = budget[2]
    B = 1250*0.8
    return B-X-Y-Z


###################
#     Script      #
###################

# Values are in 10^6 Canadian Dollars
B = 0.80 * (50+200+1000) #Change the budget contraint in the optimization function as well

# # budget for model
# X = 50 #0-50
# Y = 100 #0-200
# Z = 1000 #770-1000
# print('Valid model budget:', B == (X+Y+Z))
# budget = np.array([X, Y, Z])
# #budget = [X, Y, Z]

# intial budget for model_opt
X2 = 50 #0-50
Y2 = 0 #10-200
Z2 = 950 #770-1000
x0 = np.array([X2, Y2, Z2])
print('Valid model_opt budget:', B == (X2+Y2+Z2))

#additional arguments to pass into model
t_limit = 1500
dt = 1

# bounds for X, Y, Z
X_bound = (0,50)
Y_bound = (10,200)
Z_bound = (770,1000)
X_bound = (0,50)
Y_bound = (0,200)
Z_bound = (770,1000)
bounds = (X_bound, Y_bound, Z_bound)

# con1 = {'type':'eq', 'fun':constraint1} # currently set as equality constraint for ineqaulity use: ineq
con1 = {'type':'eq', 'fun':constraint1}

# call minimization function
sol = minimize(model_opt, x0, method='SLSQP', args=(t_limit, dt), constraints=con1, bounds=bounds, options={'disp':True})
print(sol)

#call pandemic simulator function for testing
[S, I0, Is, Ia, C, H, V, f_0, v, f_s, f_a, i_s, i_a, c, r_1, r_2, z] = model(sol.x, t_limit, dt)
# print('Casualties (model): ',C)
print('Casualties (model): ',C[-1])

#plot state variables
plt.figure(1)
t = np.arange(0,len(S),1)
plt.plot(t, S, label='Susceptible', color='k')
plt.plot(t, I0, label='Infected, no symptoms', color='m')
plt.plot(t, Is, label='Infected, symptomatic', color='c',)
plt.plot(t, Ia, label='Infected, asymptomatic', color='b')
plt.plot(t, C, label='Casualties', color='r')
plt.plot(t, H, label='Recovered', color='y')
plt.plot(t, V, label='Vaccinated', color='g')
plt.legend()
plt.title('Pandemic Model States Over Time Based on Optimized Initial Government Expenditure in X, Y, Z')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')

#plot weight variables
plt.figure(2)
t = np.arange(0,len(z),1)
plt.plot(t, f_0, label='f_0')
plt.plot(t, v, label='v')
plt.plot(t, f_s, label='f_s')
plt.plot(t, f_a, label='f_a')
plt.plot(t, i_s, label='i_s')
plt.plot(t, i_a, label='i_a')
plt.plot(t, c, label='c')
plt.plot(t, r_1, label='r_1')
plt.plot(t, r_2, label='r_2')
plt.legend()
plt.title('Pandemic Model Weights Over Time')
plt.xlabel('Time (days)')
plt.ylabel('Probability weights')

#plot z
plt.figure(3)
t = np.arange(0,len(z),1)
plt.plot(t, z, label='z')
plt.legend()
plt.title('Transmission Opportunities Per Person Over Time')
plt.xlabel('Time (days)')
plt.ylabel('Number of COVID-19 Transmission Opportunities')

plt.show()

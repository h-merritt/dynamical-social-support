# simulation of dynamical model of social support
# haily merritt
# time of development: fall 2021

# goals - outline to define what success is for model
# cost of vigilance higher
# try discrete time
# look at energy, vigilance, and group energy in different plots
# visualize individual receipt of group bonus
# two agents, vigilance space and energy space, create flow plots
# group energy needs to be higher to see noticeable change
# see when transient dynamics end, and only apply threat after they've passed

# normalize energy influence on contributuon and vigilance, may not need baseline
# using sigmoid

# import packages
import dynsupport as dyn
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
import random
import copy
import itertools
import statistics

# define parameters
size = 3         # size of support network
alpha = -0.5     # scales amount of energy lost to vigilance
beta = -0.3     # scales amount of energy lost to sharing
gamma = 1.1      # scales amount of energy gained from group
delta = 1.1      # scales benefits gained by group pool from sharing
epsilon = -0.1    # scales decay of vigilance
shareCoefPrime = 1 #
shareCoef = 1.1
gain = 1.1
vigilances = np.asarray([0.02] * size)
energies = np.linspace(2,4,num=size)
cs = np.asarray([0.01] * size)
threat = 0.1     # threat magnitude
duration = 2000  # duration of a single trial
stepsize = 0.05
threattimes = [500,900,1300,1700] # times at which threat appears
threatleaves = [505,905,1305,1705]# times when threat disappears


# define functions
# simulate dynamics of social support
def sim(contribution, threatMag):

    # initialize network
    supportNet = dyn.SocialNetwork(size, alpha, beta, gamma, delta, epsilon, vigilance, energy, contribution, shareCoef, shareCoefPrime, gain)

    # record vigilance levels, individual energy, and group energy to plot
    vig_hist = np.zeros((duration, size))
    ind_eng_hist = np.zeros((duration, size))
    group_eng_hist = np.zeros(duration)
    time_hist = np.array(range(duration))
    vigdecay_hist = np.zeros((duration,size))
    contribdecay_hist = np.zeros((duration,size))
    groupbonus_hist = np.zeros((duration,size))
    sharing_hist = np.zeros((duration,size))

    # start time
    time = 0
    # run simulation
    print("begin sim")
    while time < duration:
        # record values of variables
        vig_hist[time] = supportNet.vig
        ind_eng_hist[time] = supportNet.energy
        group_eng_hist[time] = supportNet.group
        vigdecay_hist[time] = supportNet.vigdecay
        contribdecay_hist[time] = supportNet.contribdecay
        groupbonus_hist[time] = supportNet.groupbonus
        sharing_hist[time] = supportNet.contrib

        # add threat
        if time in threattimes:
            # select victim with lowest energy
            victim = np.where(supportNet.energy == min(supportNet.energy))[0]
            #victim2 = np.where(supportNet.energy == statistics.median(supportNet.energy))[0]
            #victim3 = np.where(supportNet.energy == max(supportNet.energy))[0]
            supportNet.threaten(victim,threatMag)
            #supportNet.threaten(victim2,threatMag)
            #supportNet.threaten(victim3,threatMag)

        if time in threatleaves:
            # remove threat after 5 time steps
            supportNet.threaten(victim,0)
            #supportNet.threaten(victim2,0)
            #supportNet.threaten(victim3,0)

        # euler step network dynamics
        supportNet.euler_original(stepsize)
        #supportNet.euler_dyn_res(stepsize)

        # increase time
        time += 1


    #return ind_eng_hist, time_hist

    # below code for third experiment, increase number of victims
    """
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y','k']))
    plt.plot(time_hist, ind_eng_hist, alpha = 0.5)
    color.Colormap('Set3',size)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 1100, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 1500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 1900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.show()
    """

    # below code for first experiment

    plt.subplot(3,2,1)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y','k']))
    #color.Colormap('Set3',size)
    plt.plot(time_hist, ind_eng_hist, alpha = 0.5)
    color.Colormap('Set3',size)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.ylabel('energy')
    #plt.title("individual energies")
    #plt.show()

    plt.subplot(3,2,6)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
    plt.plot(time_hist, contribdecay_hist, alpha = 0.5)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.xlabel("time")
    plt.ylabel('energy decay from sharing')
    #plt.title("energy decay due to sharing")

    plt.subplot(3,2,3)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
    plt.plot(time_hist, vig_hist, alpha = 0.5)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.ylabel('vigilance')
    #plt.title("vigilances")

    plt.subplot(3,2,4)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
    plt.plot(time_hist, vigdecay_hist, alpha = 0.5)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.ylabel('energy decay from vigilance')
    #plt.title("energy decay due to vigilance")

    #plt.subplot(3,2,3)
    #plt.plot(time_hist, group_eng_hist, color = 'k')
    #plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    #plt.ylabel('group pool')
    #plt.title("group pool of energy")

    plt.subplot(3,2,5)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
    plt.plot(time_hist, sharing_hist, alpha = 0.5)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.xlabel("time")
    plt.ylabel('shares to group pool')
    #plt.title("contributions to group pool")

    plt.subplot(3,2,2)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
    plt.plot(time_hist, groupbonus_hist, alpha = 0.5)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.ylabel('bonus from group')
    #plt.title("bonus from group")

    #plt.suptitle("a dynamical model of social support in response to threat")

    plt.show()


# first experiment
energy = copy.deepcopy(energies)
contribution = copy.deepcopy(cs)
vigilance = copy.deepcopy(vigilances)
sim(cs, threat)
# below code to plot behavior across a range of alphas and gammas
"""
alphas = np.linspace(-0.75,-0.25,5)
gammas = np.linspace(0.0,2,5)
count = 1
for alpha in alphas:
    for gamma in gammas:
        # run sims
        energy = copy.deepcopy(energies)
        contribution = copy.deepcopy(cs)
        vigilance = copy.deepcopy(vigilances)
        etemp, time = sim(cs, threat, alpha, gamma)

        # set up plots
        plt.subplot(len(alphas),len(gammas),count)
        plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
        plt.plot(time, etemp, alpha=0.5)
        plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
        plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
        plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
        plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
        count += 1

        #print("alpha: ",alpha," & gamma: ",gamma)


        # add axes labels
        # x
        if np.where(alphas==alpha) == len(alphas)-1:
            plt.xlabel("time")
            print(np.where(alphas==alpha))
        if np.where(gammas==gamma) == 0:
            plt.ylabel("energy levels")
            print(np.where(gammas==gamma))

plt.show()
"""

# below code increases the number of individuals under threat, decreases
# intervals between threats, increases duration of threats, and increases
# magnitude of threats
#threattimes = [500,900,1300,1700] # times at which threat appears
#threatleaves = [505,905,1305,1705]# times when threat disappears
#threat = 0.1     # threat magnitude

# increase number of victims
#energy = copy.deepcopy(energies)
#contribution = copy.deepcopy(cs)
#vigilance = copy.deepcopy(vigilances)
#sim(cs, threat, threattimes, threatleaves)

# decrease intervals between threats
#threattimes = [500,700,900,1100,1300,1500,1700,1900] # times at which threat appears
#threatleaves = [505,705,905,1105,1305,1505,1705,1905]# times when threat disappears
#energy = copy.deepcopy(energies)
#contribution = copy.deepcopy(cs)
#vigilance = copy.deepcopy(vigilances)
#sim(cs, threat, threattimes, threatleaves)

# increase duration of threattimes
#energy = copy.deepcopy(energies)
#contribution = copy.deepcopy(cs)
#vigilance = copy.deepcopy(vigilances)
#threattimes = [500,900,1300,1700] # times at which threat appears
#threatleaves = [550,950,1350,1750]# times when threat disappears
#sim(cs, threat, threattimes, threatleaves)
"""
# increase magnitude of threats
threattimes = [500,900,1300,1700] # times at which threat appears
threatleaves = [505,905,1305,1705]# times when threat disappears
threats = [0.2,0.4,0.6,0.8]     # threat magnitude
count = 1
for threat in threats:
    energy = copy.deepcopy(energies)
    contribution = copy.deepcopy(cs)
    vigilance = copy.deepcopy(vigilances)
    etemp, time = sim(cs, threat)

    # set up plots
    plt.subplot(2,2,count)
    plt.gca().set_prop_cycle(plt.cycler('color', ['c','m','y']))
    plt.plot(time, etemp, alpha=0.5)
    plt.axvline(x = 500, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 900, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1300, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    plt.axvline(x = 1700, color = 'silver', alpha = 0.5, linestyle = 'dotted')
    count += 1
plt.show()
"""

# dynamical model of social support in response to threat
# haily merritt
# time of development: fall 2021 - spring 2022

# import dependencies
import math
import random
import numpy as np
from scipy.special import expit as sigmoid

class SocialNetwork():

    def __init__(self, size, alpha, beta, gamma, delta, epsilon, vigilance, energy, contribution, shareCoef, shareCoefPrime, gain):

        # group parameters
        self.size = size            # how many individuals in social network

        # scaling parameters
        self.alpha = alpha          # how much vigilance drains individual energy (<0)
        self.beta = beta            # how much energy contribution drains energy (<0)
        self.gamma = gamma          # how much energy comes from group (>0)
        self.delta = delta          # benefit (/ risk) of sharing
        self.epsilon = epsilon      # scales vigilance
        self.shareCoefPrime = shareCoefPrime
        self.shareCoef = shareCoef
        self.gain = gain

        # individual parameters
        self.vig = vigilance        # vector with starting vigilance levels of individuals
        self.energy = energy        # vector with starting energy levels of individuals
        self.contrib = sigmoid(-self.vig) # vector with starting contributions of individuals
        self.threat = np.zeros(size)        # vector with threat magnitudes of individuals
        #self.baseline = [0.5,0.5,0.5]       # vector with baseline energy levels of individuals
        self.baseline = [0.5]

        # initial conditions
        self.group = max(self.delta * np.sum(self.contrib * self.energy), 0)
        self.vigdecay = self.alpha * (self.vig * self.energy)
        self.contribdecay = self.beta * (self.contrib * self.energy)
        self.groupbonus = self.gamma * (self.vig * self.group)

    # METHODS
    def euler_original(self, stepsize):
        # euler step all dynamical variables
        # this version is my original formulationt

        # update group energy
        self.group = max(self.delta * np.sum(self.contrib * self.energy), 0)
        # update individuals' vigilance levels
        self.vig += stepsize * ((self.epsilon * self.vig) + self.threat)
        # determine contrib value based on vigilance
        self.contrib = sigmoid(-self.vig)

        # update individual energy levels
        self.vigdecay = self.alpha * (self.vig * self.energy)
        self.contribdecay = self.beta * (self.contrib * self.energy)
        self.groupbonus = self.gamma * (self.vig * self.group)

        self.energy += stepsize * (self.vigdecay + (self.contribdecay + (self.groupbonus + self.baseline)))


    def euler_dyn_res(self, stepsize):
        # euler step all dynamical variables
        # this version treats the group pool (reservoir) as a dynamical variable
        # gets rid of contribution equation

        # update reservoir of energy shared by group
        self.group += stepsize * (self.shareCoefPrime * np.sum(self.energy/(self.vig + 1)))

        # update individual vigilance levels
        self.vig += stepsize * ((self.epsilon * self.vig) + self.threat)

        # update individual energy levels
        self.energy += (self.alpha * (self.vig * self.energy) + self.shareCoef * (self.energy/(self.vig +1)) + self.gain * self.vig * self.group)


    def threaten(self, victim, magnitude):
        # apply / remove threat

        self.threat[victim] = magnitude

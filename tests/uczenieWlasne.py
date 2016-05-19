# -*- coding: utf-8 -*-
"""
Created on Wed May 18 22:13:14 2016

@author: Tomasz Sosnowski
"""

import itertools as it
import pickle
from random import sample, randint, random
from time import time
from vizdoom import *

import cv2
import numpy as np
import theano
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, get_output, get_all_params, \
    get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor
from tqdm import *
from time import sleep

def main():

    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State best Q-Value")
    r = tensor.vector("Rewards")
    nonterminal = tensor.vector("Nonterminal", dtype="int8")
    
    
    dqn = InputLayer(shape=[None, 1, 2000], input_var=s1)#zredukowalem 2 wymiary do jednego - czy dobrze?
    dqn = DenseLayer(dqn, num_units=2000, nonlinearity=rectify, W=GlorotUniform("relu"),b=Constant(.1))
    dqn = DenseLayer(dqn, num_units=2000, nonlinearity=rectify, W=GlorotUniform("relu"),b=Constant(.1))
    dqn = DenseLayer(dqn, num_units=2000, nonlinearity=rectify, W=GlorotUniform("relu"),b=Constant(.1))
    
    dqn = DenseLayer(dqn, num_units=available_actions_num, nonlinearity=None)
    
    q = get_output(dqn)
    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * nonterminal * q2)
    loss = squared_error(q, target_q).mean()
    
    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, learning_rate)
    
    function_learn = theano.function([s1, q2, a, r, nonterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    
    # Creates and initializes the environment.
    print "Initializing doom..."
    game = DoomGame()
    game.load_config("../../examples/config/learning.cfg")
    game.init()
    print "Doom initialized."
    
    # Creates all possible actions.
    n = game.get_available_buttons_size()
    actions = []
    for perm in it.product([0, 1], repeat=n):
        actions.append(list(perm))
        

def learn(epochs,steps,game):
    
    for i in range(0,epochs):
        game.new_episode()
        for j in range(0,steps):
            learningStep(game)
            

def testNetwork(epochs,steps,game):
    for i in range(0,epochs):
        game.new_episode()
        while not game.is_episode_finished():
            game.make_action(get_best_action(prepareMatrix(game.get_state().image_buffer,40,50)),skiprate+1)
    
            
def learningStep(game,epsilon):
    s1=prepareMatrix(game.get_state().image_buffer,40,50)
    if random() <= epsilon:
        a = randint(0, len(actions) - 1)
    else:
        # Chooses the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], skiprate+1)
    if game.is_episode_finished():
        s2 = None
    else:
        s2 = convert(game.get_state().image_buffer)
    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, reward)
    
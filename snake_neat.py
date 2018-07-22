"""
Snake experiment using a feed-forward neural network.
"""
from __future__ import print_function

import os
import pickle
import numpy as np

from snake import Game

import neat
import visualize
import pygame

runs_per_net = 5

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
    
        #pygame.init()
        #screen = pygame.display.set_mode((20 * 16,20 * 16))
        #screen.fill(pygame.Color('black'))
        #pygame.display.set_caption('Snake')
        #pygame.display.flip()
    
        sim = Game(20,20)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while True:
            inputs = sim.get_normalized_state()
            action = net.activate(inputs)

            # Apply action to the simulated snake
            valid = sim.step(np.argmax(action))

            # Stop if the network fails to keep the snake within the boundaries or hits itself.
            # The per-run fitness is the number of pills eaten
            if not valid:
                break
           
            fitness = sim.score

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def main(): 
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(4, eval_genome)
    winner = pop.run(pe.evaluate,10000)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'y', -3: 'l', -4: 'f', -5: 'r', -6: 'dir_x', -7: 'dir_y', 0: 'forward', 1: 'left', 2: 'right'}

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")

    
if __name__ == '__main__':
    main()
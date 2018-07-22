# NEAT-Snake
NeuroEvolution of Augmenting Topologies(NEAT) is a technique used in machine learning, which mimicks the process of natural selection to evolve a population of AI candidates which can solve a given task.

Read more about NEAT here:  
http://eplex.cs.ucf.edu/hyperNEATpage/

For this experiment I use NEAT-Python for the NEAT implementation and Pygame for the game logic:  
https://neat-python.readthedocs.io/en/latest/  
https://www.pygame.org/docs/

I wrote my own Snake clone and use the NEAT implementation to train my population.  
I use the following inputs:
 - the direction to the pill
 - the status of the cells adjacent to the head (free or obstacle)
 - the current direction of the snake
 
The possible outputs are either going left, right or forward.

First run:  
snake_neat.py

Then:  
snake.py

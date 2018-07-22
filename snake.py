import pygame
import random
import math
import time
import neat
import pickle
import os
import numpy as np

class Snake:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dir_x = 1
        self.dir_y = 0
        self.length = 1
        self.hunger = 100
        self.nodes = [(x,y)]
        
    # 0: forward, 1: left, 2: right    
    def move(self, action):
        angle = 0
        if action == 1:
            angle = -90
        if action == 2:
            angle = 90
        rad = math.radians(angle)
        
        new_x = self.dir_x * round(math.cos(rad)) - self.dir_y * round(math.sin(rad)); 
        new_y = self.dir_x * round(math.sin(rad)) + self.dir_y * round(math.cos(rad));
        self.dir_x = new_x
        self.dir_y = new_y
        
        self.x += self.dir_x
        self.y += self.dir_y
            
        if (self.x,self.y) in self.nodes:
            return False
            
        self.nodes.append((self.x, self.y))
        self.hunger -= 1
        return True
        
class Pill:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Game:
        
    def __init__(self, width, height, screen=None, player=None):
        self.width = width
        self.height = height
        self.player = player
        self.screen = screen
        self.board = [[]*self.height for x in range(self.width)]
        self.snake = None
        self.pill = None
        self.num_steps = 0
        self.score = 0
        self.reset()
        
    def reset(self):
        self.num_steps = 0
        self.score = 0
        self.snake = Snake(self.width//2, self.height//2)
        while True:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if x != self.width//2 or y != self.height//2:
                break
        self.pill = Pill(x,y)
        if self.screen is not None:
            self.draw()
        
    def draw(self):
        # reset screen
        self.screen.fill(pygame.Color('black'))
        # draw snake
        for x,y in self.snake.nodes:
            pygame.draw.rect(self.screen, pygame.Color('white'), pygame.Rect(x*16,y*16,16,16))        
        # draw pill        
        pygame.draw.circle(self.screen, pygame.Color('white'), (self.pill.x*16+8,self.pill.y*16+8), 6)
        # update screen
        pygame.display.flip()
        
    
    def step(self,action=None):
        self.num_steps += 1
        #time.sleep(0.2)
        if self.player is not None:
            action = self.player.get_action()
        valid = self.snake.move(action)
        self.check_pill()
        if (self.snake.x < 0 or self.snake.x >= self.width or
            self.snake.y < 0 or self.snake.y >= self.height or
            not valid or self.snake.hunger <= 0):
            return False   
        if self.screen is not None:
            self.draw()
        return True
        
    def check_pill(self):
        if self.snake.x == self.pill.x and self.snake.y == self.pill.y:
            valid = False
            while not valid:
                valid = True
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)
                for s_x,s_y in self.snake.nodes:
                    if x == s_x and y == s_y:
                        valid = False
                        break
            self.pill = Pill(x,y)
            self.snake.length += 1
            self.snake.hunger += 100
            self.score += 1000
        else:
            del self.snake.nodes[0]
            self.score += max(0,10 - round(((self.pill.x - self.snake.x)**2 + (self.pill.y - self.snake.y)**2)**.5))
            
    def get_normalized_state(self):
        x = self.pill.x - self.snake.x
        y = self.pill.y - self.snake.y
        length = (x**2 + y**2)**.5
        x = x/length
        y = y/length
        
        rad = math.radians(-90)
        
        new_dir_x = self.snake.dir_x * round(math.cos(rad)) - self.snake.dir_y * round(math.sin(rad)); 
        new_dir_y = self.snake.dir_x * round(math.sin(rad)) + self.snake.dir_y * round(math.cos(rad));
        
        new_x = self.snake.x + new_dir_x
        new_y = self.snake.y + new_dir_y
        
        l = 1 if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height or (new_x,new_y) in self.snake.nodes else 0
        
        rad = math.radians(90)
        
        new_dir_x = self.snake.dir_x * round(math.cos(rad)) - self.snake.dir_y * round(math.sin(rad)); 
        new_dir_y = self.snake.dir_x * round(math.sin(rad)) + self.snake.dir_y * round(math.cos(rad));
        
        new_x = self.snake.x + new_dir_x
        new_y = self.snake.y + new_dir_y
        
        r = 1 if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height or (new_x,new_y) in self.snake.nodes else 0
        
        new_x = self.snake.x + self.snake.dir_x
        new_y = self.snake.y + self.snake.dir_y
        
        f = 1 if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height or (new_x,new_y) in self.snake.nodes else 0
        return [x,y,l,f,r,self.snake.dir_x,self.snake.dir_y]
        
        
class Player:

    def __init__(self):
        self.type = 0
        self.name = 'human'
    
    def get_action(self):
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    return 2
                elif event.key == pygame.K_a:
                    return 1
                elif event.key == pygame.K_w:
                    return 0

def main():

    width = 25
    height = 25
    
    pygame.init()
    screen = pygame.display.set_mode((width * 16,height * 16))
    screen.fill(pygame.Color('black'))
    pygame.display.set_caption('Snake')
    pygame.display.flip()
    game = Game(width,height,screen)
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('winner-feedforward', 'rb') as f:
        winner = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    running = True
    while running:
        pygame.event.pump()
        inputs = game.get_normalized_state()
        action = net.activate(inputs)
        running = game.step(np.argmax(action))
        #running = game.step()
    
if __name__ == '__main__':
    main()
import torch
import random
from game import SnakeGame, Point, Direction, BLOCK_SIZE
from collections import deque
import numpy as np


# Define Constants
from model import LinearQNet, QTrainer
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
EPSILON_CONSTANT = 80

INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Control the randomness
        self.gamma = 0.9  # Discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft if exceeds maxlen

        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)
        self.last_10_scores = [0] * 10

    def get_state(self, game):
        head = game.head

        danger_state = {
            'point_l': Point(head.x - BLOCK_SIZE, head.y),
            'point_r': Point(head.x + BLOCK_SIZE, head.y),
            'point_u': Point(head.x, head.y - BLOCK_SIZE),
            'point_d': Point(head.x, head.y + BLOCK_SIZE),
            'dir_l': game.direction == Direction.LEFT,
            'dir_r': game.direction == Direction.RIGHT,
            'dir_u': game.direction == Direction.UP,
            'dir_d': game.direction == Direction.DOWN,
        }

        state = [
            # Danger Paths
            self._get_danger(danger_state, game, 'straight'),
            self._get_danger(danger_state, game, 'right'),
            self._get_danger(danger_state, game, 'left'),

            # Current direction
            danger_state['dir_l'],
            danger_state['dir_r'],
            danger_state['dir_u'],
            danger_state['dir_d'],

            # Food state
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = EPSILON_CONSTANT - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Executes the model forward method
            move_idx = torch.argmax(prediction).item()
            final_move[move_idx] = 1
        return final_move

    def _get_danger(self, danger_state, game, heading):
        # create a map 32x24
        # calc new pos of head on map
        # calc fruit pos on map
        # call path finder to get to fruit from new pos
        # if no path found, mark it in returned state
        if heading == 'straight':
            return (danger_state['dir_r'] and game.is_collision(danger_state['point_r'])) or \
                   (danger_state['dir_l'] and game.is_collision(danger_state['point_l'])) or \
                   (danger_state['dir_u'] and game.is_collision(danger_state['point_u'])) or \
                   (danger_state['dir_d'] and game.is_collision(danger_state['point_d']))
        if heading == 'right':
            return (danger_state['dir_r'] and game.is_collision(danger_state['point_d'])) or \
                   (danger_state['dir_l'] and game.is_collision(danger_state['point_u'])) or \
                   (danger_state['dir_u'] and game.is_collision(danger_state['point_r'])) or \
                   (danger_state['dir_d'] and game.is_collision(danger_state['point_l']))
        else:
            return (danger_state['dir_r'] and game.is_collision(danger_state['point_u'])) or \
                   (danger_state['dir_l'] and game.is_collision(danger_state['point_d'])) or \
                   (danger_state['dir_u'] and game.is_collision(danger_state['point_l'])) or \
                   (danger_state['dir_d'] and game.is_collision(danger_state['point_r']))


def train():
    plot_scores = []
    plot_mean = []
    plot_last_10 = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game = SnakeGame(speed=120)

    # Training loop
    while True:
        # Get current state
        current_state = agent.get_state(game)

        # Get move
        final_action = agent.get_action(current_state)

        # perform move and get new state
        done, reward, score = game.play_step_ai(final_action)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(current_state, final_action, reward, new_state, done)

        # remember
        agent.remember(current_state, final_action, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            # train long memory (replay all old games)
            agent.train_long_memory()

            if score > record_score:
                record_score = score

            print(f'Game: {agent.n_games}\nScore: {score}\nRecord: {record_score}')
            agent.last_10_scores.append(score)
            agent.last_10_scores.pop(0)
            total_score += score
            mean_score = total_score / agent.n_games
            last_10_mean = sum(agent.last_10_scores) / 10
            plot_last_10.append(last_10_mean)
            plot_mean.append(mean_score)
            plot_scores.append(score)
            plot(plot_scores, plot_mean, plot_last_10)





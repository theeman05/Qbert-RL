"""
Class which maps an entity to a location based on the color of the entity
Author: Ethan Hartman
"""
import math

import numpy as np
from Block import Block, EntityStates, Mappers

# Note: Locations are stored in the format (row, col)

# First location of blocks.
BLOCK_POINTS = np.array([
    (34, 76),
    (62, 64),
    (62, 92),
    (91, 52),
    (91, 76),
    (91, 104),
    (120, 40),
    (120, 64),
    (120, 92),
    (120, 116),
    (149, 28),
    (149, 52),
    (149, 76),
    (149, 104),
    (149, 128),
    (178, 16),
    (178, 40),
    (178, 64),
    (178, 92),
    (178, 116),
    (178, 140)
])

BLOCK_HEIGHT = 5
BLOCK_WIDTH_EXTENTS = (8, 12)  # From being scanned, blocks extend 8 pixels left, 11 pixels right (12 for subarr)
DEFAULT_POS = (-1, -1)

MIN_COLOR_VALUE = 0  # Minimum value blocks can be uncolored to.

CHARACTER_HEAD_OFFSET = np.array((19, 0))  # Because character's head will be closer to above block: 19, 0

# Location of top left corner of bars.
BAR_TOP_LEFTS = ((138, 12), (138, 140), (80, 36), (80, 116), (22, 88))
BAR_COLS = np.array(((12, 20), (140, 148)))
BAR_HEIGHT = 2
BAR_WIDTH = 9


def displayUnfoundBlocks(obs):
    """
    Locates any blocks which weren't found yet.
    :param obs: Current observation
    """
    mask = np.zeros_like(obs)
    for point in BLOCK_POINTS:
        mask[point[0]: point[0] + BLOCK_HEIGHT, point[1] - BLOCK_WIDTH_EXTENTS[0]: point[1] + BLOCK_WIDTH_EXTENTS[1]] = 1
    new_obs = obs.copy()
    new_obs[np.where(mask == 1)] *= 0
    print(np.where(new_obs == obs[BLOCK_POINTS[0, 0], BLOCK_POINTS[0, 1]]))


def ignoreSensors(obs):
    """
    Get observation without the side jumper bars in it.
    :param obs: Current observation
    :return: The unbarred observation.
    """
    mask = np.zeros_like(obs)
    for bar_top_left in BAR_TOP_LEFTS:
        mask[bar_top_left[0]: bar_top_left[0] + BAR_HEIGHT, bar_top_left[1]: bar_top_left[1] + BAR_WIDTH] = 1
    unbarred_observation = obs.copy()
    unbarred_observation[np.where(mask == 1)] *= 0
    return unbarred_observation


def getClosestBlock(to):
    """
    Get the closest block to a given point.
    :param to: Point to find closest block to
    :return: The closest block to the given point.
    """
    return Block.getBlockAtIdx(np.argmin(np.sum((BLOCK_POINTS - to) ** 2, axis=1)))


def calcMagBetweenPoints(p1, p2):
    """
    Calculate the magnitude between two points
    :param p1: First point.
    :param p2: Second point.
    :return: Distance between the two points.
    """
    return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Instantiate our blocks
for _ in BLOCK_POINTS:
    Block()


class EntityMapper:
    """
    Maps entities to positions, which will update block instances.
    """

    def __init__(self, entity_state):
        """
        Instantiate an EntityMapper with an entity state
        :param entity_state: The entity state representing the entity.
        """
        self.position = np.array(DEFAULT_POS)
        self.entity_state = entity_state
        self.cur_block = None
        Mappers[entity_state] = self

    def updateLocation(self, observation):
        """
        Update an entity's location based on the current observation
        Will update block state if necessary.
        :param observation: The current observation.
        """
        same_colors = np.where(np.all(observation == self.entity_state.value, axis=-1))
        if len(same_colors[0]) > 0:
            compare_idx = 0
            if self.entity_state == EntityStates.SPRINGY:
                compare_idx = -3
            if self.position[0] != same_colors[0][compare_idx] or not self.cur_block:
                prev_block = self.cur_block
                self.position[0] = same_colors[0][compare_idx]
                self.position[1] = same_colors[1][compare_idx]
                if compare_idx == 0:
                    cur_block = getClosestBlock(self.position + CHARACTER_HEAD_OFFSET)
                else:
                    cur_block = getClosestBlock(self.position)
                if prev_block != cur_block:
                    # Update entities on blocks
                    self.cur_block = cur_block
                    if self.entity_state == EntityStates.QBERT:
                        if cur_block.color_idx < Block.fully_colored_block_idx:
                            if Block.blocks_colored_idx != -1:
                                cur_block.color_idx += 1
                            Block.blocks_colored_idx += 1
                    elif self.entity_state == EntityStates.GREEN_DUDE:
                        if cur_block.color_idx > MIN_COLOR_VALUE:
                            cur_block.color_idx -= 1
                            Block.blocks_colored_idx -= 1
        else:
            self.cur_block = None
            self.position[0] = DEFAULT_POS[0]
            self.position[1] = DEFAULT_POS[1]

    @staticmethod
    def updateAll(observation):
        """
        Updates all observers then returns the 'best' action for QBert to execute.
        :param observation: Current observation
        :return: Integer representing the best action to execute.
        """

        sensored = ignoreSensors(observation)
        for _, mapper in Mappers.items():
            mapper.updateLocation(sensored)

        action_idx = 0
        qbert = Mappers[EntityStates.QBERT]

        if qbert.cur_block:
            if qbert.cur_block.index == 20:
                action_idx = 4

        if action_idx == 0:
            best_block = qbert.cur_block.getBestNeighbor(observation[5, 3, 0] == 0) if qbert.cur_block else None
            if best_block:
                action_idx += 1 if BLOCK_POINTS[best_block.index, 0] > qbert.position[0] else 0  # d or u
                action_idx += 2 if BLOCK_POINTS[best_block.index, 1] > qbert.position[1] else 4  # r or l

        if Block.blocks_colored_idx == Block.fully_colored_block_idx * len(BLOCK_POINTS):
            Block.resetBlocks()

        return action_idx


# Map all entity states
for state in EntityStates:
    EntityMapper(state)

"""
Class for a block in qbert.
Author: Ethan Hartman
"""
from enum import Enum
from collections import deque

Blocks = []  # List of instantiated blocks
Mappers = dict()  # Dictionary of the mappers for each entity state


class EntityStates(Enum):
    """
    Enum states for entities where the value is the entity color.
    """
    GREEN_DUDE = (50, 132, 50)
    SPRINGY = (146, 70, 192)
    QBERT = (181, 83, 40)


def getBlockNeighbors(block_index):
    """
    Return the neighbors of a block at the given index, assuming the shape is a pyramid.
    :param block_index: Index of the block to get neighbors of.
    :return: List of neighboring blocks.
    """
    row_idx = 0
    summed_blocks = 1
    while block_index >= summed_blocks:
        row_idx += 1
        summed_blocks += (row_idx + 1)

    neighbors = []
    if summed_blocks < len(Blocks):  # Guarantee bl and br blocks
        neighbors.append(Blocks[block_index + row_idx + 1])  # add bl
        neighbors.append(Blocks[block_index + row_idx + 2])  # add br

    # Now lets check ul, ur
    if block_index != 0:
        if block_index != summed_blocks - row_idx - 1:
            neighbors.append(Blocks[block_index - row_idx - 1])  # add ul
        if block_index != summed_blocks - 1:
            neighbors.append(Blocks[block_index - row_idx])  # add ur

    return neighbors


def getSafestNeighbor(block, get_best, check_best_neighbors):
    """
    Get the safest block assuming block is the start block.
    :param block: The block we'd like to get the best neighbor from.
    :param get_best: If we want to get the best block to go to if there is not safest.
    :param check_best_neighbors: If we want to ensure the best choice's neighbors are safe.
    :return: Safest block to move to.
    """
    predecessor = dict()
    to_visit = deque()
    to_visit.append(block)
    predecessor[block] = True
    while len(to_visit) > 0:
        cur = to_visit.popleft()
        if ((not Mappers[EntityStates.GREEN_DUDE].cur_block and cur.color_idx != Block.fully_colored_block_idx)
            or (cur == Mappers[EntityStates.GREEN_DUDE].cur_block)) \
                and cur != block:
            to_visit.append(cur)  # Have to add it back in
            break
        for neighbor in getBlockNeighbors(cur.index):
            if not predecessor.get(neighbor) and neighbor.isSafe():
                # Add new possible neighbor and check out one block to see if safe
                predecessor[neighbor] = cur
                safe = True
                if not get_best:
                    for neighbor_neighbor in getBlockNeighbors(cur.index):
                        if not neighbor_neighbor.isSafe():
                            safe = False
                if safe:
                    to_visit.append(neighbor)

    if len(to_visit) > 0:  # get the first safe block on the way to the desired location
        cur = to_visit.pop()
        while predecessor[cur] != block:
            cur = predecessor[cur]

        if check_best_neighbors and Mappers[EntityStates.SPRINGY].cur_block:
            neighbors = getBlockNeighbors(cur.index)
            for neighbor in neighbors:
                if not neighbor.isSafe():
                    neighbors = getBlockNeighbors(Mappers[EntityStates.QBERT].cur_block.index)
                    cur = None
                    for direct_neighbor in neighbors:
                        if direct_neighbor.isSafe() and direct_neighbor.index != 20 and direct_neighbor.index != 15:
                            safe = True
                            for neighbor_of_direct in getBlockNeighbors(direct_neighbor.index):
                                if not neighbor_of_direct.isSafe():
                                    safe = False
                                    break
                            if safe:
                                cur = direct_neighbor
                                break
                    break
        if cur:
            return cur
        elif get_best:
            return getSafestNeighbor(block, False, check_best_neighbors)


class Block:
    fully_colored_block_idx = 1
    blocks_colored_idx = -1
    reset_times = 0

    def __init__(self):
        """
        Initialize, assuming the block is the default color for the stage.
        Blocks should be added from top to bottom, left to right.
        """
        self.color_idx = 0
        self.index = len(Blocks)
        Blocks.append(self)

    def getBestNeighbor(self, check_best_neighbors=True):
        """
        Get the best neighbor to go to of a given block.
        :param check_best_neighbors: If we want to ensure the best choice's neighbors are safe. Default: True
        :return: The safest neighbor of this block.
        """
        return getSafestNeighbor(self, True, check_best_neighbors)

    def isSafe(self):
        """
        :return: True if springy is not on this block, false o/w.
        """
        return self != Mappers[EntityStates.SPRINGY].cur_block

    def __hash__(self):
        """
        :return: The index of this block, to hash with.
        """
        return self.index

    @staticmethod
    def getBlockAtIdx(idx):
        """
        :return: The block at the specified index.
        """
        return Blocks[idx]

    @staticmethod
    def resetBlocks():
        """
        Reset blocks to original state and increases the number of reset times.
        """
        Block.reset_times += 1
        Block.blocks_colored_idx = -1
        for block in Blocks:
            block.color_idx = 0

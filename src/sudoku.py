from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import plotly.graph_objects as go


class SudokuGrid:
    def __init__(self, content: list[int], solving_algorithm: SolverStrategy):
        """Initialize a sudoku field by filling it with the known numbers

        Args:
            content (list[int]): A list of numbers, starting from top left, going row by row from left to right.
            Empty cells should have the value 0.
        """
        self.grid = np.reshape(content, (9, 9))
        self.solver = solving_algorithm
        assert (
            math.sqrt(len(content)) == 9
        ), "Sudokus that are not of grid size 9x9 are currently not supported"
        assert (
            max(content) <= 9
        ), "Sudokus of grid size 9x9 must not have numbers larger than 9"
        assert (
            min(content) >= 0
        ), "Sudokus of grid size 1x1 must not have numbers smaller than 0"

    def solve(self):
        self.solver.solve(self.grid)

    def visualize(self):
        light_grey = "lightgrey"
        dark_grey = "darkgrey"
        fig = go.Figure(
            data=[
                go.Table(
                    cells=dict(
                        values=self.grid.transpose(),
                        align="center",
                        height=60,
                        fill_color=[
                            [
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                            ],
                            [
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                            ],
                            [
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                            ],
                            [
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                            ],
                            [
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                            ],
                            [
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                            ],
                            [
                                light_grey,
                                light_grey,
                                light_grey,
                                dark_grey,
                                dark_grey,
                                dark_grey,
                                light_grey,
                                light_grey,
                                light_grey,
                            ],
                        ],
                    ),
                    columnwidth=30,
                )
            ]
        )
        fig.for_each_trace(lambda t: t.update(header_fill_color="rgba(0,0,0,0)"))
        fig.layout["template"]["data"]["table"][0]["header"]["fill"][
            "color"
        ] = "rgba(0,0,0,0)"
        fig.layout["template"]["data"]["table"][0]["header"]["line"][
            "color"
        ] = "rgba(0,0,0,0)"
        fig.update_layout(height=900)
        return fig


class SolverStrategy(ABC):
    @abstractmethod
    def solve(self, sudoku: SudokuGrid):
        ...


class BruteForceStrategy(SolverStrategy):
    """Brute force with backtracking solver, based on implementation from https://www.techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking/"""

    def find_empty(self, grid: np.ndarray) -> tuple(int, int):
        """Find first non pre-filled cell (cell that contains a 0) and return corresponding indices. If no empty cell was found, return (-1, -1)

        Args:
            grid (np.ndarray): grid representation of the sudoku board

        Returns:
            tuple (int, int): i and j indices of the first encountered emtpy cell
        """
        for idx, elem in np.ndenumerate(grid):
            if elem == 0:
                return idx
        return (-1, -1)

    def solve(self, grid: np.ndarray) -> bool:
        """Backtracking algorithm using recursion.

        Args:
            sudoku (SudokuGrid): grid to solve.
            Unfilled cells are assumed to have value '0'
            Will update values in-place if solution exists.

        Returns:
            bool: True if solved, False if not solvable
        """
        empty_cell_idx = self.find_empty(grid)
        if empty_cell_idx == (-1, -1):
            return True
        row, col = empty_cell_idx
        for i in range(1, 10):
            if self.is_valid(grid, i, (row, col)):
                grid[row][col] = i
                if self.solve(grid):
                    return True
                grid[row][col] = 0
        # print("Backtracking...")
        return False

    def is_valid(self, grid, value, position) -> bool:
        """Check if adding value at position creates a valid sudoku board, by checking row rule, column rule and neighbourhood rule

        Args:
            grid (_type_): sudoku grid with empty cells having value '0'
            value (_type_): value that has been entered in cell at position 'position'
            position (_type_): position of the cell we inspect

        Returns:
            bool: True, if board is valid given that configuration, else False. Does not change board state.
        """
        # check row rule is not violated
        for i in range(0, 9):
            if grid[position[0]][i] == value and i != position[1]:
                # print(
                #     f"Filling cell at row: {position[0]}, column: {position[1]} with value: {value} violates row rule"
                # )
                return False

        # check column rule is not violated
        for i in range(0, 9):
            if grid[i][position[1]] == value and i != position[0]:
                # print(
                #     f"Filling cell at row: {position[0]}, column: {position[1]} with value: {value} violates row rule"
                # )
                return False

        # check neighbourhood rule is not violated
        # cells (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2) belong to neihgbourhood 0, 0
        # cells (3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (5,0), (5,1), (5,2) belong to neighbourhood 1, 0
        # etc.
        neighbourhood_y, neighbourhood_x = [elem // 3 for elem in position]

        for i in range(neighbourhood_y * 3, neighbourhood_y * 3 + 3):
            for j in range(neighbourhood_x * 3, neighbourhood_x * 3 + 3):
                if grid[i][j] == value and (i, j) != position:
                    # print(
                    #     f"Filling cell at row: {position[0]}, column: {position[1]} with value: {value} violates row rule"
                    # )
                    return False
        return True


class HeuristicsStrategy(SolverStrategy):
    def solve(self, sudoku: SudokuGrid):
        raise NotImplementedError

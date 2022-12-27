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
        self.grid = np.reshape(content, (9, 9)).transpose()
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
        self.solver.solve()

    def visualize(self):
        light_grey = "lightgrey"
        dark_grey = "darkgrey"
        fig = go.Figure(
            data=[
                go.Table(
                    cells=dict(
                        values=self.grid,
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
        fig.update_layout(height=3000)
        return fig


class SolverStrategy(ABC):
    @abstractmethod
    def solve(self, sudoku: SudokuGrid):
        ...


class BruteForceStrategy(SolverStrategy):
    def solve(self, sudoku: SudokuGrid):
        # raise NotImplementedError
        grid = sudoku.grid
        is_fixed = np.where(grid > 1, True, False)
        idx = 0
        idy = 0
        while True:
            if is_fixed[idx, idy]:
                idx += 1

        # for idx, elem in np.ndenumerate(grid):
        #    if is_fixed[idx]:
        #        continue
        #    if


class HeuristicsStrategy(SolverStrategy):
    def solve(self, sudoku: SudokuGrid):
        raise NotImplementedError

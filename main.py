from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.consts import dummy_grid
from src.dataloader import DataLoader
from src.img_proc import img2grid
from src.sudoku import BruteForceStrategy, SudokuGrid

st.title("Sudoku Solver")
dl = DataLoader(Path("data/raw"))
dummy_data = next(iter(dl))

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    # bytes_data = img_file_buffer.getvalue()
    # cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.imread(str(dummy_data["img_path"]))
    img_file_buffer = st.image(img)  # delete me later
    grid = img2grid(img)
    grid = dummy_data["input"]
    sudoku = SudokuGrid(grid, solving_algorithm=BruteForceStrategy())
    st.write("Parsed Sudoku:")
    st.plotly_chart(sudoku.visualize())
    sudoku.solve()
    if 0 in sudoku.grid:
        st.write("Could not solve Sudoku")
    else:
        with st.expander("Solved Sodoku:"):
            st.plotly_chart(sudoku.visualize())

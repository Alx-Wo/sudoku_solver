import json
import math

import cv2
import numpy as np
import torch
import yaml
from skimage.segmentation import clear_border
from torchvision import transforms

from src.models.mnist_model import Net


def show_image(img: np.ndarray, win_name: str) -> None:
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyWindow(win_name)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_corners(contour):
    mycont = contour.squeeze()
    min_value = 9999
    max_value = 0
    x2_value = -9999
    y2_value = -9999
    bottomright = None
    topleft = None
    bottomleft = None
    topright = None
    for elem in mycont:
        if elem[0] + elem[1] < min_value:
            min_value = elem[0] + elem[1]
            topleft = elem
        if elem[0] + elem[1] > max_value:
            max_value = elem[0] + elem[1]
            bottomright = elem
        if elem[0] - elem[1] > x2_value:
            x2_value = elem[0] - elem[1]
            topright = elem
        if elem[1] - elem[0] > y2_value:
            y2_value = elem[1] - elem[0]
            bottomleft = elem
    return topleft, bottomright, topright, bottomleft


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    # compute perspective transform matrix and apply it
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped


# Unused
# def get_contours(im_src):
#     contours, hierarchy = cv2.findContours(
#         im_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#     )
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 30000:
#             topleft, bottomright, topright, bottomleft = get_corners(contour)
#             return topleft, topright, bottomleft, bottomright


def get_contours_visualized(
    im_src: np.ndarray, im_dest: np.ndarray, area_thresholds: tuple[int, int]
):
    contours, hierarchy = cv2.findContours(
        im_src, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        # TODO: Parameterize thresholds with trackbars!
        if (
            area > im_src.shape[0] * im_src.shape[1] * area_thresholds[0]
            and area < im_src.shape[0] * im_src.shape[1] * area_thresholds[1]
        ):
            topleft, bottomright, topright, bottomleft = get_corners(contour)
            cv2.drawContours(im_dest, contour, -1, (0, 255, 0), 3)
            cv2.circle(im_dest, (bottomright[0], bottomright[1]), 10, (255, 0, 0), 20)
            cv2.circle(im_dest, (topleft[0], topleft[1]), 10, (0, 255, 0), 20)
            cv2.circle(im_dest, (bottomleft[0], bottomleft[1]), 10, (0, 0, 0), 20)
            cv2.circle(im_dest, (topright[0], topright[1]), 10, (255, 255, 255), 20)
            return im_dest, topleft, topright, bottomleft, bottomright


def concat_images(*images, rescale_factor):
    resized_img = cv2.resize(
        images[0],
        (
            int(images[0].shape[1] * rescale_factor),
            int(images[0].shape[0] * rescale_factor),
        ),
    )
    if len(images[0].shape) == 2:
        stacked_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    else:
        stacked_img = resized_img
    for i in range(1, len(images)):
        resized_img = cv2.resize(
            images[i],
            (
                int(images[i].shape[1] * rescale_factor),
                int(images[i].shape[0] * rescale_factor),
            ),
        )
        if len(images[i].shape) == 2:
            image = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        else:
            image = resized_img
        stacked_img = np.hstack((stacked_img, image))
    return stacked_img


def interactive_rectification(
    im_src: np.ndarray, yaml_params: dict, yaml_save_path: str
):
    def empty(empty):
        pass

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 512, 440)
    cv2.createTrackbar(
        "ApplyThresholding",
        "Parameters",
        yaml_params["Apply GrayValue Thresholding"],
        1,
        empty,
    )
    # cv2.createTrackbar("HistogramSpread", "Parameters", 0, 255, empty)
    cv2.createTrackbar(
        "Threshold1", "Parameters", yaml_params["GrayValueThreshold1"], 255, empty
    )
    cv2.createTrackbar(
        "Threshold2", "Parameters", yaml_params["GrayValueThreshold2"], 255, empty
    )
    cv2.createTrackbar(
        "CannyThreshold1", "Parameters", yaml_params["CannyThreshold1"], 255, empty
    )
    cv2.createTrackbar(
        "CannyThreshold2", "Parameters", yaml_params["CannyThreshold2"], 255, empty
    )
    cv2.createTrackbar(
        "DilIterations", "Parameters", yaml_params["DilationIterations"], 10, empty
    )
    cv2.createTrackbar(
        "MinAreaThreshold", "Parameters", yaml_params["MinAreaThreshold"], 100, empty
    )
    cv2.createTrackbar(
        "MaxAreaThreshold", "Parameters", yaml_params["MaxAreaThreshold"], 100, empty
    )
    cv2.createTrackbar("PixelCrop", "Parameters", yaml_params["PixelCrop"], 200, empty)
    cv2.createTrackbar("SaveParameters", "Parameters", 0, 1, empty)
    while True:
        im_border = cv2.copyMakeBorder(
            im_src, 15, 15, 15, 15, cv2.BORDER_CONSTANT, (0, 0, 0)
        )
        im_contour = im_border.copy()
        im_blur = cv2.GaussianBlur(im_border, (7, 7), 1)  # blur
        im_gray = cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)  # gray

        hist_spread_param = cv2.getTrackbarPos("HistogramSpread", "Parameters")
        # improve contrasts using hist equalization
        im_contrast_enhanced = cv2.equalizeHist(im_gray)

        # improve contrasts using CLAHE
        clahe = cv2.createCLAHE(clipLimit=hist_spread_param, tileGridSize=(8, 8))
        im_contrast_enhanced = clahe.apply(im_gray)

        # apply fixed pixel value thresholding
        apply_thresholding = cv2.getTrackbarPos("ApplyThresholding", "Parameters")
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        if apply_thresholding:
            val, im_thresholded = cv2.threshold(
                im_contrast_enhanced, threshold1, threshold2, 1
            )
        else:
            im_thresholded = im_contrast_enhanced

        # apply canny edge detection
        canny_threshold1 = cv2.getTrackbarPos("CannyThreshold1", "Parameters")
        canny_threshold2 = cv2.getTrackbarPos("CannyThreshold2", "Parameters")
        im_canny = cv2.Canny(im_thresholded, canny_threshold1, canny_threshold2)

        # apply dilation to fill gaps between edges
        kernel = np.ones((5, 5))
        dil_iteration = cv2.getTrackbarPos("DilIterations", "Parameters")
        im_dil = cv2.dilate(im_canny, kernel, iterations=dil_iteration)

        # crop border pixels
        crop_pixel = cv2.getTrackbarPos("PixelCrop", "Parameters") - 100
        min_area_threshold = cv2.getTrackbarPos("MinAreaThreshold", "Parameters") / 100
        max_area_threshold = cv2.getTrackbarPos("MaxAreaThreshold", "Parameters") / 100
        try:
            (
                im_contour,
                topleft,
                topright,
                bottomleft,
                bottomright,
            ) = get_contours_visualized(
                im_dil, im_contour, (min_area_threshold, max_area_threshold)
            )
            # make rectangle points smaller, because border information is not very important to us
            topleft = [topleft[0] + crop_pixel, topleft[1] + crop_pixel]
            topright = [topright[0] - crop_pixel, topright[1] + crop_pixel]
            bottomleft = [bottomleft[0] + crop_pixel, bottomleft[1] - crop_pixel]
            bottomright = [bottomright[0] - crop_pixel, bottomright[1] - crop_pixel]

            warped = four_point_transform(
                im_src, np.asarray(([topleft, topright, bottomleft, bottomright]))
            )
        except TypeError:
            print("Could not get area with sufficient size to fetch corner points from")
            warped = im_src

        collage = concat_images(
            im_border,
            im_gray,
            im_thresholded,
            im_canny,
            im_dil,
            im_contour,
            rescale_factor=0.25,
        )
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", collage)
        cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
        cv2.imshow("Warped", warped)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            save_flag = cv2.getTrackbarPos("SaveParameters", "Parameters")
            if save_flag:
                yaml_data = {
                    "Apply GrayValue Thresholding": apply_thresholding,
                    "GrayValueThreshold1": threshold1,
                    "GrayValueThreshold2": threshold2,
                    "CannyThreshold1": canny_threshold1,
                    "CannyThreshold2": canny_threshold2,
                    "DilationIterations": dil_iteration,
                    "PixelCrop": crop_pixel + 100,
                    "MinAreaThreshold": int(min_area_threshold * 100),
                    "MaxAreaThreshold": int(max_area_threshold * 100),
                }
                with open(yaml_save_path, "w") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False)
            cv2.destroyWindow("Result")
            cv2.destroyWindow("Warped")
            cv2.destroyWindow("Parameters")
            return warped


def detect_sudoku(img_src: np.ndarray, yaml_params: dict) -> np.ndarray:
    im_border = cv2.copyMakeBorder(
        img_src, 15, 15, 15, 15, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    im_contour = im_border.copy()
    im_blur = cv2.GaussianBlur(im_border, (7, 7), 1)  # blur
    im_gray = cv2.cvtColor(im_blur, cv2.COLOR_BGR2GRAY)  # gray

    # improve contrasts using hist equalization
    im_contrast_enhanced = cv2.equalizeHist(im_gray)

    # improve contrasts using CLAHE
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(8, 8))
    im_contrast_enhanced = clahe.apply(im_gray)

    # apply fixed pixel value thresholding
    apply_thresholding = yaml_params["Apply GrayValue Thresholding"]
    threshold1 = yaml_params["GrayValueThreshold1"]
    threshold2 = yaml_params["GrayValueThreshold2"]
    if apply_thresholding:
        val, im_thresholded = cv2.threshold(
            im_contrast_enhanced, threshold1, threshold2, 1
        )
    else:
        im_thresholded = im_contrast_enhanced

    # apply canny edge detection
    canny_threshold1 = yaml_params["CannyThreshold1"]
    canny_threshold2 = yaml_params["CannyThreshold2"]
    im_canny = cv2.Canny(im_thresholded, canny_threshold1, canny_threshold2)

    # apply dilation to fill gaps between edges
    kernel = np.ones((5, 5))
    dil_iteration = yaml_params["DilationIterations"]
    im_dil = cv2.dilate(im_canny, kernel, iterations=dil_iteration)

    # crop border pixels
    crop_pixel = yaml_params["PixelCrop"] - 100
    min_area_threshold = yaml_params["MinAreaThreshold"] / 100
    max_area_threshold = yaml_params["MaxAreaThreshold"] / 100
    try:
        (
            im_contour,
            topleft,
            topright,
            bottomleft,
            bottomright,
        ) = get_contours_visualized(
            im_dil, im_contour, (min_area_threshold, max_area_threshold)
        )
        # make rectangle points smaller, because border information is not very important to us
        topleft = [topleft[0] + crop_pixel, topleft[1] + crop_pixel]
        topright = [topright[0] - crop_pixel, topright[1] + crop_pixel]
        bottomleft = [bottomleft[0] + crop_pixel, bottomleft[1] - crop_pixel]
        bottomright = [bottomright[0] - crop_pixel, bottomright[1] - crop_pixel]

        warped = four_point_transform(
            img_src, np.asarray(([topleft, topright, bottomleft, bottomright]))
        )
    except TypeError:
        print("Could not get area with sufficient size to fetch corner points from")
        warped = img_src
    return warped


def apply_grid(img: np.ndarray, offset: int = 15) -> list[np.ndarray]:
    """Add a grid on top of the rectified puzzle.

    Return a list of each cell as image,
    from top left to bottom right, row by row.
    Only works for 9x9 sudokus for now.

    Args:
        img (np.ndarray): rectified sudoku puzzle

    Returns:
        list[np.ndarray]: list of puzzle cells
    """
    img_height, img_width = img.shape[:2]
    x_step_size = math.floor(img_width / 9)
    y_step_size = math.floor(img_height / 9)
    result = []
    for i in range(9):
        for j in range(9):
            cell = img[
                max((y_step_size * i) - offset, 0) : min(
                    (y_step_size * (i + 1)) + offset, img_height
                ),
                max((x_step_size * j) - offset, 0) : min(
                    (x_step_size * (j + 1)) + offset, img_width
                ),
            ]
            result.append(cell)
    return result


def preprocess_cell(img: np.ndarray) -> np.ndarray | None:
    # Gray, blur, contrast enhancement
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
    _, im_thresholded = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    im_thresholded = clear_border(im_thresholded)
    # im_canny = cv2.Canny(im_thresholded, 93, 177)
    # kernel = np.ones((3, 3))
    # dil_iteration = 1
    # im_dil = cv2.dilate(im_canny, kernel, iterations=dil_iteration)

    contours, _ = cv2.findContours(
        im_thresholded, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
    )
    # im_contours = img.copy()
    max_area = 0
    max_contour = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    # print(f"Rel area: {max_area / (img.shape[0] * img.shape[1])}")
    # if max_area > (img.shape[0] * img.shape[1]) * 0.04:
    #     cv2.drawContours(im_contours, max_contour, -1, (0, 255, 0), 3)
    # else:
    #     print("Contour too small to be a number")
    mask = np.zeros(im_thresholded.shape, dtype="uint8")
    if max_area > (img.shape[0] * img.shape[1]) * 0.04:
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        # show_image(mask, "Mask")
        # show_image(im_thresholded, "Thresholded Image")
        x, y, w, h = cv2.boundingRect(max_contour)

        digit = cv2.bitwise_and(im_thresholded, im_thresholded, mask=mask)
        digit = digit[y : y + h, x : x + w]
        digit = cv2.copyMakeBorder(
            digit,
            top=10,
            bottom=10,
            left=10,
            right=10,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        show_image(digit, "Masked Image")
        return digit
        # show_image(digit, "Masked Image")
    # collage = concat_images(
    #     img,
    #     img_gray,
    #     # img_contrast_enhanced,
    #     im_thresholded,
    #     im_canny,
    #     im_dil,
    #     im_contours,
    #     rescale_factor=0.3,
    # )

    # show_image(collage, "Cell processing")
    return None


def classify_cell(img: np.ndarray, model: torch.nn.Module):
    cell = preprocess_cell(img)
    if cell is not None:
        # classify cell with MNIST model
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        cell = cv2.resize(cell, (28, 28))
        cell = transform(cell)
        cell = cell[None, :]
        cell = cell.to("cpu")
        output = model(cell)
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item()
    else:
        return 0


if __name__ == "__main__":
    # img_path = "data/raw/1672131911403.jpg"
    # img_path = "data/raw/1672131911393.jpg"
    # img_path = "data/raw/1672131911380.jpg"
    # img_path = "data/raw/1672131911368.jpg"
    # img_path = "data/raw/1672131911352.jpg"  # This one will not work, because sudoku does not make up more than 30% of image area
    img_path = "data/raw/1672131911333.jpg"
    img_path = "data/raw/1672131911403.jpg"
    img = cv2.imread(img_path)
    with open("tmp_params.yaml", "r", encoding="utf-8") as fid:
        yaml_params = yaml.load(fid, Loader=yaml.loader.SafeLoader)
    # interactive_rectification(
    #    img,
    #    yaml_params=yaml_params,
    #    yaml_save_path="tmp_params.yaml",
    # )
    puzzle_img = detect_sudoku(img, yaml_params)
    # show_image(puzzle_img, "Detected Puzzle")

    cells = apply_grid(puzzle_img)
    # for idx, cell_img in enumerate(cells):
    #     show_image(cell_img, f"cell ({idx // 9}, {idx % 9})")

    model = Net().to("cpu")
    weights = torch.load("mnist_cnn_v1.pt")
    model.load_state_dict(weights)
    puzzle = []
    for cell in cells:
        puzzle.append(classify_cell(cell, model))
    print(puzzle)
    with open("data/raw/1672131911333.json", "r", encoding="utf-8") as fid:
        dat = json.load(fid)
    print(dat["input"])

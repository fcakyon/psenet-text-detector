import os
import cv2
import copy
import gdown
import random
import numpy as np
from PIL import Image


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def scale_image(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def download(url: str, save_path: str):
    """
    Downloads file from gdrive, shows progress.
    Example inputs:
        url: 'ftp://smartengines.com/midv-500/dataset/01_alb_id.zip'
        save_path: 'data/file.zip'
    """

    # create save_dir if not present
    create_dir(os.path.dirname(save_path))
    # download file
    gdown.download(url, save_path, quiet=False)


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def select_random_color():
    """
    Selects random color.
    """
    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255],
              [255, 255, 0], [255, 0, 255], [80, 70, 180], [250, 80, 190],
              [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    return colors[random.randrange(0, 10)]


def visualize_detection(image_path,
                        image: np.array,
                        quads: list,
                        output_dir: str = None,
                        file_name: str = "result_",
                        color: tuple = (0, 0, 0)):
    """
    Draws given quads onto given image and exports to given output path.
    Returns the resultant image.
    """
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # get result file name and path
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    res_img_file = os.path.join(output_dir, "result_" + filename + '.png')

    # select random color if not specified
    if color == (0, 0, 0):
        color = select_random_color()

    # draw quads over image
    for quad in quads:
        # convert to numpy array
        np_quad = np.array([[pair] for pair in quad])
        # draw quad
        cv2.drawContours(image, [np_quad], 0, color, 3)

    # export image if output_dir is given
    if output_dir is not None:
        # create output folder if not present
        create_dir(output_dir)
        # export the image with drawns quads on top of it
        cv2.imwrite(res_img_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # return result
    return image


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def rectify_boxes(image: np.array, boxes: list):
    """
    Unwarps given rotated boxes to rectangles and returns resulting image.
    """
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # rectify quads
    rectified_boxes = []
    for ind, box in enumerate(boxes):
        pts = np.array(box, dtype="float32")
        # unwarp four point region to rectangle
        rectified_box = four_point_transform(image, pts)
        rectified_boxes.append(rectified_box)

    # return the rectified quads
    return rectified_boxes


def export_rectified_boxes(image_path,
                           rectified_quads,
                           output_dir: str = "output/"):
    """
    Exports the given rectified boxes and returns their file paths.
    """
    # get file name
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))

    # create crops dir
    crops_dir = os.path.join(output_dir, file_name + "_crops")
    create_dir(crops_dir)

    exported_file_paths = []
    for ind, rectified_quad in enumerate(rectified_quads):
        # get save path
        save_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        # export rectified quad
        cv2.imwrite(save_path, cv2.cvtColor(rectified_quad, cv2.COLOR_RGB2BGR))
        # note export path
        exported_file_paths.append(save_path)

    return exported_file_paths


def export_detected_regions(image_path, image, boxes, output_dir="output/"):
    # fix rotation
    rectified_boxes = rectify_boxes(image, boxes)
    exported_file_paths = export_rectified_boxes(image_path,
                                                 rectified_boxes,
                                                 output_dir)
    return exported_file_paths

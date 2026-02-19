import platform
import os
import cv2
from tool_algorithm import get_parameter_number_anwser



# ============================================ MERGE IMAGES =======================================
def mergeImages(filename, coord_array, array_img_graft, background_image, imgInfo, args):
    file_extension = filename.split(".")[-1]
    filename_cut = filename.rsplit(".", 1)[0]
    height, width, _ = imgInfo.shape
    background_image[0: 0 + height, 500: 500 + width] = imgInfo / 255
    for i in range(len(coord_array)):
        x_coord = coord_array[i][0]
        y_coord = coord_array[i][1]
        height, width, _ = array_img_graft[i].shape
        background_image[y_coord: y_coord + height, x_coord: x_coord + width] = array_img_graft[i] / 255
    handled_scored_img = f"images/answer_sheets/{args.input}/HandledSheets/handled_{filename_cut}.{file_extension}"
    cv2.imwrite(handled_scored_img, background_image * 255)
    return handled_scored_img


# ============================================ CUT IMAGE COLUMN ANSWER =======================================
def crop_image_answer(img, numberAnswer):
    ans_blocks = []
    arrayX = [30, 350, 660]
    for i in range(3):
        y = 480
        width = 350
        height = 896
        cropped_image = img[y: y + height, arrayX[i]: arrayX[i] + width]
        ans_blocks.append(cropped_image)
    sorted_ans_blocks = ans_blocks
    if numberAnswer == 20 or numberAnswer == 40 or numberAnswer == 60:
        numbers_question_pictures = get_parameter_number_anwser(numberAnswer)
    else:
        numbers_question_pictures = get_parameter_number_anwser(numberAnswer) + 1
    sorted_ans_blocks = sorted_ans_blocks[:numbers_question_pictures]
    sorted_ans_blocks_resize = []
    coord_array = []
    size_array = []
    for i, sorted_ans_block in enumerate(sorted_ans_blocks):
        img2 = sorted_ans_block[0]
        width, height = img2.shape
        size_array.append((350, 896))
        img_resize = cv2.resize(sorted_ans_block, (250, 640), interpolation=cv2.INTER_AREA)
        # cv2.imshow("answersheet", img_resize)
        # cv2.waitKey(0)
        sorted_ans_blocks_resize.append(img_resize)
        coord_array.append((arrayX[i], 480))

    return sorted_ans_blocks_resize, size_array, coord_array


# ============================================ CUT IMAGE INFO =======================================

def crop_image_info(img):
    left = 500
    top = 0
    right = 1006
    bottom = 500
    cropped_image = img[top:bottom, left:right]
    cropped_image = cv2.convertScaleAbs(cropped_image * 255)
    img_resize = cv2.resize(cropped_image, (640, 640), interpolation=cv2.INTER_AREA)
    return img_resize
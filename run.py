import cv2 as cv
import numpy as np
from easyocr.easyocr import *
from PIL import ImageFont, ImageDraw, Image

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)

def put_test(img, str_text, filename):

    # 한글 깨짐
    # font = cv.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10, 500)
    # fontScale = 1
    # fontColor = (0, 0, 255)
    # thickness = 1
    # lineType = 2
    #
    # cv.putText(img, str_text,
    #             bottomLeftCornerOfText,
    #             font,
    #             fontScale,
    #             fontColor,
    #             thickness,
    #             lineType)
    bottomLeftCornerOfText = (10, 500)
    font = ImageFont.truetype("fonts/gulim.ttc", 20)
    ##text_img = np.full((200,300,3), (0, 0, 255), np.unit8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text(bottomLeftCornerOfText, str_text, font=font, fill=(0,0,0))
    img = np.array(img)
    cv.imshow(filename, img)



if __name__ == '__main__':

    # # Using default model
    # reader = Reader(['en'], gpu=True)

    # Using custom model case english
    # reader = Reader(['en'], gpu=False,
    #                 model_storage_directory='model',
    #                 user_network_directory='user_network',
    #                 recog_network='custom_en')

    # Using custom model case english
    reader = Reader(['ko','en'], gpu=False,
                    model_storage_directory='model',
                    user_network_directory='user_network',
                    recog_network='custom_ko')

    files, count = get_files('demo_image')  #orig 'examples'

    for idx, file in enumerate(files):
        filename = os.path.basename(file)

        result = reader.readtext(file)

        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))
            print('bbox: ', bbox)
            img = cv.imread(file)
            cv.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 0, 255), 3)
            # put_test(img, string, filename)

            bottomLeftCornerOfText = (10, 500)
            font = ImageFont.truetype("fonts/gulim.ttc", 50)
            ##text_img = np.full((200,300,3), (0, 0, 255), np.unit8)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text(bottomLeftCornerOfText, string, font=font, fill=(0, 0, 255))
            img = np.array(img)

            cv.imshow(filename, img)
            cv.waitKey(0)
            cv.destroyWindow(filename)

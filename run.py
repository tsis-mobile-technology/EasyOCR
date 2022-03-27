import cv2 as cv
import numpy as np
from easyocr.easyocr import *
from PIL import ImageFont, ImageDraw, Image
from video_processing_parallel import WebcamStream
import time
from threading import Thread  # library for implementing multi-threaded processing

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def get_files(path):
    file_list = []

    lists = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    lists.sort()
    abspath = os.path.abspath(path)
    for onelist in lists:
        file_path = os.path.join(abspath, onelist)
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

def get_text(img_color):
    result = reader.readtext(img_color)
    print("result.__len__():", result.__len__())
    if result.__len__() > 0:
        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for idx, (bbox, string, confidence) in enumerate(result):
            print("confidence: %.4f, string: '%s'" % (confidence, string))
            # print('bbox: ', bbox)
            if True:
                cv.rectangle(img_color, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 0, 255),
                             3)
                # put_test(img, string, filename)

                bottomLeftCornerOfText = (int(bbox[0][0]), int(bbox[0][1]) - 30)
                # font = ImageFont.truetype("fonts/gulim.ttc", 20)
                ##text_img = np.full((200,300,3), (0, 0, 255), np.unit8)
                img_color = Image.fromarray(img_color)
                draw = ImageDraw.Draw(img_color)
                w, h = font.getsize(string)
                draw.rectangle((int(bbox[0][0]), int(bbox[0][1]) - 30, int(bbox[0][0]) + w, int(bbox[0][1]) - 30 + h),
                               fill='gray')
                draw.text(bottomLeftCornerOfText, string, font=font, fill=(0, 255, 255))
                img_color = np.array(img_color)
                if idx == 4:
                    cv.imshow("test", img_color)
                    cv.waitKey(0)
                    cv.destroyWindow("test")

if __name__ == '__main__':
    if False:

        # cap = cv.VideoCapture(0)
        webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
        webcam_stream.start()
        reader = Reader(['ko', 'en'], gpu=False,
                        model_storage_directory='./model',
                        user_network_directory='./user_network',
                        # recog_network='TPS-ResNet-BiLSTM-CTC-0311-wild')
                        # # recog_network='TPS-ResNet-BiLSTM-Attn-0316-wild')
                        recog_network='TPS-ResNet-BiLSTM-Attn-wild-syllable-0317')
                        # recog_network='TPS-ResNet-BiLSTM-CTC-syllable-word-0316')
        font = ImageFont.truetype("fonts/gulim.ttc", 30)
        while(True):
            # ret, img_color = cap.read()
            img_color = webcam_stream.read()
            height, width = img_color.shape[:2]
            img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
            # img_color = cv.flip(img_color, 1)  # 좌우반전
            delay = 0.03  # delay value in seconds. so, delay=1 is equivalent to 1 second
            # delay = 2
            time.sleep(delay)
            # cv.imshow("test", img_color)
            # t = Thread(target=get_text, args=(img_color,))
            # t.start()
            key = cv.waitKey(1)
            if key == ord('s'):
                get_text(img_color)
                # in thread
                # t = Thread(target=get_text, args=(img_color,))
                # t.start()
                """
                result = reader.readtext(img_color)
                print("result.__len__():", result.__len__())
                if result.__len__() > 0:
                    # ./easyocr/utils.py 733 lines
                    # result[0]: bbox
                    # result[1]: string
                    # result[2]: confidence
                    for (bbox, string, confidence) in result:
                        print("confidence: %.4f, string: '%s'" % (confidence, string))
                        # print('bbox: ', bbox)
                        cv.rectangle(img_color, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (0, 0, 255),3)
                        # put_test(img, string, filename)

                        bottomLeftCornerOfText = (int(bbox[0][0]), int(bbox[0][1])-30)
                        # font = ImageFont.truetype("fonts/gulim.ttc", 20)
                        ##text_img = np.full((200,300,3), (0, 0, 255), np.unit8)
                        img_color = Image.fromarray(img_color)
                        draw = ImageDraw.Draw(img_color)
                        w, h = font.getsize(string)
                        draw.rectangle((int(bbox[0][0]), int(bbox[0][1])-30, int(bbox[0][0]) + w, int(bbox[0][1])-30 + h), fill='gray')
                        draw.text(bottomLeftCornerOfText, string, font=font, fill=(0, 255, 255))
                        img_color = np.array(img_color)
                        cv.imshow("test", img_color)
                        cv.waitKey(0)
                        cv.destroyWindow("test")
                """
            # cv.waitKey(0)
            # ESC 키누르면 종료
            elif key == 27:
                cv.destroyWindow("test")
                break
            else:
                cv.imshow("test", img_color)
                continue

        webcam_stream.stop()
        cv.destroyAllWindows()
    else :
        # # Using default model
        # reader = Reader(['en'], gpu=True)

        # Using custom model case english
        # reader = Reader(['en'], gpu=False,
        #                 model_storage_directory='model',
        #                 user_network_directory='user_network',
        #                 recog_network='custom_en')

        # Using custom model case english
        # reader = Reader(['ko','en'], gpu=False,
        #                 model_storage_directory='model',
        #                 user_network_directory='user_network',
        #                 recog_network='custom_ko')
        reader = Reader(['en', 'ko'], gpu=False,
                        model_storage_directory='../trainning_model/TPS-ResNet-BiLSTM-Attn-0316-wild',
                        # model_storage_directory='./model',
                        user_network_directory='./user_network',
                        # recog_network='craft_mlt_25k')
                        # # recog_network='TPS-ResNet-BiLSTM-CTC-0311-wild')
                        # recog_network='TPS-ResNet-BiLSTM-Attn-0316-wild')
                        # # recog_network='TPS-ResNet-BiLSTM-Attn-wild-syllable-0317')
                        # recog_network='TPS-ResNet-BiLSTM-CTC-syllable-word-wild-0316')
                        recog_network='best_accuracy')

        # files, count = get_files('../aihub_data/Text_in_the_wild/data/Goods/test')  #orig 'examples'
        files, count = get_files('examples')  # orig

        for idx, file in enumerate(files):
            filename = os.path.basename(file)
            print("file:", file )
            result = reader.readtext(file)

            # ./easyocr/utils.py 733 lines
            # result[0]: bbox
            # result[1]: string
            # result[2]: confidence
            img = cv.imread(file)
            for (bbox, string, confidence) in result:
                print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))
                print('bbox: ', bbox)
                # img = cv.imread(file)
                cv.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (255, 255, 0), 3)
                # put_test(img, string, filename)

                # bottomLeftCornerOfText = (10, 500)
                bottomLeftCornerOfText = (int(bbox[0][0]), int(bbox[0][1]) - 5)
                font = ImageFont.truetype("fonts/gulim.ttc", 15)
                ##text_img = np.full((200,300,3), (0, 0, 255), np.unit8)
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                w, h = font.getsize(string)
                draw.rectangle((int(bbox[0][0]), int(bbox[0][1]) - 5, int(bbox[0][0]) + w, int(bbox[0][1]) - 5 + h), fill='black')
                draw.text(bottomLeftCornerOfText, string, font=font, fill=(0, 0, 255))
                img = np.array(img)
            cv.imshow(filename, img)
            cv.waitKey(0)
            cv.destroyAllWindows()

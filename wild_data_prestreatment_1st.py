import json
from pprint import pprint
import matplotlib.pyplot as plt
import cv2

json_file = json.load(open("../Text_in_the_wild/textinthewild_data_info.json"))
pprint(json_file.keys()) # dict_keys(['info', 'images', 'annotations', 'licenses'])
pprint(json_file['info']) # {'name': 'Text in the wild Dataset', 'date_created': '2019-10-14 04:31:48'}
print(type(json_file['images'])) # <class 'list'>

print(json_file['images'][:3]) # [{'id': '00000001', 'width': 1920, 'height': 1440, 'file_name': 'FFF2F34A08347075F55E72B240EFE691.jpg', 'type': 'book'}, {'id': '00000002', 'width': 1920, 'height': 1440, 'file_name': 'FFDE6BAEADC9EDD31E51A1D1F687310F.jpg', 'type': 'book'}, {'id': '00000003', 'width': 1920, 'height': 1440, 'file_name': 'FFCFEB9E1D09544D6B458717DB4D6B7C.jpg', 'type': 'book'}]

if json_file['images'][0]['type'] == 'book' : # True
    print(True)
goods = [f for f in json_file['images'] if f['type']=='product']
print(len(goods)) #26358

annotation = [a for a in json_file['annotations'] if a['image_id'] == goods[0]['id'] and a['attributes']['class']=='word']
pprint(annotation[:3])

print(goods[0]['file_name'])

img = cv2.imread('/Users/gotaejong/ExternHard/97_Workspace/jupyter/Text_in_the_wild/Goods/'+goods[0]['file_name'])
plt.imshow(img)
# cv2.imshow(goods[0]['file_name'], img)
plt.show()
# cv2.waitKey(0)
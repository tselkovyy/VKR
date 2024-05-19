import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2

source_image_folders = ['ds/leftImg8bit/train', 'ds/leftImg8bit/val', 'ds/leftImg8bit/test']
destination_image_folders = ['ds/tikhon/images/train', 'ds/tikhon/images/val', 'ds/tikhon/images/test']
destination_label_folders = ['ds/tikhon/labels/train', 'ds/tikhon/labels/val', 'ds/tikhon/labels/test']
destination_mask_folders = ['ds/tikhon/masks/train', 'ds/tikhon/masks/val', 'ds/tikhon/masks/test']
class_mapping = ['person','rider','car','truck','bus','caravan','trailer','train','motorcycle','bicycle','ground','road','sidewalk']

def polygon_to_yolo_format(polygon):
    return [str(coord) for point in polygon for coord in point]

# Функция для преобразования объектов в требуемый формат
def objects_to_yolo_format(json_data, outfile,outmask):
    result = []
    for i, obj in enumerate(json_data['objects']):
        mask = np.zeros((json_data['imgHeight'], json_data['imgWidth']), np.uint8)
        label = obj["label"]
        if label not in class_mapping:
            # class_mapping[label] = len(class_mapping)  # Добавляем новый класс в отображение
            continue
        else:
            label_index = class_mapping.index(label)
        polygon = np.array(obj["polygon"])
        cv2.fillPoly(mask, pts=[polygon], color=255)
        cv2.imwrite(f'{outmask}_{i}_{label_index}.png',mask)
        polygon = polygon/np.array([json_data['imgWidth'],json_data['imgHeight']])
        polygon = np.clip(polygon,0.,1.)
        yolo_format = [str(label_index)] + polygon_to_yolo_format(polygon.tolist())
        result.append(' '.join(yolo_format))
    with open(outfile, 'a') as f:
        f.write('\n'.join(result))


for source_image_folder, destination_image_folder, destination_label_folder,destination_mask_folder in zip(source_image_folders, destination_image_folders, destination_label_folders, destination_mask_folders):
    # Создайте папки, если они еще не существуют
    os.makedirs(destination_image_folder, exist_ok=True)
    os.makedirs(destination_label_folder, exist_ok=True)
    os.makedirs(destination_mask_folder, exist_ok=True)

    for root, dirs, files in os.walk(source_image_folder):
        for filename in tqdm(files):
            if filename.endswith('_leftImg8bit.png'):
                # Перемещение изображений
                shutil.copy(os.path.join(root, filename), destination_image_folder)

                # Поиск соответствующего JSON-файла
                json_filename = filename.replace('_leftImg8bit.png', '_gtFine_polygons.json')
                json_path = os.path.join(root, json_filename)
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        # Создание файла аннотации YOLO
                        objects_to_yolo_format(json_data, destination_label_folder+f'/{filename[:-4]}.txt', destination_mask_folder+f'/{filename[:-4]}')

print(len(class_mapping), class_mapping)
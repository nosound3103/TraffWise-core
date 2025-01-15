import os
import random
import xml.etree.ElementTree as ET
import shutil
import glob
import json
from PIL import Image
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def split_yolo_data(data_path):
    """Split YOLO dataset into train and validation sets

    Args:
        data_path (str): path to YOLO dataset
    """

    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    train_images_path = os.path.join(images_path, 'train')
    val_images_path = os.path.join(images_path, 'val')
    train_labels_path = os.path.join(labels_path, 'train')
    val_labels_path = os.path.join(labels_path, 'val')

    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    labels = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    for img, lbl in zip(train_images, train_labels):
        shutil.move(os.path.join(images_path, img), train_images_path)
        shutil.move(os.path.join(labels_path, lbl), train_labels_path)

    for img, lbl in zip(val_images, val_labels):
        shutil.move(os.path.join(images_path, img), val_images_path)
        shutil.move(os.path.join(labels_path, lbl), val_labels_path)


def yolo_to_labelme(yolo_path, label_dict):
    """Convert YOLO dataset to LabelMe dataset with folder

    Args:
        yolo_path (str): path to YOLO dataset
        labelme_path (str): path to LabelMe dataset
    """

    labelme_path = os.path.join(yolo_path, 'labelme')
    train_labelme_path = os.path.join(labelme_path, 'train')
    val_labelme_path = os.path.join(labelme_path, 'val')

    os.makedirs(labelme_path, exist_ok=True)
    os.makedirs(train_labelme_path, exist_ok=True)
    os.makedirs(val_labelme_path, exist_ok=True)

    train_yolo_labels = glob.glob(os.path.join(
        yolo_path, 'labels', 'train', '*.txt'))
    val_yolo_labels = glob.glob(os.path.join(
        yolo_path, 'labels', 'val', '*.txt'))

    for label in train_yolo_labels:
        labelme_file = os.path.join(
            train_labelme_path, os.path.basename(label).replace('.txt', '.json'))
        yolo_to_labelme_single(label, labelme_file, label_dict)

    for label in val_yolo_labels:
        labelme_file = os.path.join(
            val_labelme_path, os.path.basename(label).replace('.txt', '.json'))
        yolo_to_labelme_single(label, labelme_file, label_dict)


def yolo_to_labelme_single(yolo_label_path, labelme_file, label_dict):
    """Convert each YOLO annotation file to LabelMe annotation file

    Args:
        yolo_label_path (str): yolo label file path
        labelme_file (str): labelme file path
    """

    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()

    img_path = yolo_label_path.replace(
        'labels', 'images').replace('.txt', '.jpg')

    img_width, img_height = Image.open(img_path).size

    shutil.copy(img_path, os.path.join(os.path.dirname(
        labelme_file), os.path.basename(img_path)))

    annotations = []
    for line in lines:
        data = list(map(float, line.split()))
        if len(data) == 5:
            label_idx, x_center, y_center, width, height = data
            label = label_dict[int(label_idx)]
            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)

            points = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]

        else:
            label_idx, *points = data
            label = label_dict[int(label_idx)]
            x_points = points[::2]
            y_points = points[1::2]
            points = [[int(x * img_width), int(y * img_height)]
                      for x, y in zip(x_points, y_points)]

        annotation = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        annotations.append(annotation)

    labelme_content = {
        "version": "5.6.0",
        "flags": {},
        "shapes": annotations,
        "imagePath": os.path.basename(yolo_label_path).replace('.txt', '.jpg'),
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    with open(labelme_file, 'w') as f:
        json.dump(labelme_content, f)


def create_yolo_annotation_with_xml(xml_file_path, yolo_label_path, label_dict):
    """Create YOLO annotation file from Pascal VOC XML file

    Args:
        xml_file_path (str): xml file path
        yolo_label_path (str): yolo label file path
        label_dict (dict): label dictionary
    """

    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    annotations = []

    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in label_dict:
            return False

        label_idx = label_dict[label]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        annotations.append(
            f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(yolo_label_path, 'w') as f:
        f.write("\n".join(annotations))

    return True


def convert_pascal_voc_to_yolo(pascal_voc_path, yolo_path):
    """Convert Pascal VOC to yolo

    Args:
        pascal_voc_path (str): path to Pascal VOC dataset
        yolo_path (str): path to yolo dataset
    """

    yolo_dirs = [
        os.path.join(yolo_path, 'images', 'train'),
        os.path.join(yolo_path, 'images', 'val'),
        os.path.join(yolo_path, 'labels', 'train'),
        os.path.join(yolo_path, 'labels', 'val')
    ]

    for yolo_dir in yolo_dirs:
        os.makedirs(yolo_dir, exist_ok=True)

    jpeg_images_dir = os.path.join(
        pascal_voc_path, 'VOC2012_train_val', 'JPEGImages')
    annotations_dir = os.path.join(
        pascal_voc_path, 'VOC2012_train_val', 'Annotations')
    if not os.path.exists(jpeg_images_dir) or not os.path.exists(annotations_dir):
        raise FileNotFoundError(
            f"The directory {jpeg_images_dir} or {annotations_dir} does not exist. Please verify the dataset path.")

    image_filenames = os.listdir(jpeg_images_dir)
    image_ids = [os.path.splitext(
        filename)[0] for filename in image_filenames if filename.endswith('.jpg')]
    random.seed(42)
    random.shuffle(image_ids)
    split_index = int(0.8 * len(image_ids))

    train_ids = image_ids[:split_index]
    val_ids = image_ids[split_index:]

    label_dict = {
        'bus': 0,
        'car': 1,
        'motorbike': 2
    }

    for image_set, ids in [('train', train_ids), ('val', val_ids)]:
        for img_id in ids:
            img_src_path = os.path.join(jpeg_images_dir, f'{img_id}.jpg')
            label_dst_path = os.path.join(
                yolo_path, 'labels', image_set, f'{img_id}.txt')

            xml_file_path = os.path.join(annotations_dir, f'{img_id}.xml')
            if not os.path.exists(xml_file_path):
                print(
                    f"Warning: Annotation {xml_file_path} not found, skipping.")
                continue

            flag = create_yolo_annotation_with_xml(
                xml_file_path, label_dst_path, label_dict)

            if not flag:
                continue

            img_dst_path = os.path.join(
                yolo_path, 'images', image_set, f'{img_id}.jpg')
            shutil.copy(img_src_path, img_dst_path)

    yaml_content = f"""
    train: {os.path.join(yolo_path, 'images/train')}
    val: {os.path.join(yolo_path, 'images/val')}

    nc: {len(label_dict)}
    names: {list(label_dict.keys())}
    """

    with open(os.path.join(yolo_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    label_dict_reverse = {v: k for k, v in label_dict.items()}

    yolo_to_labelme(yolo_path, label_dict_reverse)


def convert_8_classes_dataset_to_yolo(data_path, yolo_path):
    """Convert vehicle detection 8 classes dataset to YOLO dataset

    Args:
        data_path (str): path to vehicle detection 8 classes dataset
        yolo_path (str): path to YOLO dataset
    """

    image_path = os.path.join(data_path, 'images')
    label_path = os.path.join(data_path, 'labels')

    os.makedirs(os.path.join(yolo_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'labels/val'), exist_ok=True)

    images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    labels = [f.replace('.jpg', '.txt') for f in images]

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    for img, lbl in zip(train_images, train_labels):
        shutil.copy(os.path.join(image_path, img),
                    os.path.join(yolo_path, 'images/train', img))
        shutil.copy(os.path.join(label_path, lbl),
                    os.path.join(yolo_path, 'labels/train', lbl))

    for img, lbl in zip(test_images, test_labels):
        shutil.copy(os.path.join(image_path, img),
                    os.path.join(yolo_path, 'images/val', img))
        shutil.copy(os.path.join(label_path, lbl),
                    os.path.join(yolo_path, 'labels/val', lbl))

    label_dict = {
        'auto': 0,
        'bus': 1,
        'car': 2,
        'lcv': 3,
        'motorcycle': 4,
        'multiaxle': 5,
        'tractor': 6,
        'truck': 7
    }

    label_dict_reverse = {v: k for k, v in label_dict.items()}

    yolo_to_labelme(yolo_path, label_dict_reverse)


def convert_by9xs_dataset_to_yolo(by9xs_path, yolo_path):
    """Convert BY9XS dataset to YOLO dataset

    Args:
        by9xs_path (str): path to BY9XS dataset
        yolo_path (str): path to YOLO dataset
    """

    image_path = os.path.join(by9xs_path, 'images')
    label_path = os.path.join(by9xs_path, 'labels')

    os.makedirs(os.path.join(yolo_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'labels/val'), exist_ok=True)

    images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    labels = [f.replace('.jpg', '.txt') for f in images]

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    for img, lbl in zip(train_images, train_labels):
        shutil.copy(os.path.join(image_path, img),
                    os.path.join(yolo_path, 'images/train', img))
        shutil.copy(os.path.join(label_path, lbl),
                    os.path.join(yolo_path, 'labels/train', lbl))

    for img, lbl in zip(test_images, test_labels):
        shutil.copy(os.path.join(image_path, img),
                    os.path.join(yolo_path, 'images/val', img))
        shutil.copy(os.path.join(label_path, lbl),
                    os.path.join(yolo_path, 'labels/val', lbl))

    label_dict = {
        'bus': 0,
        'car': 1,
        'motorbike': 2,
        'truck': 3
    }

    label_dict_reverse = {v: k for k, v in label_dict.items()}

    yolo_to_labelme(yolo_path, label_dict_reverse)


def merge_label_in_yolo_data(yolo_path, merge_dict):
    """Merge labels in YOLO dataset

    Args:
        yolo_path (str): path to YOLO dataset
        label_dict (str): label dictionary
    """
    label_files = glob.glob(os.path.join(yolo_path, 'labels', '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            label_idx, x_center, y_center, width, height = map(
                float, line.split())
            label = merge_dict[int(label_idx)]
            new_lines.append(
                f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(label_file, 'w') as f:
            f.write("\n".join(new_lines))


def merge_datasets(list_folders, output_folder):
    """Merge multiple datasets into one

    Args:
        list_folders (list): list of folders to merge
        output_folder (str): output folder
    """

    subfolders = ['images/train', 'images/val', 'labels/train', 'labels/val']

    for subfolder in subfolders:
        dest = os.path.join(output_folder, subfolder)
        os.makedirs(dest, exist_ok=True)

        for idx, source in enumerate(list_folders):
            src = os.path.join(source, subfolder)

            if os.path.exists(src):
                for file_name in os.listdir(src):
                    src_file = os.path.join(src, file_name)
                    dest_file = os.path.join(dest, file_name)

                    if os.path.exists(dest_file):
                        base, ext = os.path.splitext(file_name)
                        dest_file = os.path.join(
                            dest, f"{base}_source{idx + 1}{ext}")

                    shutil.copy(src_file, dest_file)


def convert_segment_to_detect(label_path):
    """Convert segmentation labels to detection labels

    Args:
        label_path (str): path to segmentation labels
    """

    label_files = glob.glob(os.path.join(label_path, "*", '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            data = list(map(float, line.split()))
            label_idx, *points = data
            x_points = points[::2]
            y_points = points[1::2]

            xmin = min(x_points)
            ymin = min(y_points)
            xmax = max(x_points)
            ymax = max(y_points)

            x_center = ((xmin + xmax) / 2)
            y_center = ((ymin + ymax) / 2)
            width = (xmax - xmin)
            height = (ymax - ymin)

            new_lines.append(
                f"{int(label_idx)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(label_file, 'w') as f:
            f.write("\n".join(new_lines))

# convert_pascal_voc_to_yolo(
#     pascal_voc_path='data/dataset/raw/Pascal VOC 2012/VOC2012_train_val',
#     yolo_path='data/dataset/labelme-yolo/pascal_voc')


# convert_8_classes_dataset_to_yolo(
#     data_path='data/dataset/raw/Vehicle detection 8 classes/train',
#     yolo_path='data/dataset/labelme-yolo/8-classes')


# merge_label_in_yolo_data(
#     yolo_path='data/dataset/raw/vehicle-detection-by9xs/train',
#     merge_dict={
#         0: 0,
#         1: 1,
#         2: 1,
#         3: 2,
#         4: 3,
#         5: 3,
#     }
# )

# convert_by9xs_dataset_to_yolo(
#     by9xs_path='data/dataset/raw/vehicle-detection-by9xs/train',
#     yolo_path='data/dataset/labelme-yolo/by9xs'
# )

merge_datasets(
    list_folders=[
        'data/dataset/labelme-yolo/pascal_voc',
        'data/dataset/labelme-yolo/by9xs',
        "data/dataset/labelme-yolo/coco"
    ],
    output_folder='data/dataset/combination'
)

# split_yolo_data(
#     data_path='data/dataset/labelme-yolo/coco'
# )

# convert_segment_to_detect(
#     label_path='data/dataset/labelme-yolo/coco/labels'
# )

# yolo_to_labelme(
#     yolo_path='data/dataset/labelme-yolo/coco',
#     label_dict={
#         0: 'bus',
#         1: 'car',
#         2: 'motorbike',
#         3: 'truck'
#     }
# )

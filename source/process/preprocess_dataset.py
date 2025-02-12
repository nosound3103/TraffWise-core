import os
import random
import xml.etree.ElementTree as ET
import shutil
import glob
import json
from tqdm import tqdm
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

    for img, lbl in tqdm(zip(train_images, train_labels)):
        shutil.move(os.path.join(images_path, img), train_images_path)
        shutil.move(os.path.join(labels_path, lbl), train_labels_path)

    for img, lbl in tqdm(zip(val_images, val_labels)):
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

    for label in tqdm(train_yolo_labels):
        labelme_file = os.path.join(
            train_labelme_path, os.path.basename(label).replace('.txt', '.json'))
        yolo_to_labelme_single(label, labelme_file, label_dict)

    for label in tqdm(val_yolo_labels):
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
    for line in tqdm(lines):
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


def labelme_to_yolo(labelme_path, yolo_path, label_dict):
    """Convert LabelMe dataset to YOLO dataset

    Args:
        labelme_path (str): path to LabelMe dataset
        yolo_path (str): path to YOLO dataset
        label_dict (dict): label dictionary
    """

    yolo_img_path = os.path.join(yolo_path, 'images')
    yolo_label_path = os.path.join(yolo_path, 'labels')
    os.makedirs(yolo_img_path, exist_ok=True)
    os.makedirs(yolo_label_path, exist_ok=True)

    img_files = glob.glob(os.path.join(labelme_path, '*.jpg'))
    for img_file in tqdm(img_files):
        labelme_file = img_file.replace('.jpg', '.json')
        yolo_label_file = os.path.join(
            yolo_label_path, os.path.basename(labelme_file).replace('.json', '.txt'))
        labelme_to_yolo_single(labelme_file, yolo_label_file, label_dict)


def labelme_to_yolo_single(labelme_file, yolo_label_path, label_dict):
    """Convert each LabelMe annotation file to YOLO annotation file

    Args:
        labelme_file (str): labelme file path
        yolo_label_path (str): yolo label file path
        label_dict (dict): label dictionary
    """

    with open(labelme_file, 'r') as f:
        labelme_content = json.load(f)

    img_path = labelme_file.replace('.json', '.jpg')
    img_width, img_height = labelme_content['imageWidth'], labelme_content['imageHeight']

    yolo_img_path = os.path.join(os.path.dirname(os.path.dirname(
        yolo_label_path)), "images", os.path.basename(img_path))

    shutil.copy(img_path, yolo_img_path)

    annotations = labelme_content['shapes']
    yolo_annotations = []
    for annotation in tqdm(annotations):
        label = annotation['label']
        points = annotation['points']

        if annotation['shape_type'] == 'polygon':
            x_points = [x / img_width for x, _ in points]
            y_points = [y / img_height for _, y in points]

            xmin = min(x_points)
            ymin = min(y_points)
            xmax = max(x_points)
            ymax = max(y_points)

        elif annotation['shape_type'] == 'rectangle':
            xmin, ymin = points[0]
            xmax, ymax = points[1]

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height

        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        label_idx = int(label_dict[label])

        yolo_annotations.append(
            f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    with open(yolo_label_path, 'w') as f:
        f.write("\n".join(yolo_annotations))


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

    for obj in tqdm(root.findall('object')):
        label = obj.find('name').text
        if label not in label_dict:
            return False

        label_idx = int(label_dict[label])
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

    for image_set, ids in tqdm([('train', train_ids), ('val', val_ids)]):
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

    for img, lbl in tqdm(zip(train_images, train_labels)):
        shutil.copy(os.path.join(image_path, img),
                    os.path.join(yolo_path, 'images/train', img))
        shutil.copy(os.path.join(label_path, lbl),
                    os.path.join(yolo_path, 'labels/train', lbl))

    for img, lbl in tqdm(zip(test_images, test_labels)):
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

    for img, lbl in tqdm(zip(train_images, train_labels)):
        shutil.copy(os.path.join(image_path, img),
                    os.path.join(yolo_path, 'images/train', img))
        shutil.copy(os.path.join(label_path, lbl),
                    os.path.join(yolo_path, 'labels/train', lbl))

    for img, lbl in tqdm(zip(test_images, test_labels)):
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


def process_daynight_dataset(daynight_path, yolo_path):
    """Process daynight dataset

    Args:
        daynight_path (str): path to daynight dataset
        yolo_path (str): path to YOLO dataset
    """

    os.makedirs(os.path.join(yolo_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'labels'), exist_ok=True)

    img_files = glob.glob(os.path.join(daynight_path, '*', '*', '*.jpg'))

    for img_file in tqdm(img_files):
        label_file = img_file.replace('.jpg', '.txt')
        shutil.copy(img_file, os.path.join(
            yolo_path, 'images', os.path.basename(img_file)))
        shutil.copy(label_file, os.path.join(
            yolo_path, 'labels', os.path.basename(label_file)))


def process_v9i_dataset(v9i_path, yolo_path):
    """Process V9I dataset

    Args:
        v9i_path (str): path to V9I dataset
        yolo_path (str): path to YOLO dataset
    """

    os.makedirs(os.path.join(yolo_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, 'labels'), exist_ok=True)

    img_files = glob.glob(os.path.join(v9i_path, 'train', '*', '*.jpg')) + \
        glob.glob(os.path.join(v9i_path, 'val', '*', '*.jpg'))

    for img_file in tqdm(img_files):
        label_file = img_file.replace(
            '.jpg', '.txt').replace('images', 'labels')
        shutil.copy(img_file, os.path.join(
            yolo_path, 'images', os.path.basename(img_file)))
        shutil.copy(label_file, os.path.join(
            yolo_path, 'labels', os.path.basename(label_file)))


def merge_label_in_yolo_data(yolo_path, merge_dict):
    """Merge labels in YOLO dataset

    Args:
        yolo_path (str): path to YOLO dataset
        label_dict (str): label dictionary
    """
    label_files = glob.glob(os.path.join(yolo_path, 'labels', '*.txt'))
    for label_file in tqdm(label_files):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()

            data = list(map(float, line.split()))

            if len(data) == 0:
                continue

            label_idx, x_center, y_center, width, height = data
            label = merge_dict[int(label_idx)]
            new_lines.append(
                f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(label_file, 'w') as f:
            f.write("\n".join(new_lines))


def merge_datasets(list_folders, output_folder, amounts=[]):
    """Merge multiple datasets into one

    Args:
        list_folders (list): list of folders to merge
        output_folder (str): output folder
    """
    if len(amounts) == 0:
        amounts = [1] * len(list_folders)

    subfolders = ['images/train', 'images/val', 'labels/train', 'labels/val']

    for subfolder in subfolders:
        dest = os.path.join(output_folder, subfolder)
        os.makedirs(dest, exist_ok=True)

        for idx, source in enumerate(list_folders):
            src = os.path.join(source, subfolder)

            if os.path.exists(src):
                list_files = os.listdir(src)
                amount = int(amounts[idx] * len(list_files))
                random.seed(42)
                random.shuffle(list_files)
                list_files = list_files[:amount]

                for file_name in list_files:
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
    for label_file in tqdm(label_files):
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


def clear_non_label_in_yolo(yolo_path):
    """Clear non label files in YOLO dataset

    Args:
        yolo_path (str): path to YOLO dataset
    """
    count = 0
    label_files = glob.glob(os.path.join(yolo_path, 'labels', '*', '*.txt'))
    for label_file in tqdm(label_files):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            os.remove(label_file)
            img_file = label_file.replace(
                'labels', 'images').replace('.txt', '.jpg')
            os.remove(img_file)
            count += 1

    print(f"Removed {count} non-label files. {yolo_path}")


# Pascal VOC
# convert_pascal_voc_to_yolo(
#     pascal_voc_path='data/dataset/raw/Pascal VOC 2012/VOC2012_train_val',
#     yolo_path='data/dataset/labelme-yolo/pascal_voc')
# clear_non_label_in_yolo('data/dataset/labelme-yolo/pascal_voc')

# Vehicle detection 8 classes
# convert_8_classes_dataset_to_yolo(
#     data_path='data/dataset/raw/Vehicle detection 8 classes/train',
#     yolo_path='data/dataset/labelme-yolo/8-classes')


# BY9XS
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
# clear_non_label_in_yolo('data/dataset/labelme-yolo/by9xs')

# Coco
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

# Xe ba gác
# labelme_to_yolo(
#     labelme_path='data/dataset/raw/xe_ba_gac',
#     yolo_path='data/dataset/labelme-yolo/xe_ba_gac',
#     label_dict={
#         'bus': 0,
#         'car': 1,
#         'motorbike': 2,
#         'truck': 3
#     }
# )
# split_yolo_data(
#     data_path='data/dataset/labelme-yolo/xe_ba_gac'
# )
# clear_non_label_in_yolo('data/dataset/labelme-yolo/xe_ba_gac')

# daynight
# process_daynight_dataset(
#     daynight_path='data/dataset/raw/Vietnamese Vehicles Dataset',
#     yolo_path='data/dataset/labelme-yolo/daynight'
# )
# merge_label_in_yolo_data(
#     yolo_path='data/dataset/labelme-yolo/daynight',
#     merge_dict={
#         0: 2,
#         1: 1,
#         2: 0,
#         3: 3,
#         4: 2,
#         5: 1,
#         6: 0,
#         7: 3
#     }
# )
# split_yolo_data(
#     data_path='data/dataset/labelme-yolo/daynight'
# )
# yolo_to_labelme(
#     yolo_path='data/dataset/labelme-yolo/daynight',
#     label_dict={
#         0: 'bus',
#         1: 'car',
#         2: 'motorbike',
#         3: 'truck'
#     }
# )
# clear_non_label_in_yolo('data/dataset/labelme-yolo/daynight')

# V9I
# process_v9i_dataset(
#     v9i_path='data/dataset/raw/vehicle detection.v9i',
#     yolo_path='data/dataset/labelme-yolo/v9i'
# )
# split_yolo_data(
#     data_path='data/dataset/labelme-yolo/v9i'
# )
# yolo_to_labelme(
#     yolo_path='data/dataset/labelme-yolo/v9i',
#     label_dict={
#         0: 'bus',
#         1: 'car',
#         2: 'motorbike',
#         3: 'truck'
#     }
# )
# clear_non_label_in_yolo('data/dataset/labelme-yolo/v9i')

random.seed(42)

# merge_datasets(
#     list_folders=[
#         'data/dataset/labelme-yolo/by9xs',
#         'data/dataset/labelme-yolo/xe_ba_gac',
#         'data/dataset/labelme-yolo/daynight',
#         'data/dataset/labelme-yolo/v9i',
#     ],
#     output_folder='data/dataset/combination'
# )

# merge_datasets(
#     list_folders=[
#         'data/dataset/labelme-yolo/by9xs',
#         'data/dataset/labelme-yolo/xe_ba_gac',
#         'data/dataset/labelme-yolo/daynight',
#         'data/dataset/labelme-yolo/v9i',
#     ],
#     output_folder='data/dataset/combination-lite',
#     amounts=[0.33, 1, 0.33, 0.33]
# )

merge_datasets(
    list_folders=[
        'data/dataset/labelme-yolo/xe_ba_gac',
        'data/dataset/labelme-yolo/coco',
        'data/dataset/labelme-yolo/pascal_voc',
    ],
    output_folder='data/dataset/coco-v2',
    amounts=[1, 1, 1]
)

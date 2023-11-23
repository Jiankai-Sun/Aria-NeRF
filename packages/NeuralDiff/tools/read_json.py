import json
import os
import shutil
import cv2

def read_json(input_path='../data/EPIC-Diff/P01_01/meta.json'):
    with open(input_path, 'r') as f:
        data = json.load(f)
    print('data.keys(): ', data.keys())
    # data.keys():  dict_keys(['ids_all', 'ids_train', 'ids_test', 'ids_val', 'poses', 'nears', 'fars', 'images', 'intrinsics'])
    print(len(data['ids_all']))  # 860
    print(len(data['ids_train']))  # 752
    print(len(data['ids_test']))  # 54
    print(len(data['ids_val']))  # 54
    print(len(data['poses']), data['poses']['993'])  # 860
    print(len(data['nears']), data['nears']['993'])  # 860
    print(len(data['fars']), data['fars']['993'])  # 860
    print(len(data['images']), data['images']['993'])  # 860
    print(len(data['intrinsics']), data['intrinsics'])  # 3 [[127.37926483154297, 0.0, 114.0], [0.0, 127.37926483154297, 64.0], [0.0, 0.0, 1.0]]

def generate_json(input_path='meta.json'):
    ids_all = []
    ids_train = []
    ids_test = []
    ids_val = []
    poses = {}
    nears = {}
    fars = {}
    images = {}
    intrinsics = []
    data = {'ids_all': ids_all, 'ids_train': ids_train, 'ids_test': ids_test, 'ids_val': ids_val, 'poses': poses,
            'nears': nears, 'fars': fars, 'images': images, 'intrinsics': intrinsics}

    with open(input_path, 'w') as f:
        json.dump(data, f)

def convert_to_epicdiff(input_path='../data/EPIC-Diff/P01_01/meta.json', output_path=''):
    os.makedirs(output_path, exist_ok=True)
    input_img_dir = os.path.join(input_path, 'images')
    img_name_list = sorted(os.listdir(input_img_dir))
    input_transforms_path = os.path.join(input_path, 'transforms.json')
    output_frame_dir = os.path.join(output_path, 'P01_01', 'frames')
    output_annotations_dir = os.path.join(output_path, 'P01_01', 'annotations')
    os.makedirs(output_frame_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)
    output_meta_dir = os.path.join(output_path, 'P01_01', 'meta.json')
    ids_all = []
    ids_train = []
    ids_test = []
    ids_val = []
    poses = {}
    nears = {}
    fars = {}
    images = {}
    intrinsics = [[127.37926483154297, 0.0, 114.0], [0.0, 127.37926483154297, 64.0], [0.0, 0.0, 1.0]]
    meta_dict = {'ids_all': ids_all, 'ids_train': ids_train, 'ids_test': ids_test, 'ids_val': ids_val, 'poses': poses,
            'nears': nears, 'fars': fars, 'images': images, 'intrinsics': intrinsics}
    len_img_name_list = len(img_name_list)
    with open(input_transforms_path, 'r') as f:
        transforms_data = json.load(f)
    input_w = transforms_data['w']
    input_h = transforms_data['h']
    input_fl_x = transforms_data['fl_x']
    input_fl_y = transforms_data['fl_y']
    input_cx = transforms_data['cx']
    input_cy = transforms_data['cy']
    input_k1 = transforms_data['k1']
    input_k2 = transforms_data['k2']
    input_p1 = transforms_data['p1']
    input_camera_model = transforms_data['camera_model']
    input_frames = transforms_data['frames']
    input_applied_transform = transforms_data['applied_transform']

    for idx, each_img_name in enumerate(img_name_list):
        each_img_path = os.path.join(input_img_dir, each_img_name)
        img = cv2.imread(each_img_path)
        img_idx = int(each_img_name[6:])
        output_img_name = 'IMG_{:10d}.bmp'.format(img_idx)
        output_img_path = os.path.join(output_frame_dir, output_img_name)
        cv2.imwrite(output_img_path, img)
        meta_dict['ids_all'].append(img_idx)
        if idx < len_img_name_list * 0.8:
            meta_dict['ids_train'].append(img_idx)
        elif idx < len_img_name_list * 0.9:
            meta_dict['ids_test'].append(img_idx)
        else:
            meta_dict['ids_val'].append(img_idx)
        meta_dict['poses'][str(img_idx)] = 0
        meta_dict['nears'][str(img_idx)] = 0
        meta_dict['fars'][str(img_idx)] = 0
        meta_dict['images'][str(img_idx)] = output_img_name

if __name__ == "__main__":
    read_json(input_path='../data/EPIC-Diff/P01_01/meta.json')
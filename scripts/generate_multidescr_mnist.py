import os
import json
import numpy as np
import idx2numpy
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize as img_resize


def form_full_data_arrays(path_train_images, path_train_labels, path_test_images, path_test_labels):
    train_images = idx2numpy.convert_from_file(path_train_images)
    train_labels = idx2numpy.convert_from_file(path_train_labels)
    test_images = idx2numpy.convert_from_file(path_test_images)
    test_labels = idx2numpy.convert_from_file(path_test_labels)
    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)
    return images, labels


def to_rgb(a):
    return np.repeat(a[np.newaxis, :, :], 3, axis=0)


def change_color(x, new_color):
    existing_colors = ['w', 'r', 'g', 'b']
    if new_color == existing_colors[0]:
        pass
    elif new_color == existing_colors[1]:
        x[1, :, :] = 0
        x[2, :, :] = 0
    elif new_color == existing_colors[2]:
        x[0, :, :] = 0
        x[2, :, :] = 0
    elif new_color == existing_colors[3]:
        x[0, :, :] = 0
        x[1, :, :] = 0
    else:
        raise ValueError('Not existing color.')
    return x


def modify_image(img, new_border_size, new_color):
    x = img.copy()
    x = img_resize(x, (new_border_size, new_border_size))
    x = to_rgb(x)
    x = change_color(x, new_color=new_color)
    return x


def get_horizontal_position_indecies(digit_index, digit_size, border_size):
    inside_size = (128 - border_size * 4) // 3
    start = border_size * (digit_index + 1) + inside_size * digit_index
    start = start + (inside_size - digit_size) // 2
    return start, start + digit_size


def get_vertical_position_indecies(position, digit_size, border_size):
    existing_positions = ['up', 'middle', 'down']
    s, e = get_horizontal_position_indecies(existing_positions.index(position), digit_size, border_size)
    return s, e


def get_randomized_index(n_obs, n_samples):
    index = []
    _index = list(range(n_obs))
    for i in range(int(np.ceil(n_samples / n_obs))):
        np.random.shuffle(_index)
        index += _index
    return index[:n_samples]


def get_sample(index_list, images, labels, border_size_list, color_list, position_list):
    new_img = np.zeros((3, 128, 128))
    new_label = []
    for di in range(3):
        j = index_list[di]
        x = images[j, :, :]
        label = labels[j]
        new_size = np.random.choice(border_size_list)
        new_color = np.random.choice(color_list)
        new_position = np.random.choice(position_list)
        new_label += [label, new_size, new_color, new_position]
        new_x = modify_image(x, new_size, new_color)
        row_start, row_end = get_horizontal_position_indecies(
            digit_index=di,
            digit_size=new_size,
            border_size=2)
        col_start, col_end = get_vertical_position_indecies(
            position=new_position,
            digit_size=new_size,
            border_size=2)
        new_img[:, col_start:col_end, row_start:row_end] = new_x
    return new_img, new_label


def save_separate_image(img, dir_path, file_name):
    full_name = file_name + '.png'
    path = os.path.join(dir_path, full_name)
    plt.imsave(path, np.moveaxis(img, 0, 2))


def save_full_txt_description(desription, dir_path, file_name, obs_name):
    full_name = file_name + '.txt'
    path = os.path.join(dir_path, full_name)
    txt_info = 'Obs {}\n'.format(obs_name)
    for i in range(3):
        txt_info += 'digit_{}\n'.format(i)
        txt_info += '+ value: {}\n'.format(int(desription[4 * i + 0]))
        txt_info += '+ size: {}\n'.format(int(desription[4 * i + 1]))
        txt_info += '+ color: {}\n'.format(str(desription[4 * i + 2]))
        txt_info += '+ position: {}\n'.format(str(desription[4 * i + 3]))
    with open(path, 'a') as outfile:
        outfile.write(txt_info)


def save_separate_json_description(desription, dir_path, file_name):
    full_name = file_name + '.json'
    path = os.path.join(dir_path, full_name)
    desc_dict = {}
    for i in range(3):
        desc_dict['digit_{}'.format(i)] = {
            'value': int(desription[4 * i + 0]),
            'size': int(desription[4 * i + 1]),
            'color': str(desription[4 * i + 2]),
            'position': str(desription[4 * i + 3])
        }
    with open(path, 'w') as outfile:
        json.dump(desc_dict, outfile, indent=4)


def save_label_info_json(dir_path, sizes_list, colors_list, positions_list):
    path = os.path.join(dir_path, 'labels_info.json')
    desc_dict = {}

    desc_dict['value_id_tree'] = {
        'number': dict(zip(range(10), range(10))),
        'size': {},
        'color': {},
        'position': {}
    }
    end_id = 10
    for s in sizes_list:
        desc_dict['value_id_tree']['size'][s] = end_id
        end_id += 1
    for c in colors_list:
        desc_dict['value_id_tree']['color'][c] = end_id
        end_id += 1
    for p in positions_list:
        desc_dict['value_id_tree']['position'][p] = end_id
        end_id += 1

    desc_dict['num_values'] = end_id
    with open(path, 'w') as outfile:
        json.dump(desc_dict, outfile, indent=4)


def create_dataset(n_samples,
                   data_root_save_folder,
                   path_train_images,
                   path_train_labels,
                   path_test_images,
                   path_test_labels,
                   border_size_list=[20, 30, 40],
                   color_list=['w', 'r', 'g', 'b'],
                   position_list=['up', 'middle', 'down'],
                   verbose=True):

    images, labels = form_full_data_arrays(
        path_train_images=path_train_images,
        path_train_labels=path_train_labels,
        path_test_images=path_test_images,
        path_test_labels=path_test_labels
    )

    n_digit_samples = n_samples * 3

    data_img_path = os.path.join(data_root_save_folder, 'images')
    data_desc_path = os.path.join(data_root_save_folder, 'description')
    if not os.path.exists(data_img_path):
        os.makedirs(data_img_path)
    if not os.path.exists(data_desc_path):
        os.makedirs(data_desc_path)

    save_label_info_json(data_desc_path, border_size_list, color_list, position_list)

    index = get_randomized_index(n_obs=images.shape[0], n_samples=n_digit_samples)
    start_digit_i = 0
    for sample_num in range(n_samples):
        new_img, new_label = get_sample(
            index_list=index[start_digit_i:start_digit_i + 3],
            images=images,
            labels=labels,
            border_size_list=border_size_list,
            color_list=color_list,
            position_list=position_list)

        obs_name = str(sample_num) + '_' + ''.join(map(str, new_label))
        save_separate_image(new_img, data_img_path, obs_name)
        save_full_txt_description(new_label, data_desc_path, 'images_description', obs_name)

        if verbose: print('{:<10} {}'.format(sample_num, obs_name))
        start_digit_i += 3


if __name__=="__main__":
    # create_dataset(
    #     n_samples=100_000,
    #     data_root_save_folder='/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/multi_descriptive_MNIST/',
    #     path_train_images='/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/MNIST/MNIST/raw/train-images-idx3-ubyte',
    #     path_train_labels='/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/MNIST/MNIST/raw/train-labels-idx1-ubyte',
    #     path_test_images='/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/MNIST/MNIST/raw/t10k-images-idx3-ubyte',
    #     path_test_labels='/u/82/sukhoba1/unix/Desktop/TA-VQVAE/data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte',
    #     border_size_list=[20, 30, 40],
    #     color_list=['w', 'r', 'g', 'b'],
    #     position_list=['up', 'middle', 'down'],
    #     verbose=True)

    create_dataset(
        n_samples=100,
        data_root_save_folder='/home/andrey/Aalto/thesis/TA-VQVAE/data/multi_descriptive_MNIST/',
        path_train_images='/home/andrey/Aalto/thesis/TA-VQVAE/data/MNIST/MNIST/raw/train-images-idx3-ubyte',
        path_train_labels='/home/andrey/Aalto/thesis/TA-VQVAE/data/MNIST/MNIST/raw/train-labels-idx1-ubyte',
        path_test_images='/home/andrey/Aalto/thesis/TA-VQVAE/data/MNIST/MNIST/raw/t10k-images-idx3-ubyte',
        path_test_labels='/home/andrey/Aalto/thesis/TA-VQVAE/data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte',
        border_size_list=[20, 30, 40],
        color_list=['w', 'r', 'g', 'b'],
        position_list=['up', 'middle', 'down'],
        verbose=True)






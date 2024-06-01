from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

import torchvision
import seaborn as sns
from cjm_pil_utils.core import get_img_files
from matplotlib import pyplot as plt, cm
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.tv_tensors import Mask
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm

torchvision.disable_beta_transforms_warning()


def create_polygon_mask(image_size, vertices):
    mask_img = np.zeros(image_size, dtype=np.uint8)
    if vertices.shape[0] >= 3:
        mask_img = cv2.fillPoly(mask_img, pts=[vertices.astype(int)], color=255)
    return mask_img


def height_to_clr(x, mn, mx):
    norm = matplotlib.colors.Normalize(vmin=mn, vmax=mx, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('cool'))
    return mapper.to_rgba(x)


def load_data(path: Path, debug=True):
    img_file_paths = get_img_files(path / 'images')
    annotation_file_paths = list((path / 'ground_truth_files').glob('*.json'))
    if debug:
        img_file_paths = img_file_paths[:20]
        annotation_file_paths = annotation_file_paths[:20]

    cls_dataframes = (pd.read_json(f, orient='index').transpose() for f in tqdm(annotation_file_paths))

    annotation_df = pd.concat(cls_dataframes, ignore_index=False)

    annotation_df['index'] = annotation_df.apply(lambda row: row['imagePath'].split('.')[0], axis=1)
    annotation_df = annotation_df.set_index('index')
    img_dict = {file.stem: file for file in img_file_paths}
    annotation_df = annotation_df.loc[list(img_dict.keys())]

    shapes_df = annotation_df['shapes'].explode().to_frame().shapes.apply(pd.Series)

    # Stat
    # sns.displot(shapes_df, x='group_id', binwidth=3)
    # plt.show()

    shapes_df['points'] = shapes_df['points'].apply(np.array)
    images = [(path.stem, cv2.imread(str(path))) for path in tqdm(img_file_paths)]
    dims = []
    x_data = []
    y_data = []
    for i, (name, img) in enumerate(tqdm(images)):
        df = shapes_df.loc[name]
        mask_imgs = [create_polygon_mask(img.shape[:2], points) for points in df['points']]
        for j, mask in enumerate(mask_imgs):
            x_data.append((img, mask))
            if type(df['group_id']) == np.int64:
                y_data.append(df['group_id'])
            else:
                y_data.append(df['group_id'].iloc[j])
        if not debug:
            continue
        # masks = torch.concat([Mask(transforms.ToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
        # dims.append(masks.shape[0])
        #
        # mn = min(df['group_id'])
        # mx = max(df['group_id'])
        # colors = [255 * np.array(height_to_clr(h, mn, mx)[:3]) for h in df['group_id']]
        # annotated_tensor = draw_segmentation_masks(
        #     image=transforms.ToTensor()(img),
        #     masks=masks,
        #     alpha=0.8,
        #     colors=colors
        # )
        # im = annotated_tensor.transpose(2, 0).numpy()
        # plt.imshow(im)
        # plt.show()
    # print(f"Max buildings {max(dims)}")
    return x_data, y_data


class HeightEstimationDataset(Dataset):
    def __init__(self, root_dir: Path):
        self._data = load_data(root_dir, debug=True)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img = self.transform(self._data[0][idx][0])
        mask = self.transform(self._data[0][idx][1])
        h = torch.tensor(self._data[1][idx], dtype=torch.float32)
        return (img, mask), h


if __name__ == '__main__':
    def main():
        load_data(Path('data/mlc_training_data'), debug=True)


    main()

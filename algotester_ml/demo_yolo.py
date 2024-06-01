from pathlib import Path

from cjm_pil_utils.core import get_img_files
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torchvision.utils import draw_segmentation_masks
from ultralytics import YOLO


img_file_paths = get_img_files(Path('data/mlc_test_images'))
model = YOLO(r"C:\Users\PC\Development\sc_ml\runs\segment\train24\weights\best.pt")

for path in img_file_paths:
    res = model.predict(path)[0]
    img = res.orig_img
    annotated_tensor = draw_segmentation_masks(
                    image=transforms.ToTensor()(img),
                    masks=(res.masks.data.detach().cpu() > 0.1),
                    alpha=0.8,
                )
    im = annotated_tensor.transpose(2, 0).numpy()
    plt.imshow(im)
    plt.show()

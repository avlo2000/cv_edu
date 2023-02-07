from pathlib import Path

import cv2
import pandas as pd


data_root = '../data/ikea/'
df = pd.read_pickle(data_root + 'text_data/item_to_room.p')
for item_path, rooms in df.items():
    if 'clock' not in item_path:
        continue
    item = cv2.imread(data_root + item_path)
    if item is None:
        continue
    cv2.imshow('item', item)
    for idx, room_path in enumerate(rooms):
        room = cv2.imread(data_root + room_path)
        if room is None:
            continue
        cv2.imshow(f'room_{idx}', room)
    cv2.waitKey()

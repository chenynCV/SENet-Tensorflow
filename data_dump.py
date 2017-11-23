import numpy as np
import json
import os
from IPython import embed

label_path = '/data0/AIChallenger/places_devkit/categories_places365.txt'
data_path = '/data0/AIChallenger/data_256'

result = []
with open(label_path, 'r') as f:
    lines = (line.strip() for line in f)
    for line in lines:
        path, label_id = line.split()
        path = path[1:]
        for filename in os.listdir(os.path.join(data_path, path)):
            image = {}
            image['image_id'] = os.path.join(path, filename)
            image['label_id'] = label_id
            result.append(image)

with open('/data0/AIChallenger/data_256.json', 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result))

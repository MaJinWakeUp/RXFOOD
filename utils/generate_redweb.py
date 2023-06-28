import os

root_dir = "/home/majin/datasets/ReDWeb-S"
split = ['train', 'test']

for s in split:
    rgb_dir = os.path.join(root_dir, s+'set', 'RGB')
    img_names = os.listdir(rgb_dir)
    img_names.sort()
    images = [os.path.join(rgb_dir, name) for name in img_names]
    
    depth_dir = os.path.join(root_dir, s+'set', 'depth')
    depths = [os.path.join(depth_dir, name.replace('.jpg', '.png')) for name in img_names]

    target_dir = os.path.join(root_dir, s+'set', 'GT')
    targets = [os.path.join(target_dir, name.replace('.jpg', '.png')) for name in img_names]
    
    with open(os.path.join(root_dir, s+'.txt'), 'w') as f:
        for i in range(len(images)):
            f.write('{} {} {}\n'.format(images[i], depths[i], targets[i]))
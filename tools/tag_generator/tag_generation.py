import shutil
import os.path as osp
import os

# Tag List
# 'urban', 'highway', 'countryside', 'alleyway', 'parkinglots', 'shoulder', 'mountain', 'university'
# 'day', 'night'
# 'normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow'

### Here to change ###
PATH_TAG = '/home/donghee/Desktop/KRadar/rdr_tags'
dict_tag = {
    '1': ['urban', 'night', 'normal'],
    '2': ['highway', 'night', 'normal'],
    '3': ['highway', 'night', 'normal'],
    '4': ['highway', 'night', 'normal'],
    '5': ['urban', 'day', 'normal'],
    '6': ['urban', 'night', 'normal'],
    '7': ['alleyway', 'night', 'normal'],
    '8': ['university', 'night', 'normal'],
    '9': ['highway', 'day', 'normal'],
    '10': ['highway', 'day', 'normal'],
    '11': ['highway', 'day', 'normal'],
    '12': ['highway', 'day', 'normal'],
    '13': ['highway', 'day', 'overcast'],
    '14': ['urban', 'day', 'normal'],
    '15': ['urban', 'day', 'normal'],
    '16': ['university', 'day', 'normal'],
    '17': ['university', 'day', 'normal'],
    '18': ['urban', 'day', 'normal'],
    '19': ['alleyway', 'day', 'normal'],
    '20': ['urban', 'day', 'normal'],
    '21': ['alleyway', 'night', 'rain'],
    '22': ['urban', 'night', 'overcast'],
    '23': ['urban', 'night', 'rain'],
    '24': ['urban', 'night', 'rain'],
    '25': ['urban', 'night', 'rain'],
    '26': ['countryside', 'day', 'rain'],
    '27': ['countryside', 'day', 'sleet'],
    '28': ['mountain', 'day', 'sleet'],
    '29': ['mountain', 'day', 'sleet'],
    '30': ['parkinglots', 'day', 'sleet'],
    '31': ['countryside', 'day', 'sleet'],
    '32': ['countryside', 'day', 'rain'],
    '33': ['countryside', 'day', 'rain'],
    '34': ['countryside', 'night', 'rain'],
    '35': ['parkinglots', 'night', 'sleet'],
    '36': ['parkinglots', 'night', 'sleet'],
    '37': ['countryside', 'night', 'sleet'],
    '38': ['mountain', 'day', 'fog'],
    '39': ['mountain', 'day', 'fog'],
    '40': ['mountain', 'day', 'fog'],
    '41': ['mountain', 'day', 'fog'],
    '42': ['urban', 'day', 'lightsnow'],
    '43': ['urban', 'day', 'lightsnow'],
    '44': ['shoulder', 'day', 'fog'],
    '45': ['shoulder', 'day', 'fog'],
    '46': ['highway', 'night', 'heavysnow'],
    '47': ['highway', 'night', 'heavysnow'],
    '48': ['highway', 'night', 'lightsnow'],
    '49': ['highway', 'night', 'lightsnow'],
    '50': ['highway', 'night', 'sleet'],
    '51': ['highway', 'night', 'sleet'],
    '52': ['highway', 'night', 'sleet'],
    '53': ['highway', 'day', 'sleet'],
    '54': ['urban', 'day', 'heavysnow'],
    '55': ['urban', 'day', 'heavysnow'],
    '56': ['urban', 'day', 'heavysnow'],
    '57': ['urban', 'day', 'heavysnow'],
    '58': ['urban', 'day', 'heavysnow'],
}
### Here to change ###

def tag_generation(path_tag, dict_tag):
    for k, v in dict_tag.items():
        path_dir = osp.join(path_tag, k)
        os.makedirs(path_dir, exist_ok=True)
        path_txt = osp.join(path_dir, 'description.txt')
        desc = f'{v[0]},{v[1]},{v[2]}'
        f = open(path_txt, 'w')
        f.write(desc)
        f.close()

if __name__=='__main__':
    tag_generation(PATH_TAG, dict_tag)

import os

from toolkit import dataset

import shift_file_name

# os.rename('a.txt', 'b.kml')
if __name__ == '__main__':

    to_shift_up = [2,3,5]
    shift = len(to_shift_up)
    ds = dataset.get_all_classes_dict()
    for su in to_shift_up:
        if not f'{su}' in ds:
            print('error')
            exit(-1)
    shift_file_name.main(shift)
    from_to = []

    counter = 0
    for su in to_shift_up:
        os.rename('data/' + str(su+shift) + '_' + ds[f'{su}'], 'data/' + str(counter) + '_' + ds[f'{su}'])
        counter += 1

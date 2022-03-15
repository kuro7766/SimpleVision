import os

from toolkit import dataset


# os.rename('a.txt', 'b.kml')
def main(shift=0):
    ds = dataset.get_all_classes_dict()
    from_to = []
    for k in ds:
        n = int(k)
        path = f'data/{n}_{ds[k]}'
        check = n + shift
        if check < 0:
            print('error')
            exit(-1)
        to = f'data/{check}_{ds[k]}'
        from_to.append({'f': path, 'to': to})

    for d in from_to:
        os.rename(d['f'], d['to'])


if __name__ == '__main__':
    main(1)

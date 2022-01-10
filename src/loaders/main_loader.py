from src.panotools.house import House
from multiprocessing import Pool


def load_house(name, args):
    house = House(name, args)
    return house

def load_dataset(args, set_name=None):
    names = [line.rstrip() for line in open(args.train_set)]
    test_names = [line.rstrip() for line in open(args.test_set)]
    if set_name is None:
        house_list = names+test_names
    else:
        if set_name == 'train':
            house_list = names
        else:
            house_list = test_names

    # pool = Pool(10)
    # houses = pool.map(load_house, house_list)
    # pool.close()
    houses = [load_house(x, args) for x in house_list]
    return houses

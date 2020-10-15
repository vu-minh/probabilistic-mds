import random
from itertools import chain, islice


def chunks(in_list, size=100, shuffle=True):
    """Generator to chunk `in_list` in to small chunks of size `size`.
    """
    if shuffle:
        in_list = random.sample(list(in_list), k=len(in_list))

    iterator = iter(in_list)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


if __name__ == "__main__":
    a = list(range(10))

    for c in chunks(a, size=3, shuffle=True):
        print(list(c))

    for c in chunks(a, size=3, shuffle=False):
        print(list(c))

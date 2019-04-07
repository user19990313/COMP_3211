import qsort
import random


def gen_arr(size, rangement):
    sol = []
    for i in range(size):
        sol.append(random.randint(0, rangement))
    return sol


def check():
    arr = gen_arr(50, 10)
    qsort.qsort(arr)
    return _checkfun(arr)


def _checkfun(arr):
    for i in range(1, len(arr)):
        if arr[i-1] > arr[i]:
            return False
    return True


if __name__ == "__main__":
    if check():
        print("true")
    else:
        print("false")
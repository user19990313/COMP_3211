from buggy import buggy1
from buggy import buggy2
from buggy import buggy3
from buggy import buggy4
from buggy import buggy5
from buggy import buggy6
import random


def gen_arr(size, rangement):
    sol = []
    for i in range(size):
        sol.append(random.randint(0, rangement))
    return sol


def check1():
    # arr = gen_arr(50, 10)
    # qsort.qsort(arr)
    return _checkfun(buggy1.buggy_sort(gen_arr(50, 10)))


def check2():
    return _checkfun(buggy2.buggy_sort(gen_arr(50, 10)))


def check3():
    return _checkfun(buggy3.buggy_sort(gen_arr(50, 10)))


def check4():
    return _checkfun(buggy4.buggy_sort(gen_arr(50, 10)))


def check5():
    return _checkfun(buggy5.buggy_sort(gen_arr(50, 10)))


def check6():
    return _checkfun(buggy6.buggy_sort(gen_arr(50, 10)))


def _checkfun(arr):
    for i in range(1, len(arr)):
        if arr[i-1] > arr[i]:
            return False
    return True


if __name__ == "__main__":
    if check1():
        print("true")
    else:
        print("false")

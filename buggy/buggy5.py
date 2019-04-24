import random


def bubble_sort (arr):
    while True:
        is_sorted = True
        for x in range(0, len(arr) - 1):
            if arr[x] > arr[x + 1]:
                arr[x], arr[x + 1] = arr[x + 1], arr[x]
                is_sorted = False
        if is_sorted:
            return arr


def partition (arr, left, right):
    pivot = left
    for i in range(left + 1, right + 1):
        if arr[i] <= arr[left]:
            pivot += 1
            arr[i], arr[pivot] = arr[pivot], arr[i]
    arr[pivot], arr[left] = arr[left], arr[pivot]
    return pivot


def _qsort (arr, left, right):
    if left >= right:
        return
    p = partition(arr, left, right)
    _qsort(arr, left, p - 1)
    _qsort(arr, p + 1, right)


def qsort (arr):
    length = len(arr)
    if length <= 1:
        return arr
    _qsort(arr, 0, len(arr) - 1)
    return arr


def selection_sort (arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort (arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 2      # BUG: WRONG STEP
        arr[j + 1] = key
    return arr


def if_sorted (arr):
    for x in range(len(arr) - 1):
        if arr[x] > arr[x + 1]:
            return False
    return True


def get_random_array ():
    arr = []
    for x in range(10):
        arr.append(random.randint(0, 100))
    return arr


def merge(arr1, arr2):
    arr = []
    while len(arr1) * len(arr2) > 0:
        if arr1[0] < arr2[0]:
            arr.append(arr1[0])
            del arr1[0]
        else:
            arr.append(arr2[0])
            del arr2[0]
    if len(arr1) > 0:
        arr = arr + arr1
    elif len(arr2) > 0:
        arr = arr + arr2
    return arr


def _sort(arr, number):
    if number == 1: return qsort(arr)
    elif number == 2: return bubble_sort(arr)
    elif number == 3: return selection_sort(arr)
    elif number == 4: return insertion_sort(arr)


def buggy_sort(arr):
    algorithms = random.sample([1, 2, 3, 4], 2)
    arr1 = _sort(arr[:len(arr) // 2], algorithms[0])
    arr2 = _sort(arr[len(arr) // 2:], algorithms[1])
    return merge(arr1, arr2)


if __name__ == "__main__":
    count = 0
    for x in range(0, 100):
        array = buggy_sort(get_random_array())
        if if_sorted(array):
            count += 1
    print("Success: " + str(count) + " / 100")

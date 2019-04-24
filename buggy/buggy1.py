import random


def bubble_sort (array):
    while True:
        is_sorted = True
        for x in range(1, len(array) - 1):  # BUG: ignore first element
            if array[x] > array[x + 1]:
                array[x], array[x + 1] = array[x + 1], array[x]
                is_sorted = False
        if is_sorted:
            return array


def if_sorted (array):
    for x in range(0, len(array) - 1):
        if array[x] > array[x + 1]:
            return False
    return True


def get_random_array ():
    array = []
    if random.randint(0, 100) < 50:
        array.append(0)
        for x in range(0, 3):
            array.append(random.randint(1, 100))
    else:
        for x in range(0, 4):
            array.append(random.randint(0, 100))
    return array


count = 0
for x in range(0, 100):
    if if_sorted(bubble_sort(get_random_array())):
        count += 1
print("Success: " + str(count) + " / 100")

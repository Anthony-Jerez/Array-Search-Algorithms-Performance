# Array-Search Algorithms

import math
import random
import time
import pandas as pd
from matplotlib import pyplot as plt


# returns a sorted list of random numbers.
def random_list(size):
    arr = [0] * size
    for i in range(1, size):
        arr[i] = arr[i - 1] + random.randint(1, 10)
    return arr


# native_search
def native_search(arr, key, n):
    return arr.index(key)


# linear_search
def linear_search(arr, key, n):
    n = len(arr)
    for i in range(0, n):
        if arr[i] == key:
            return i
    return -1


# binary_search_recursive(arr, key) - see https://www.geeksforgeeks.org/binary-search/
def binary_search_recursive(arr, key, n):
    return binary_search_recursive_helper(arr, key, 0, n - 1)


def binary_search_recursive_helper(arr, key, l, r):
    if r >= l:
        m = l + int((r - l) / 2)
        if arr[m] == key:
            return m
        elif arr[m] > key:
            return binary_search_recursive_helper(arr, key, l, m - 1)
        else:
            return binary_search_recursive_helper(arr, key, m + 1, r)
    else:
        return -1


# binary_search_iterative(arr, key) - see https://www.geeksforgeeks.org/the-ubiquitous-binary-search-set-1/
def binary_search_iterative(arr, key, n):
    l, r = 0, n - 1
    while r - l > 1:
        m = l + int((r - l) / 2)
        if arr[m] <= key:
            l = m
        else:
            r = m
    if arr[l] == key:
        return l
    if arr[r] == key:
        return r
    return -1


# binary_search_randomized(arr, key) - see  https://www.geeksforgeeks.org/randomized-binary-search-algorithm/
def binary_search_randomized(arr, key, n):
    l, r = 0, n - 1
    while l <= r:
        m = random.randint(l, r)
        if arr[m] == key:
            return m
        elif arr[m] < key:
            l = m + 1
        else:
            r = m - 1
    return -1


# ternary_search_recursive(arr, key) - https://www.geeksforgeeks.org/ternary-search/
def ternary_search_recursive_helper(arr, key, l, r):
    if r >= l:
        m1 = l + int((r - l) / 3)
        m2 = r - int((r - l) / 3)
        if arr[m1] == key:
            return m1
        if arr[m2] == key:
            return m2
        if key < arr[m1]:
            return ternary_search_recursive_helper(arr, key, l, m1 - 1)
        elif key > arr[m2]:
            return ternary_search_recursive_helper(arr, key, m2 + 1, r)
        else:
            return ternary_search_recursive_helper(arr, key, m1 + 1, m2 - 1)
    return -1


def ternary_search_recursive(arr, key, n):
    return ternary_search_recursive_helper(arr, key, 0, n - 1)


# ternary_search_iterative(arr, key) - https://www.geeksforgeeks.org/ternary-search/
def ternary_search_iterative(arr, key, n):
    l, r = 0, n - 1
    while r >= l:
        m1 = l + int((r - l) / 3)
        m2 = r - int((r - l) / 3)
        if key == arr[m1]:
            return m1
        if key == arr[m2]:
            return m2
        if key < arr[m1]:
            r = m1 - 1
        elif key > arr[m2]:
            l = m2 + 1
        else:
            l = m1 + 1
            r = m2 - 1
    return -1


# exponential_search(arr, key) - see https://www.geeksforgeeks.org/exponential-search/
def exponential_search(arr, key, n):
    i = 1
    while i < n and arr[i] < key:
        i *= 2
    left = i // 2
    right = min(i, n - 1)
    while left <= right:
        mid = int((left + right) / 2)
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# interpolation_search(arr, key) - see https://www.geeksforgeeks.org/interpolation-search/
def interpolation_search_recursive(arr, key, n):
    return interpolation_search_recursive_helper(arr, key, 0, n - 1)


def interpolation_search_recursive_helper(arr, key, lo, hi):
    if arr[lo] == arr[hi]:
        if key == arr[lo]:
            return lo
        else:
            return -1
    if lo <= hi and arr[lo] <= key <= arr[hi]:
        pos = int(lo + ((hi - lo) / float(arr[hi] - arr[lo])) * float((key - arr[lo])))
        if arr[pos] == key:
            return pos
        elif arr[pos] < key:
            return interpolation_search_recursive_helper(arr, key, pos + 1, hi)
        else:
            return interpolation_search_recursive_helper(arr, key, lo, pos - 1)
    return -1


# jump_search(arr, key) - https://www.geeksforgeeks.org/jump-search/
def jump_search(arr, key, n):
    step = math.sqrt(n)
    prev = 0
    while arr[int(min(step, n) - 1)] < key:
        prev = step
        step += math.sqrt(n)
        if prev >= n:
            return -1
    prev = int(prev)
    while arr[prev] < key:
        prev += 1
        if prev == min(step, n):
            return -1
    if arr[prev] == key:
        return prev
    return -1


# fibonacci_search(arr, key) - https://www.geeksforgeeks.org/fibonacci-search/
def fibonacci_search(arr, key, n):
    fib2 = 0
    fib1 = 1
    fib3 = fib2 + fib1
    while fib3 < n:
        fib2 = fib1
        fib1 = fib3
        fib3 = fib2 + fib1
    offset = -1
    while fib3 > 1:
        i = min(offset + fib2, n - 1)
        if arr[i] < key:
            fib3 = fib1
            fib1 = fib2
            fib2 = fib3 - fib1
            offset = i
        elif arr[i] > key:
            fib3 = fib2
            fib1 = fib1 - fib2
            fib2 = fib3 - fib1
        else:
            return i
    if fib1 and arr[n - 1] == key:
        return n - 1
    return -1


def plot_times(dict_algs, sizes, trials, algs, file_name):
    alg_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for alg in algs:
        alg_num += 1
        d = dict_algs[alg.__name__]
        x_axis = [j + 0.05 * alg_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=0.05, alpha=0.75, label=alg.__name__)
    plt.legend()
    plt.title("Runtime of Algorithms")
    plt.xlabel("Size of elements")
    plt.ylabel(f"Time for {trials} trials (ms)")
    plt.savefig(file_name)
    plt.show()


# print the timings in a table
def print_times(dict_algs):
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(dict_algs).T
    print(df)


# used for running the searches:
def run_algs(algs, sizes, trials):
    dict_algs = {}
    for alg in algs:
        dict_algs[alg.__name__] = {}
    for size in sizes:
        for alg in algs:
            dict_algs[alg.__name__][size] = 0
        for trial in range(1, trials + 1):
            arr = random_list(size)
            idx = random.randint(0, size - 1)
            key = arr[idx]
            for alg in algs:
                start_time = time.time()
                idx_found = alg(arr, key, size)
                end_time = time.time()
                if idx_found != idx:
                    print(alg.__name__, "wrong index found", arr, idx, idx_found)
                net_time = end_time - start_time
                dict_algs[alg.__name__][size] += 1000 * net_time
    return dict_algs


def main():
    sizes = [10, 100, 1000, 10000]
    searches = [native_search, linear_search, binary_search_recursive, binary_search_iterative,
                binary_search_randomized, ternary_search_recursive, ternary_search_iterative, exponential_search,
                jump_search, fibonacci_search, interpolation_search_recursive]
    trials = 1000
    dict_searches = run_algs(searches, sizes, trials)
    print_times(dict_searches)
    plot_times(dict_searches, sizes, trials, searches, "array-searches.png")


if __name__ == "__main__":
    main()

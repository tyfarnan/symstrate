def find_min_index(arr, start, end):
    min_idx = start
    i = start + 1
    while i <= end:
        if arr[i] < arr[min_idx]:
            min_idx = i
        i = i + 1
    return min_idx

def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1 
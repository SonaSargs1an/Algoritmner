Heap sort
# Բրգային տեսակավորում ռեկուրսիայով (Heap Sort)
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # Կառուցում ենք բուրգը
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Տեղափոխում ենք բուրգի գագաթը վերջում
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


arr = [5, 3, 8, 4, 2]
heap_sort(arr)
print("Heap Sort:", arr)
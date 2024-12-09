1)Naive string-matching
def naive_string_matching(text, pattern):
    n = len(text)  
    m = len(pattern)  
    for i in range(n - m + 1):  
        if text[i:i + m] == pattern:  
            print(f'գտնվեց {i} ինդեքսում')
text = "catdogcatmousecat"
pattern = "cat"
naive_string_matching(text, pattern)

2)Finite-automata
def finite_automata_search(text, pattern):
    def build_transition_table(pattern):
        m = len(pattern)
        alphabet = set(pattern) 
        table = [{} for _ in range(m + 1)]  
        for state in range(m + 1):
            for char in alphabet:
                next_state = 0
                if state < m and char == pattern[state]:
                    next_state = state + 1
                else:
                    for k in range(state, 0, -1):
                        if pattern[:k] == pattern[state-k+1:state] + char:
                            next_state = k
                            break
                table[state][char] = next_state
        return table

    m, n = len(pattern), len(text)
    table = build_transition_table(pattern)  
    state = 0
    for i in range(n):  
        state = table[state].get(text[i], 0)
        if state == m:  
            print(f" գտնվեց {i - m + 1} ինդեքսում")

text = "abcdefgababcababcab"
pattern = "abc"
finite_automata_search(text, pattern)


3)Knuth-Morris-Pratt
def kmp_search(text, pattern):
    def build_lps(pattern):
        m = len(pattern)
        lps = [0] * m
        length = 0  
        i = 1
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    n, m = len(text), len(pattern)
    lps = build_lps(pattern)
    i = 0  
    j = 0  
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            print(f"գտնվեց {i - j} ինդեքսում")
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
text = "abaragbcaragbabcaaragbabarab"
pattern = "arag"
kmp_search(text, pattern)

4)Boyer-Moore
def bad_char_heuristic(pattern):
    m = len(pattern)
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i
    return bad_char

def boyer_moore_search(text, pattern):
    n, m = len(text), len(pattern)
    bad_char = bad_char_heuristic(pattern)
    s = 0
    while(s <= n - m):
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            print(f"Pattern found at index {s}")
            s += (m - bad_char.get(text[s + m], -1)) if s + m < n else 1
        else:
            s += max(1, j - bad_char.get(text[s + j], -1))
text = "asdfefghklmjiefghporstefgh"
pattern = "efgh"
boyer_moore_search(text, pattern)

5)Palindrome
def is_palindrome(text):
    text = ''.join(e for e in text if e.isalnum()).lower() 
    return text == text[::-1]  
text = "Was it a car or a cat I saw?"
if is_palindrome(text):
    print("Տեքստը պալինդրոմ է")
else:
    print("Տեքստը պալինդրոմ չէ")

6)Longest Palindromic Substring

def longest_palindromic_substring(s):
    if len(s) < 1:
        return ""
    
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    longest = ""
    for i in range(len(s)):
        odd_palindrome = expand_around_center(i, i)
        even_palindrome = expand_around_center(i, i + 1)
        longest = max(longest, odd_palindrome, even_palindrome, key=len)
    
    return longest
text = "anana"
print(longest_palindromic_substring(text))  

7)Find the string can be Palindromic
def can_form_palindrome(s):
    freq = {}
    for char in s:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1
    odd_count = 0
    for count in freq.values():
        if count % 2 != 0:
            odd_count += 1
    return odd_count <= 1

print(can_form_palindrome("civic"))  
print(can_form_palindrome("ivicc"))  
print(can_form_palindrome("hello"))  

 8)Linear Search 
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i  
    return -1  
arr = [5, 11, 20, 9, 33]
target = 9
print(linear_search(arr, target))  

9)Binary Search
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2  
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
            
    return -1  
arr = [20,33,55,12,85,99,10]
target = 99
print(binary_search(arr, target))

10)Ternary  Search
def ternary_search(arr, target, low, high):
   if high >= low:
        mid1 = low + (high - low) // 3
        mid2 = high - (high - low) // 3
        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2
        if target < arr[mid1]:
            return ternary_search(arr, target, low, mid1 - 1)
        elif target > arr[mid2]:
            return ternary_search(arr, target, mid2 + 1, high)
        else:
            return ternary_search(arr, target, mid1 + 1, mid2 - 1)
    
    return -1  
arr = [2,4,6,8,10,12,14,16,18,20]
target = 10
print(ternary_search(arr, target, 0, len(arr)-1))  

11)Jump Search
import math
def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))  
    prev = 0
    while arr[min(step, n) - 1] < target: 
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:  
            return -1
    for idx in range(prev, min(step, n)):
        if arr[idx] == target:
            return idx 
    
    return -1 
arr = [100,200,300,400,500,600,700,800,900,1000]
target = 500
print(jump_search(arr, target)) 

12)Interpolation Search
def interpolation_search(arr, target):
    low = 0
    high = len(arr) - 1
    
    while low <= high and target >= arr[low] and target <= arr[high]:
        pos = low + int(((target - arr[low]) * (high - low)) / (arr[high] - arr[low]))
        if arr[pos] == target:
            return pos
        if arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    
    return -1  
arr = [11,22,33,44,55,66,77,88,99,]
target = 77
print(interpolation_search(arr, target))  


13)Exponential Search

def binary_search(arr, target, low, high):
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def exponential_search(arr, target):
    if arr[0] == target:
        return 0
    index = 1
    while index < len(arr) and arr[index] <= target:
      index *= 2  
    return binary_search(arr, target, index // 2, min(index, len(arr) - 1))
arr = [3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]
target = 30
print(exponential_search(arr, target))  



14)Convex Hull
def orientation(p, q, r):
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
def graham_scan(points):
    points.sort() 
    hull = []
    for point in points:
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull
points = [(1, 2), (3, 1), (5, 4), (7, 6), (6, 8), (8, 7), (2, 3), (4, 5), (0, 0)]
result = graham_scan(points)
print("Convex Hull:", result)



15)Graham Scam
def graham_scan(points):
    points.sort()  
    hull = []
    for p in points:
        while len(hull) >= 2 and (hull[-2][0] - p[0]) * (hull[-1][1] - p[1]) - (hull[-2][1] - p[1]) * (hull[-1][0] - p[0]) <= 0:
            hull.pop()  
        hull.append(p)  
    return hull  
points = [(2, 4), (1, 1), (4, 2), (6, 5), (7, 7), (3, 3), (5, 6), (0, 0)]
print("Convex Hull:", graham_scan(points))


16)Line Intersection
def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None
    
    if m1 == m2:  
        return None
    if m1 is not None and m2 is not None:
        c1 = y1 - m1 * x1
        c2 = y3 - m2 * x3
        x_intersection = (c2 - c1) / (m1 - m2)
        y_intersection = m1 * x_intersection + c1
        return (x_intersection, y_intersection)
    return None 
x1, y1 = 2, 2
x2, y2 = 5, 5
x3, y3 = 2, 5
x4, y4 = 5, 2

intersection = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
print("Intersection:", intersection)








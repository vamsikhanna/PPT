#!/usr/bin/env python
# coding: utf-8

# # <aside>
# 💡 **Question 1**
# 
# A permutation perm of n + 1 integers of all the integers in the range [0, n] can be represented as a string s of length n where:
# 
# - s[i] == 'I' if perm[i] < perm[i + 1], and
# - s[i] == 'D' if perm[i] > perm[i + 1].
# 
# Given a string s, reconstruct the permutation perm and return it. If there are multiple valid permutations perm, return **any of them**.
# 
# **Example 1:**
# 
# **Input:** s = "IDID"
# 
# **Output:**
# 
# [0,4,1,3,2]
# 
# </aside>

# In[1]:


def reconstruct_permutation(s):
    n = len(s)
    perm = [0] * (n + 1)

    # Initialize the first element to 0
    perm[0] = 0

    # Reconstruct the permutation based on the characters in the string 's'
    for i in range(n):
        if s[i] == 'I':
            perm[i + 1] = perm[i] + 1
        else:
            # If the character is 'D', find the maximum value assigned so far
            # and decrement it to get a decreasing value
            max_val = max(perm[:i + 1])
            perm[i + 1] = max_val - 1

    return perm

# Test case
s = "IDID"
result = reconstruct_permutation(s)
print(result)


# # <aside>
# 💡 **Question 2**
# 
# You are given an m x n integer matrix matrix with the following two properties:
# 
# - Each row is sorted in non-decreasing order.
# - The first integer of each row is greater than the last integer of the previous row.
# 
# Given an integer target, return true *if* target *is in* matrix *or* false *otherwise*.
# 
# You must write a solution in O(log(m * n)) time complexity.
# 
# **Input:** matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
# 
# **Output:** true
# 
# </aside>

# In[2]:


def search_matrix(matrix, target):
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1

    while left <= right:
        mid = left + (right - left) // 2
        mid_val = matrix[mid // n][mid % n]

        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

# Test case
matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]
target = 3
result = search_matrix(matrix, target)
print(result)


# In[ ]:


<aside>
💡 **Question 3**

Given an array of integers arr, return *true if and only if it is a valid mountain array*.

Recall that arr is a mountain array if and only if:

- arr.length >= 3
- There exists some i with 0 < i < arr.length - 1 such that:
    - arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
    - arr[i] > arr[i + 1] > ... > arr[arr.length - 1]

**Example 1:**

**Input:** arr = [2,1]

**Output:**

false


# In[3]:


def valid_mountain_array(arr):
    n = len(arr)
    if n < 3:
        return False

    i = 0

    # Find the increasing part of the mountain
    while i < n - 1 and arr[i] < arr[i + 1]:
        i += 1

    # If there is no increasing part or the peak is the first or last element,
    # then the array is not a valid mountain array
    if i == 0 or i == n - 1:
        return False

    # Find the decreasing part of the mountain
    while i < n - 1 and arr[i] > arr[i + 1]:
        i += 1

    # If we reached the end of the array, it's a valid mountain array
    return i == n - 1

# Test case
arr = [2, 1]
result = valid_mountain_array(arr)
print(result)


# # <aside>
# 💡 **Question 4**
# 
# Given a binary array nums, return *the maximum length of a contiguous subarray with an equal number of* 0 *and* 1.
# 
# **Example 1:**
# 
# **Input:** nums = [0,1]
# 
# **Output:** 2
# 
# **Explanation:**
# 
# [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.
# 
# </aside>

# In[4]:


def find_max_length_contiguous_subarray(nums):
    max_length = 0
    count = 0
    count_map = {0: -1}  # Initialize with count 0 at index -1

    for i, num in enumerate(nums):
        count += 1 if num == 1 else -1

        if count in count_map:
            max_length = max(max_length, i - count_map[count])
        else:
            count_map[count] = i

    return max_length

# Test case
nums = [0, 1]
result = find_max_length_contiguous_subarray(nums)
print(result)


# # <aside>
# 💡 **Question 5**
# 
# The **product sum** of two equal-length arrays a and b is equal to the sum of a[i] * b[i] for all 0 <= i < a.length (**0-indexed**).
# 
# - For example, if a = [1,2,3,4] and b = [5,2,3,1], the **product sum** would be 1*5 + 2*2 + 3*3 + 4*1 = 22.
# 
# Given two arrays nums1 and nums2 of length n, return *the **minimum product sum** if you are allowed to **rearrange** the **order** of the elements in* nums1.
# 
# **Example 1:**
# 
# **Input:** nums1 = [5,3,4,2], nums2 = [4,2,2,5]
# 
# **Output:** 40
# 
# **Explanation:**
# 
# We can rearrange nums1 to become [3,5,4,2]. The product sum of [3,5,4,2] and [4,2,2,5] is 3*4 + 5*2 + 4*2 + 2*5 = 40.
# 
# </aside>

# In[5]:


def minimum_product_sum(nums1, nums2):
    nums1.sort()  # Sort nums1 in ascending order
    nums2.sort(reverse=True)  # Sort nums2 in descending order

    return sum(nums1[i] * nums2[i] for i in range(len(nums1)))

# Test case
nums1 = [5, 3, 4, 2]
nums2 = [4, 2, 2, 5]
result = minimum_product_sum(nums1, nums2)
print(result)


# # <aside>
# 💡 **Question 6**
# 
# An integer array original is transformed into a **doubled** array changed by appending **twice the value** of every element in original, and then randomly **shuffling** the resulting array.
# 
# Given an array changed, return original *if* changed *is a **doubled** array. If* changed *is not a **doubled** array, return an empty array. The elements in* original *may be returned in **any** order*.
# 
# **Example 1:**
# 
# **Input:** changed = [1,3,4,2,6,8]
# 
# **Output:** [1,3,4]
# 
# **Explanation:** One possible original array could be [1,3,4]:
# 
# - Twice the value of 1 is 1 * 2 = 2.
# - Twice the value of 3 is 3 * 2 = 6.
# - Twice the value of 4 is 4 * 2 = 8.
# 
# Other original arrays could be [4,3,1] or [3,1,4]
# 
# </aside>

# In[7]:


def find_original(changed):
    count_map = {}
    
    # Count the occurrences of each element in the 'changed' array
    for num in changed:
        count_map[num] = count_map.get(num, 0) + 1

    original = []
    
    # Reconstruct the 'original' array based on the count
    for num in count_map:
        # If there is a number 'num' and its count is 2, it means that 'num' is a candidate
        # for the original array. We add 'num' to the original array once, as it would
        # have appeared twice in the changed array.
        if count_map[num] == 2:
            original.append(num)

    # Check if the 'original' array is valid by comparing its size with half of the changed array
    if len(original) == len(changed) // 2:
        return original
    else:
        return []

# Test case
changed = [1, 3, 4, 2, 6, 8]
result = find_original(changed)
print(result)


# In[ ]:


<aside>
💡 **Question 7**

Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.

**Example 1**

**Input:** n = 3

**Output:** [[1,2,3],[8,9,4],[7,6,5]]


# In[8]:


def generate_spiral_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    direction_index = 0
    num = 1
    row, col = 0, 0

    for _ in range(n * n):
        matrix[row][col] = num
        num += 1

        # Calculate the next position in the current direction
        next_row = row + directions[direction_index][0]
        next_col = col + directions[direction_index][1]

        # If the next position is out of bounds or already visited, change direction
        if (
            next_row < 0
            or next_row >= n
            or next_col < 0
            or next_col >= n
            or matrix[next_row][next_col] != 0
        ):
            direction_index = (direction_index + 1) % 4

        # Update row and column to the next position in the current direction
        row += directions[direction_index][0]
        col += directions[direction_index][1]

    return matrix

# Test case
n = 3
result = generate_spiral_matrix(n)
for row in result:
    print(row)


# # Question 8**
# 
# Given two [sparse matrices](https://en.wikipedia.org/wiki/Sparse_matrix) mat1 of size m x k and mat2 of size k x n, return the result of mat1 x mat2. You may assume that multiplication is always possible.
# 
# **Example 1:**
# 
# **Input:** mat1 = [[1,0,0],[-1,0,3]], mat2 = [[7,0,0],[0,0,0],[0,0,1]]
# 
# **Output:**
# 
# [[7,0,0],[-7,0,3]]

# In[9]:


def multiply_sparse_matrices(mat1, mat2):
    m, k, n = len(mat1), len(mat1[0]), len(mat2[0])
    result = [[0] * n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            # Calculate the dot product of the i-th row of mat1 and the j-th column of mat2
            for p in range(k):
                result[i][j] += mat1[i][p] * mat2[p][j]

    return result

# Test case
mat1 = [[1, 0, 0], [-1, 0, 3]]
mat2 = [[7, 0, 0], [0, 0, 0], [0, 0, 1]]
result = multiply_sparse_matrices(mat1, mat2)
for row in result:
    print(row)


# In[ ]:





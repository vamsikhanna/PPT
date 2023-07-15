#!/usr/bin/env python
# coding: utf-8

# # <aside>
# 💡 **Question 1**
# Given three integer arrays arr1, arr2 and arr3 **sorted** in **strictly increasing** order, return a sorted array of **only** the integers that appeared in **all** three arrays.
# 
# **Example 1:**
# 
# Input: arr1 = [1,2,3,4,5], arr2 = [1,2,5,7,9], arr3 = [1,3,4,5,8]
# 
# Output: [1,5]
# 
# **Explanation:** Only 1 and 5 appeared in the three arrays.
# 
# </aside>

# In[1]:


def find_common_elements(arr1, arr2, arr3):
    result = []
    i, j, k = 0, 0, 0

    while i < len(arr1) and j < len(arr2) and k < len(arr3):
        if arr1[i] == arr2[j] == arr3[k]:
            result.append(arr1[i])
            i += 1
            j += 1
            k += 1
        elif arr1[i] <= arr2[j] and arr1[i] <= arr3[k]:
            i += 1
        elif arr2[j] <= arr1[i] and arr2[j] <= arr3[k]:
            j += 1
        else:
            k += 1

    return result

# Example usage:
arr1 = [1, 2, 3, 4, 5]
arr2 = [1, 2, 5, 7, 9]
arr3 = [1, 3, 4, 5, 8]
result = find_common_elements(arr1, arr2, arr3)
print(result)  # Output: [1, 5]


# # <aside>
# 💡 **Question 2**
# 
# Given two **0-indexed** integer arrays nums1 and nums2, return *a list* answer *of size* 2 *where:*
# 
# - answer[0] *is a list of all **distinct** integers in* nums1 *which are **not** present in* nums2*.*
# - answer[1] *is a list of all **distinct** integers in* nums2 *which are **not** present in* nums1.
# 
# **Note** that the integers in the lists may be returned in **any** order.
# 
# **Example 1:**
# 
# **Input:** nums1 = [1,2,3], nums2 = [2,4,6]
# 
# **Output:** [[1,3],[4,6]]
# 
# **Explanation:**
# 
# For nums1, nums1[1] = 2 is present at index 0 of nums2, whereas nums1[0] = 1 and nums1[2] = 3 are not present in nums2. Therefore, answer[0] = [1,3].
# 
# For nums2, nums2[0] = 2 is present at index 1 of nums1, whereas nums2[1] = 4 and nums2[2] = 6 are not present in nums2. Therefore, answer[1] = [4,6].
# 
# </aside>

# In[2]:


def find_missing_elements(nums1, nums2):
    set_nums1 = set(nums1)
    set_nums2 = set(nums2)

    result_nums1 = list(set_nums1.difference(set_nums2))
    result_nums2 = list(set_nums2.difference(set_nums1))

    return [result_nums1, result_nums2]

# Example usage:
nums1 = [1, 2, 3]
nums2 = [2, 4, 6]
result = find_missing_elements(nums1, nums2)
print(result)  # Output: [[1, 3], [4, 6]]


# # <aside>
# 💡 **Question 3**
# Given a 2D integer array matrix, return *the **transpose** of* matrix.
# 
# The **transpose** of a matrix is the matrix flipped over its main diagonal, switching the matrix's row and column indices.
# 
# **Example 1:**
# 
# Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
# 
# Output: [[1,4,7],[2,5,8],[3,6,9]]
# 
# </aside>

# In[3]:


def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    # Initialize the transposed matrix with all elements as 0
    transposed_matrix = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            # Switch the row and column indices to create the transpose
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

# Example usage:
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = transpose(matrix)
print(result)  # Output: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]


# # <aside>
# 💡 **Question 4**
# Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2), ..., (an, bn) such that the sum of min(ai, bi) for all i is **maximized**. Return *the maximized sum*.
# 
# **Example 1:**
# 
# Input: nums = [1,4,3,2]
# 
# Output: 4
# 
# **Explanation:** All possible pairings (ignoring the ordering of elements) are:
# 
# 1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
# 
# 2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
# 
# 3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
# 
# So the maximum possible sum is 4.
# 
# </aside>

# In[4]:


def array_pair_sum(nums):
    nums.sort()  # Sort the array in ascending order
    max_sum = 0

    for i in range(0, len(nums), 2):
        max_sum += nums[i]

    return max_sum

# Example usage:
nums = [1, 4, 3, 2]
result = array_pair_sum(nums)
print(result)  # Output: 4


# # <aside>
# 💡 **Question 5**
# You have n coins and you want to build a staircase with these coins. The staircase consists of k rows where the ith row has exactly i coins. The last row of the staircase **may be** incomplete.
# 
# Given the integer n, return *the number of **complete rows** of the staircase you will build*.
# 
# **Example 1:**
# 
# []()
# 
# ![v2.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4bd91cfa-d2b1-47b3-8197-a72e8dcfff4b/v2.jpg)
# 
# **Input:** n = 5
# 
# **Output:** 2
# 
# **Explanation:** Because the 3rd row is incomplete, we return 2.
# 
# </aside>

# In[5]:


def arrange_coins(n):
    left, right = 0, n

    while left <= right:
        mid = (left + right) // 2
        total_coins = mid * (mid + 1) // 2

        if total_coins == n:
            return mid  # Found the highest complete row
        elif total_coins < n:
            left = mid + 1  # Move to the right half of the range
        else:
            right = mid - 1  # Move to the left half of the range

    return right  # Return the highest complete row found

# Example usage:
n = 5
result = arrange_coins(n)
print(result)  # Output: 2


# # <aside>
# 💡 **Question 6**
# Given an integer array nums sorted in **non-decreasing** order, return *an array of **the squares of each number** sorted in non-decreasing order*.
# 
# **Example 1:**
# 
# Input: nums = [-4,-1,0,3,10]
# 
# Output: [0,1,9,16,100]
# 
# **Explanation:** After squaring, the array becomes [16,1,0,9,100].
# After sorting, it becomes [0,1,9,16,100]
# 
# </aside>

# In[6]:


def sorted_squares(nums):
    n = len(nums)
    result = [0] * n
    left, right = 0, n - 1

    for i in range(n - 1, -1, -1):
        left_square = nums[left] * nums[left]
        right_square = nums[right] * nums[right]

        if left_square > right_square:
            result[i] = left_square
            left += 1
        else:
            result[i] = right_square
            right -= 1

    return result

# Example usage:
nums = [-4, -1, 0, 3, 10]
result = sorted_squares(nums)
print(result)  # Output: [0, 1, 9, 16, 100]


# # <aside>
# 💡 **Question 7**
# You are given an m x n matrix M initialized with all 0's and an array of operations ops, where ops[i] = [ai, bi] means M[x][y] should be incremented by one for all 0 <= x < ai and 0 <= y < bi.
# 
# Count and return *the number of maximum integers in the matrix after performing all the operations*
# 
# **Example 1:**
# 
# ![q4.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4d0890d0-7bc7-4f59-be8e-352d9f3c1c52/q4.jpg)
# 
# **Input:** m = 3, n = 3, ops = [[2,2],[3,3]]
# 
# **Output:** 4
# 
# **Explanation:** The maximum integer in M is 2, and there are four of it in M. So return 4.
# 
# </aside>

# In[7]:


def max_count(m, n, ops):
    min_x = m
    min_y = n

    for x, y in ops:
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    return min_x * min_y

# Example usage:
m = 3
n = 3
ops = [[2, 2], [3, 3]]
result = max_count(m, n, ops)
print(result)  # Output: 4


# # <aside>
# 💡 **Question 8**
# 
# Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].
# 
# *Return the array in the form* [x1,y1,x2,y2,...,xn,yn].
# 
# **Example 1:**
# 
# **Input:** nums = [2,5,1,3,4,7], n = 3
# 
# **Output:** [2,3,5,4,1,7]
# 
# **Explanation:** Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].
# 
# </aside>

# In[8]:


def shuffle(nums, n):
    result = []
    for i in range(n):
        result.append(nums[i])
        result.append(nums[i + n])
    return result

# Example usage:
nums = [2, 5, 1, 3, 4, 7]
n = 3
result = shuffle(nums, n)
print(result)  # Output: [2, 3, 5, 4, 1, 7]


# In[ ]:





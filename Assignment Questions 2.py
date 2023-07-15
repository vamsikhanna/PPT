#!/usr/bin/env python
# coding: utf-8

# # <aside>
# 💡 **Question 1**
# Given an integer array nums of 2n integers, group these integers into n pairs (a1, b1), (a2, b2),..., (an, bn) such that the sum of min(ai, bi) for all i is maximized. Return the maximized sum.
# 
# **Example 1:**
# Input: nums = [1,4,3,2]
# Output: 4
# 
# **Explanation:** All possible pairings (ignoring the ordering of elements) are:
# 
# 1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
# 2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
# 3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
# So the maximum possible sum is 4
# </aside>

# In[1]:


def array_pair_sum(nums):
    # Sort the array in non-decreasing order
    nums.sort()

    # Calculate the sum of the minimum values of pairs formed by adjacent elements
    max_sum = 0
    for i in range(0, len(nums), 2):
        max_sum += nums[i]

    return max_sum

# Example usage:
nums = [1, 4, 3, 2]
result = array_pair_sum(nums)
print(result)  # Output: 4


# # Question 2
# Alice has n candies, where the ith candy is of type candyType[i]. Alice noticed that she started to gain weight, so she visited a doctor. 
# 
# The doctor advised Alice to only eat n / 2 of the candies she has (n is always even). Alice likes her candies very much, and she wants to eat the maximum number of different types of candies while still following the doctor's advice. 
# 
# Given the integer array candyType of length n, return the maximum number of different types of candies she can eat if she only eats n / 2 of them.
# 
# Example 1:
# Input: candyType = [1,1,2,2,3,3]
# Output: 3
# 
# Explanation: Alice can only eat 6 / 2 = 3 candies. Since there are only 3 types, she can eat one of each type.

# In[2]:


def max_different_types(candyType):
    # Calculate the maximum number of different types of candies Alice can eat
    max_types = min(len(set(candyType)), len(candyType) // 2)
    return max_types

# Example usage:
candyType = [1, 1, 2, 2, 3, 3]
result = max_different_types(candyType)
print(result)  # Output: 3


# # Question 3
# We define a harmonious array as an array where the difference between its maximum value
# and its minimum value is exactly 1.
# 
# Given an integer array nums, return the length of its longest harmonious subsequence
# among all its possible subsequences.
# 
# A subsequence of an array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.
# 
# Example 1:
# Input: nums = [1,3,2,2,5,2,3,7]
# Output: 5
# 
# Explanation: The longest harmonious subsequence is [3,2,2,2,3].

# In[3]:


def find_longest_harmonious_subsequence(nums):
    # Create a dictionary to count the occurrences of each number in the array
    num_count = {}
    for num in nums:
        num_count[num] = num_count.get(num, 0) + 1

    max_length = 0

    # For each number, check if its adjacent numbers are present in the dictionary
    for num in num_count:
        if num + 1 in num_count:
            max_length = max(max_length, num_count[num] + num_count[num + 1])
    
    return max_length

# Example usage:
nums = [1, 3, 2, 2, 5, 2, 3, 7]
result = find_longest_harmonious_subsequence(nums)
print(result)  # Output: 5


# # Question 4
# You have a long flowerbed in which some of the plots are planted, and some are not.
# However, flowers cannot be planted in adjacent plots.
# Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers rule and false otherwise.
# 
# Example 1:
# Input: flowerbed = [1,0,0,0,1], n = 1
# Output: true

# In[4]:


def can_place_flowers(flowerbed, n):
    i = 0
    while i < len(flowerbed):
        # Check if the current plot and its adjacent plots are empty (0)
        if flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0):
            flowerbed[i] = 1  # Plant a flower in the current plot
            n -= 1  # Decrement n, indicating a flower has been planted

        if n == 0:
            return True  # All n flowers have been planted

        i += 1

    return False  # Not enough empty plots to plant all n flowers

# Example usage:
flowerbed = [1, 0, 0, 0, 1]
n = 1
result = can_place_flowers(flowerbed, n)
print(result)  # Output: True


# # Question 5
# Given an integer array nums, find three numbers whose product is maximum and return the maximum product.
# 
# Example 1:
# Input: nums = [1,2,3]
# Output: 6

# In[ ]:


def maximum_product_of_three(nums):
    # Sort the array in non-decreasing order
    nums.sort()

    # Calculate the maximum product of three numbers
    n = len(nums)
    return max(nums[n-1] * nums[n-2] * nums[n-3], nums[0] * nums[1] * nums[n-1])

# Example usage:
nums = [1, 2, 3]
result = maximum_product_of_three(nums)
print(result)  # Output: 6


# # Question 6
# Given an array of integers nums which is sorted in ascending order, and an integer target,
# write a function to search target in nums. If target exists, then return its index. Otherwise,
# return -1.
# 
# You must write an algorithm with O(log n) runtime complexity.
# 
# Input: nums = [-1,0,3,5,9,12], target = 9
# Output: 4
# 
# Explanation: 9 exists in nums and its index is 4

# In[5]:


def binary_search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Example usage:
nums = [-1, 0, 3, 5, 9, 12]
target = 9
result = binary_search(nums, target)
print(result)  # Output: 4


# # Question 7
# An array is monotonic if it is either monotone increasing or monotone decreasing.
# 
# An array nums is monotone increasing if for all i <= j, nums[i] <= nums[j]. An array nums is
# monotone decreasing if for all i <= j, nums[i] >= nums[j].
# 
# Given an integer array nums, return true if the given array is monotonic, or false otherwise.
# 
# Example 1:
# Input: nums = [1,2,2,3]
# Output: true

# In[6]:


def is_monotonic(nums):
    # Check if the array is monotone increasing
    def is_increasing():
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                return False
        return True

    # Check if the array is monotone decreasing
    def is_decreasing():
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                return False
        return True

    # Return True if the array is monotone increasing or decreasing
    return is_increasing() or is_decreasing()

# Example usage:
nums = [1, 2, 2, 3]
result = is_monotonic(nums)
print(result)  # Output: True


# # Question 8
# You are given an integer array nums and an integer k.
# 
# In one operation, you can choose any index i where 0 <= i < nums.length and change nums[i] to nums[i] + x where x is an integer from the range [-k, k]. You can apply this operation at most once for each index i.
# 
# The score of nums is the difference between the maximum and minimum elements in nums.
# 
# Return the minimum score of nums after applying the mentioned operation at most once for each index in it.
# 
# Example 1:
# Input: nums = [1], k = 0
# Output: 0
# 
# Explanation: The score is max(nums) - min(nums) = 1 - 1 = 0.

# In[7]:


def minimum_score(nums, k):
    nums.sort()  # Sort the array in non-decreasing order

    # Apply the operation to minimize the difference between maximum and minimum elements
    for i in range(len(nums)):
        # Update the minimum element
        nums[i] = max(nums[i] - k, nums[0] + k)

    # Return the minimum score
    return max(nums) - min(nums)

# Example usage:
nums = [1]
k = 0
result = minimum_score(nums, k)
print(result)  # Output: 0


# In[ ]:





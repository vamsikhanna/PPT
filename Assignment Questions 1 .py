#!/usr/bin/env python
# coding: utf-8

# # 💡 **Q1.** Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# 
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# 
# You can return the answer in any order.
# 
# **Example:**
# Input: nums = [2,7,11,15], target = 9
# Output0 [0,1]
# 
# **Explanation:** Because nums[0] + nums[1] == 9, we return [0, 1]
# 
# </aside>

# In[1]:


def two_sum(nums, target):
    # Create a dictionary to store the values and their indices
    num_dict = {}

    # Iterate through the array
    for i, num in enumerate(nums):
        # Calculate the complement needed to reach the target
        complement = target - num

        # Check if the complement is already in the dictionary
        if complement in num_dict:
            # Return the indices of the two numbers
            return [num_dict[complement], i]

        # If the complement is not in the dictionary, store the current number and its index
        num_dict[num] = i

    # If no valid pair is found, return an empty list or handle the situation as required.
    return []


# In[2]:


# Example usage:
nums = [2, 7, 11, 15]
target = 9
print(two_sum(nums, target))  # Output: [0, 1]


# #
# 💡 **Q2.** Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.
# 
# Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:
# 
# - Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
# - Return k.
# 
# **Example :**
# Input: nums = [3,2,2,3], val = 3
# Output: 2, nums = [2,2,_*,_*]
# 
# **Explanation:** Your function should return k = 2, with the first two elements of nums being 2. It does not matter what you leave beyond the returned k (hence they are underscores)[
# 
# </aside>

# In[3]:


def remove_element(nums, val):
    # Initialize two pointers, one for iteration and one for keeping track of the non-val elements
    k = 0

    # Iterate through the array
    for i in range(len(nums)):
        # If the current element is not equal to val, replace the element at the k-th position with the current element
        if nums[i] != val:
            nums[k] = nums[i]
            k += 1

    # The first k elements contain the elements which are not equal to val
    # Return k, which represents the number of elements not equal to val
    return k


# In[4]:


# Example usage:
nums = [3, 2, 2, 3]
val = 3
result = remove_element(nums, val)
print(result)  # Output: 2, nums = [2, 2, _, _]


# # <aside>
# 💡 **Q3.** Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
# 
# You must write an algorithm with O(log n) runtime complexity.
# 
# **Example 1:**
# Input: nums = [1,3,5,6], target = 5
# 
# Output: 2
# 
# </aside>

# In[5]:


def search_insert(nums, target):
    # Initialize the pointers for binary search
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # At this point, the target is not found in the array.
    # The 'left' pointer indicates the index where the target should be inserted.
    return left


# In[6]:


# Example usage:
nums = [1, 3, 5, 6]
target = 5
result = search_insert(nums, target)
print(result)  # Output: 2


# # <aside>
# 💡 **Q4.** You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
# 
# Increment the large integer by one and return the resulting array of digits.
# 
# **Example 1:**
# Input: digits = [1,2,3]
# Output: [1,2,4]
# 
# **Explanation:** The array represents the integer 123.
# 
# Incrementing by one gives 123 + 1 = 124.
# Thus, the result should be [1,2,4].
# 
# </aside>

# In[8]:


def increment_large_integer(digits):
    n = len(digits)

    # Start from the least significant digit
    carry = 1
    for i in range(n - 1, -1, -1):
        sum_val = digits[i] + carry
        digits[i] = sum_val % 10
        carry = sum_val // 10

    # If there's a carry left after the loop, insert it at the beginning of the array
    if carry:
        digits.insert(0, carry)

    return digits


# In[9]:


# Example usage:
digits = [1, 2, 3]
result = increment_large_integer(digits)
print(result)  # Output: [1, 2, 4]


# # <aside>
# 💡 **Q5.** You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.
# 
# Merge nums1 and nums2 into a single array sorted in non-decreasing order.
# 
# The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
# 
# **Example 1:**
# Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
# Output: [1,2,2,3,5,6]
# 
# **Explanation:** The arrays we are merging are [1,2,3] and [2,5,6].
# The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
# 
# </aside>

# In[10]:


def merge_sorted_arrays(nums1, m, nums2, n):
    # Initialize pointers for nums1, nums2, and the merged array
    p1, p2, p_merged = m - 1, n - 1, m + n - 1

    # Merge the arrays from right to left
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] >= nums2[p2]:
            nums1[p_merged] = nums1[p1]
            p1 -= 1
        else:
            nums1[p_merged] = nums2[p2]
            p2 -= 1
        p_merged -= 1

    # If there are remaining elements in nums2, copy them to the merged array
    while p2 >= 0:
        nums1[p_merged] = nums2[p2]
        p2 -= 1
        p_merged -= 1


# In[11]:


# Example usage:
nums1 = [1, 2, 3, 0, 0, 0]
m = 3
nums2 = [2, 5, 6]
n = 3
merge_sorted_arrays(nums1, m, nums2, n)
print(nums1)  # Output: [1, 2, 2, 3, 5, 6]


# # <aside>
# 💡 **Q6.** Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
# 
# **Example 1:**
# Input: nums = [1,2,3,1]
# 
# Output: true
# 
# </aside>

# In[12]:


def contains_duplicate(nums):
    # Create an empty set to store encountered elements
    num_set = set()

    # Iterate through the array
    for num in nums:
        # If the element is already in the set, there is a duplicate
        if num in num_set:
            return True
        # Add the element to the set
        num_set.add(num)

    # If no duplicates are found, return False
    return False


# In[13]:


# Example usage:
nums = [1, 2, 3, 1]
result = contains_duplicate(nums)
print(result)  # Output: True


# # <aside>
# 💡 **Q7.** Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the nonzero elements.
# 
# Note that you must do this in-place without making a copy of the array.
# 
# **Example 1:**
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]
# 
# </aside>

# In[14]:


def move_zeros_to_end(nums):
    # Initialize a pointer to keep track of the position to place the next nonzero element
    next_nonzero = 0

    # Iterate through the array
    for num in nums:
        # If the current element is nonzero, move it to the position pointed by next_nonzero
        if num != 0:
            nums[next_nonzero] = num
            next_nonzero += 1

    # Set the remaining elements from next_nonzero to the end of the array to zero
    while next_nonzero < len(nums):
        nums[next_nonzero] = 0
        next_nonzero += 1


# In[15]:


# Example usage:
nums = [0, 1, 0, 3, 12]
move_zeros_to_end(nums)
print(nums)  # Output: [1, 3, 12, 0, 0]


# # <aside>
# 💡 **Q8.** You have a set of integers s, which originally contains all the numbers from 1 to n. Unfortunately, due to some error, one of the numbers in s got duplicated to another number in the set, which results in repetition of one number and loss of another number.
# 
# You are given an integer array nums representing the data status of this set after the error.
# 
# Find the number that occurs twice and the number that is missing and return them in the form of an array.
# 
# **Example 1:**
# Input: nums = [1,2,2,4]
# Output: [2,3]
# 
# </aside>

# In[16]:


def find_error_nums(nums):
    n = len(nums)
    
    # Calculate the expected sum of numbers from 1 to n (assuming no errors)
    expected_sum = (n * (n + 1)) // 2
    
    # Calculate the actual sum of elements in the nums array
    actual_sum = sum(nums)
    
    # Find the missing number by subtracting the actual sum from the expected sum
    missing_num = expected_sum - actual_sum
    
    # Find the duplicate number using a set
    num_set = set()
    duplicate_num = None
    for num in nums:
        if num in num_set:
            duplicate_num = num
            break
        num_set.add(num)
    
    return [duplicate_num, missing_num]


# In[17]:


# Example usage:
nums = [1, 2, 2, 4]
result = find_error_nums(nums)
print(result)  # Output: [2, 3]


# In[ ]:





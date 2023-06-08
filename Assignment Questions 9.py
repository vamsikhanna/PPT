#!/usr/bin/env python
# coding: utf-8

# Given an integer `n`, return *`true` if it is a power of two. Otherwise, return `false`*.
# 
# An integer `n` is a power of two, if there exists an integer `x` such that `n == 2x`.
# 
# **Example 1:**
# Input: n = 1 
# 
# Output: true
# 
# **Example 2:**
# Input: n = 16 
# 
# Output: true
# 
# **Example 3:**
# Input: n = 3 
# 
# Output: false

# In[1]:


def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0

# Example usage
print(isPowerOfTwo(1))  # Output: True
print(isPowerOfTwo(16))  # Output: True
print(isPowerOfTwo(3))  # Output: False


# Given a number n, find the sum of the first natural numbers.
# 
# **Example 1:**
# 
# Input: n = 3 
# 
# Output: 6
# 
# **Example 2:**
# 
# Input  : 5 
# 
# Output : 15

# In[2]:


def sumOfFirstNNumbers(n):
    return (n * (n + 1)) // 2

# Example usage
print(sumOfFirstNNumbers(3))  # Output: 6
print(sumOfFirstNNumbers(5))  # Output: 15


# Given a positive integer, N. Find the factorial of N. 
# 
# **Example 1:**
# 
# Input: N = 5 
# 
# Output: 120
# 
# **Example 2:**
# 
# Input: N = 4
# 
# Output: 24

# In[3]:


def factorial(N):
    if N == 1:
        return 1
    else:
        return N * factorial(N - 1)

# Example usage
print(factorial(5))  # Output: 120
print(factorial(4))  # Output: 24


# Given a number N and a power P, the task is to find the exponent of this number raised to the given power, i.e. N^P.
# 
# **Example 1 :** 
# 
# Input: N = 5, P = 2
# 
# Output: 25
# 
# **Example 2 :**
# Input: N = 2, P = 5
# 
# Output: 32

# In[4]:


def exponentiation(N, P):
    return N ** P

# Example usage
print(exponentiation(5, 2))  # Output: 25
print(exponentiation(2, 5))  # Output: 32


# Given an array of integers **arr**, the task is to find maximum element of that array using recursion.
# 
# **Example 1:**
# 
# Input: arr = {1, 4, 3, -5, -4, 8, 6};
# Output: 8
# 
# **Example 2:**
# 
# Input: arr = {1, 4, 45, 6, 10, -8};
# Output: 45

# In[5]:


def findMax(arr, n):
    # Base case: if n becomes 1, return the single element as maximum
    if n == 1:
        return arr[0]

    # Recursive case: divide the array into two halves
    mid = n // 2
    left_max = findMax(arr[:mid], mid)
    right_max = findMax(arr[mid:], n - mid)

    # Compare the maximum of the left half with the maximum of the right half
    if left_max > right_max:
        return left_max
    else:
        return right_max

# Example usage
arr = [1, 4, 3, -5, -4, 8, 6]
print(findMax(arr, len(arr)))  # Output: 8

arr = [1, 4, 45, 6, 10, -8]
print(findMax(arr, len(arr)))  # Output: 45


# Given first term (a), common difference (d) and a integer N of the Arithmetic Progression series, the task is to find Nth term of the series.
# 
# **Example 1:**
# 
# Input : a = 2 d = 1 N = 5
# Output : 6
# The 5th term of the series is : 6
# 
# **Example 2:**
# 
# Input : a = 5 d = 2 N = 10
# Output : 23
# The 10th term of the series is : 23

# In[6]:


def findNthTerm(a, d, N):
    nth_term = a + (N - 1) * d
    return nth_term

# Example usage
print(findNthTerm(2, 1, 5))  # Output: 6
print(findNthTerm(5, 2, 10))  # Output: 23


# Given a string S, the task is to write a program to print all permutations of a given string.
# 
# **Example 1:**
# 
# ***Input:***
# 
# *S = “ABC”*
# 
# ***Output:***
# 
# *“ABC”, “ACB”, “BAC”, “BCA”, “CBA”, “CAB”*
# 
# **Example 2:**
# 
# ***Input:***
# 
# *S = “XY”*
# 
# ***Output:***
# 
# *“XY”, “YX”*

# In[7]:


def permute(S, prefix, result):
    if len(S) == 0:
        result.append(prefix)
        return
    
    for i in range(len(S)):
        c = S[i]
        new_S = S[:i] + S[i+1:]
        permute(new_S, prefix + c, result)

def printAllPermutations(S):
    result = []
    permute(S, "", result)
    return result

# Example usage
print(printAllPermutations("ABC"))  # Output: ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA']
print(printAllPermutations("XY"))  # Output: ['XY', 'YX']


# Given an array, find a product of all array elements.
# 
# **Example 1:**
# 
# Input  : arr[] = {1, 2, 3, 4, 5}
# Output : 120
# **Example 2:**
# 
# Input  : arr[] = {1, 6, 3}
# Output : 18

# In[8]:


def findProduct(arr):
    product = 1
    for num in arr:
        product *= num
    return product

# Example usage
arr = [1, 2, 3, 4, 5]
print(findProduct(arr))  # Output: 120

arr = [1, 6, 3]
print(findProduct(arr))  # Output: 18


# In[ ]:





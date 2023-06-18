#!/usr/bin/env python
# coding: utf-8

# Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well. You must not use any built-in exponent function or operator. 
# 
#  Example 1:
# Input: x = 4 Output: 2 Explanation: The square root of 4 is 2, so we return 2.
# Example 2:
# 
# Input: x = 8 Output: 2 Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
# Constraints:
# 
# 0 <= x <= 2^31 - 1
# 
# Note: Create a GitHub file for the solution and add the file link the the answer section below.

# In[1]:


def mySqrt(x):
    if x <= 1:
        return x

    left = 0
    right = x

    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1

    return right


# In[2]:


mySqrt(4)


# In[4]:


x=8
mySqrt(8)


# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
# 
# You may assume the two numbers do not contain any leading zero, except the number 0 itself.
# 
# 
# Example 1:
# 
# Input: l1 = [2,4,3], l2 = [5,6,4] Output: [7,0,8] Explanation: 342 + 465 = 807.
# 
# Example 2:
# 
# Input: l1 = [0], l2 = [0] Output: [0]
# 
# Example 3:
# 
# Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9] Output: [8,9,9,9,0,0,0,1]
# 
#  
# 
# Constraints:
# 
# The number of nodes in each linked list is in the range [1, 100].
# 0 <= Node.val <= 9 It is guaranteed that the list represents a number that does not have leading zeros.
# 
# 

# In[5]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1, l2):
    dummy = ListNode()
    curr = dummy
    carry = 0

    while l1 or l2 or carry:
        sum = carry

        if l1:
            sum += l1.val
            l1 = l1.next

        if l2:
            sum += l2.val
            l2 = l2.next

        carry = sum // 10
        digit = sum % 10

        curr.next = ListNode(digit)
        curr = curr.next

    return dummy.next


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# Given two strings s1 and s2, return *the lowest **ASCII** sum of deleted characters to make two strings equal*.
# 
# **Example 1:**
# 
# **Input:** s1 = "sea", s2 = "eat"
# 
# **Output:** 231
# 
# **Explanation:** Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.
# 
# Deleting "t" from "eat" adds 116 to the sum.
# 
# At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.

# In[1]:


def minimumDeleteSum(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + ord(s1[i])
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    sum_ascii = sum(ord(c) for c in s1) + sum(ord(c) for c in s2)
    min_deleted_sum = sum_ascii - 2 * dp[m][n]

    return min_deleted_sum

# Example usage
s1 = "sea"
s2 = "eat"
result = minimumDeleteSum(s1, s2)
print(result)  # Output: 231


# Given a string s containing only three types of characters: '(', ')' and '*', return true *if* s *is **valid***.
# 
# The following rules define a **valid** string:
# 
# - Any left parenthesis '(' must have a corresponding right parenthesis ')'.
# - Any right parenthesis ')' must have a corresponding left parenthesis '('.
# - Left parenthesis '(' must go before the corresponding right parenthesis ')'.
# - '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
# 
# **Example 1:**
# 
# **Input:** s = "()"
# 
# **Output:**
# 
# true

# In[2]:


def isValid(s):
    stack = []

    for c in s:
        if c == '(' or c == '*':
            stack.append(c)
        elif c == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            elif stack and stack[-1] == '*':
                stack.pop()
            else:
                return False

    while stack:
        if stack[-1] == '(':
            stack.pop()
        elif stack[-1] == '*':
            stack.pop()
        else:
            break

    return len(stack) == 0

# Example usage
s = "()"
result = isValid(s)
print(result)  # Output: True


# Given two strings word1 and word2, return *the minimum number of **steps** required to make* word1 *and* word2 *the same*.
# 
# In one **step**, you can delete exactly one character in either string.
# 
# **Example 1:**
# 
# **Input:** word1 = "sea", word2 = "eat"
# 
# **Output:** 2
# 
# **Explanation:** You need one step to make "sea" to "ea" and another step to make "eat" to "ea".

# In[3]:


def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if word1[i] == word2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    return m + n - 2 * dp[m][n]

# Example usage
word1 = "sea"
word2 = "eat"
result = minDistance(word1, word2)
print(result)  # Output: 2


# You need to construct a binary tree from a string consisting of parenthesis and integers.
# 
# The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.
# You always start to construct the **left** child node of the parent first if it exists.

# In[5]:


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def constructTree(s):
    stack = []

    i = 0
    while i < len(s):
        if s[i] == '(':
            i += 1
        elif s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            node_val = int(s[i:j])
            node = TreeNode(node_val)
            if stack:
                parent = stack[-1]
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node
            stack.append(node)
            i = j
        elif s[i] == ')':
            stack.pop()
            i += 1

    return stack[-1]

# Example usage
s = "4(2(3)(1))(6(5))"
root = constructTree(s)
root


# Given an array of characters chars, compress it using the following algorithm:
# 
# Begin with an empty string s. For each group of **consecutive repeating characters** in chars:
# 
# - If the group's length is 1, append the character to s.
# - Otherwise, append the character followed by the group's length.
# 
# The compressed string s **should not be returned separately**, but instead, be stored **in the input character array chars**. Note that group lengths that are 10 or longer will be split into multiple characters in chars.
# 
# After you are done **modifying the input array,** return *the new length of the array*.
# 
# You must write an algorithm that uses only constant extra space.
# 
# **Example 1:**
# 
# **Input:** chars = ["a","a","b","b","c","c","c"]
# 
# **Output:** Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
# 
# **Explanation:**
# 
# The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".

# In[6]:


def compress(chars):
    write = 0
    count = 1

    for read in range(1, len(chars)):
        if chars[read] == chars[read - 1]:
            count += 1
        else:
            chars[write] = chars[read - 1]
            write += 1
            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1
            count = 1

    chars[write] = chars[-1]
    write += 1
    if count > 1:
        for digit in str(count):
            chars[write] = digit
            write += 1

    return write

# Example usage
chars = ["a","a","b","b","c","c","c"]
new_length = compress(chars)
print(new_length)  # Output: 6
print(chars[:new_length])  # Output: ["a","2","b","2","c","3"]


# Given an encoded string, return its decoded string.
# 
# The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.
# 
# You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].
# 
# The test cases are generated so that the length of the output will never exceed 105.
# 
# **Example 1:**
# 
# **Input:** s = "3[a]2[bc]"
# 
# **Output:** "aaabcbc"

# In[7]:


def decodeString(s):
    stack = []

    for c in s:
        if c.isdigit():
            stack.append(int(c))
        elif c == '[':
            stack.append('[')
        elif c == ']':
            substring = ''
            while stack and stack[-1] != '[':
                substring = stack.pop() + substring

            stack.pop()  # Remove '[' from the stack

            repetitions = stack.pop()
            stack.append(substring * repetitions)
        else:
            stack.append(c)

    return ''.join(stack[::-1])

# Example usage
s = "3[a]2[bc]"
decoded_string = decodeString(s)
print(decoded_string)  # Output: "aaabcbc"


# Given two strings s and goal, return true *if you can swap two letters in* s *so the result is equal to* goal*, otherwise, return* false*.*
# 
# Swapping letters is defined as taking two indices i and j (0-indexed) such that i != j and swapping the characters at s[i] and s[j].
# 
# - For example, swapping at indices 0 and 2 in "abcd" results in "cbad".
# 
# **Example 1:**
# 
# **Input:** s = "ab", goal = "ba"
# 
# **Output:** true
# 
# **Explanation:** You can swap s[0] = 'a' and s[1] = 'b' to get "ba", which is equal to goal

# In[8]:


def buddyStrings(s, goal):
    if s == goal:
        # Condition 1: If s and goal are equal, no swaps needed
        # We can swap two identical letters in s to get s again
        # So, we return True if s has at least one repeated letter
        return len(set(s)) < len(s)

    if len(s) != len(goal):
        # Condition 2: If the lengths are different, no swaps can make s equal to goal
        return False

    diff_indices = []
    for i in range(len(s)):
        if s[i] != goal[i]:
            diff_indices.append(i)

    if len(diff_indices) != 2:
        # Condition 3: More than two indices differ or the characters at these indices are different
        return False

    i, j = diff_indices
    return s[i] == goal[j] and s[j] == goal[i]

# Example usage
s = "ab"
goal = "ba"
print(buddyStrings(s, goal))  # Output: True


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Given two strings s and t, *determine if they are isomorphic*.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.

**Example 1:**

**Input:** s = "egg", t = "add"

**Output:** true


# In[1]:


def isIsomorphic(s, t):
    if len(s) != len(t):
        return False

    s_map = {}
    t_map = {}

    for char_s, char_t in zip(s, t):
        if char_s in s_map:
            if s_map[char_s] != char_t:
                return False
        else:
            s_map[char_s] = char_t

        if char_t in t_map:
            if t_map[char_t] != char_s:
                return False
        else:
            t_map[char_t] = char_s

    return True


# In[2]:


s = "egg"
t = "add"

print(isIsomorphic(s, t))


# In[ ]:


<aside>
💡 **Question 2**

Given a string num which represents an integer, return true *if* num *is a **strobogrammatic number***.

A **strobogrammatic number** is a number that looks the same when rotated 180 degrees (looked at upside down).

**Example 1:**

**Input:** num = "69"

**Output:**

true

</aside>


# In[3]:


def isStrobogrammatic(num):
    rotation_map = {
        '0': '0',
        '1': '1',
        '6': '9',
        '8': '8',
        '9': '6'
    }

    left, right = 0, len(num) - 1

    while left <= right:
        if num[left] not in rotation_map or num[right] != rotation_map[num[left]]:
            return False
        left += 1
        right -= 1

    return True


# In[4]:


num = "69"

print(isStrobogrammatic(num))


# In[ ]:


<aside>
💡 **Question 3**

Given two non-negative integers, num1 and num2 represented as string, return *the sum of* num1 *and* num2 *as a string*.

You must solve the problem without using any built-in library for handling large integers (such as BigInteger). You must also not convert the inputs to integers directly.

**Example 1:**

**Input:** num1 = "11", num2 = "123"

**Output:**

"134"

</aside>


# In[5]:


def addStrings(num1, num2):
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    result = ""

    while i >= 0 or j >= 0 or carry != 0:
        if i >= 0:
            carry += int(num1[i])
        if j >= 0:
            carry += int(num2[j])
        result += str(carry % 10)
        carry //= 10
        i -= 1
        j -= 1

    return result[::-1]


# In[6]:


num1 = "11"
num2 = "123"

print(addStrings(num1, num2))


# <aside>
# 💡 **Question 4**
# 
# Given a string s, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.
# 
# **Example 1:**
# 
# **Input:** s = "Let's take LeetCode contest"
# 
# **Output:** "s'teL ekat edoCteeL tsetnoc"
# 
# </aside>

# In[7]:


def reverseWords(s):
    words = s.split()
    reversed_words = [word[::-1] for word in words]
    reversed_sentence = ' '.join(reversed_words)
    return reversed_sentence


# In[8]:


s = "Let's take LeetCode contest"

print(reverseWords(s))


# In[ ]:


<aside>
💡 **Question 5**

Given a string s and an integer k, reverse the first k characters for every 2k characters counting from the start of the string.

If there are fewer than k characters left, reverse all of them. If there are less than 2k but greater than or equal to k characters, then reverse the first k characters and leave the other as original.

**Example 1:**

**Input:** s = "abcdefg", k = 2

**Output:**

"bacdfeg"

</aside>


# In[9]:


def reverseStr(s, k):
    chars = list(s)
    n = len(chars)

    for i in range(0, n, 2 * k):
        chars[i:i+k] = reversed(chars[i:i+k])

    return ''.join(chars)


# In[10]:


s = "abcdefg"
k = 2

print(reverseStr(s, k))


# In[ ]:


<aside>
💡 **Question 6**

Given two strings s and goal, return true *if and only if* s *can become* goal *after some number of **shifts** on* s.

A **shift** on s consists of moving the leftmost character of s to the rightmost position.

- For example, if s = "abcde", then it will be "bcdea" after one shift.

**Example 1:**

**Input:** s = "abcde", goal = "cdeab"

**Output:**

true

</aside>


# In[11]:


def rotateString(s, goal):
    if len(s) != len(goal):
        return False

    s_double = s + s

    if goal in s_double:
        return True
    else:
        return False


# In[12]:


s = "abcde"
goal = "cdeab"

print(rotateString(s, goal))


# In[ ]:


<aside>
💡 **Question 7**

Given two strings s and t, return true *if they are equal when both are typed into empty text editors*. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

**Example 1:**

**Input:** s = "ab#c", t = "ad#c"

**Output:** true

**Explanation:**

Both s and t become "ac".

</aside>


# In[13]:


def backspaceCompare(s, t):
    stack_s = []
    stack_t = []

    for char in s:
        if char != '#':
            stack_s.append(char)
        elif stack_s:
            stack_s.pop()

    for char in t:
        if char != '#':
            stack_t.append(char)
        elif stack_t:
            stack_t.pop()

    final_s = ''.join(stack_s)
    final_t = ''.join(stack_t)

    return final_s == final_t


# In[14]:


s = "ab#c"
t = "ad#c"

print(backspaceCompare(s, t))


# In[ ]:


You are given an array coordinates, coordinates[i] = [x, y], where [x, y] represents the coordinate of a point. Check if these points make a straight line in the XY plane.

**Example 1:**


# In[ ]:


**Input:** coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]

**Output:** true


# In[15]:


def checkStraightLine(coordinates):
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    initial_slope = (y2 - y1) / (x2 - x1)

    for i in range(2, len(coordinates)):
        xi, yi = coordinates[i]
        xi_1, yi_1 = coordinates[i-1]
        slope = (yi - yi_1) / (xi - xi_1)
        if slope != initial_slope:
            return False

    return True


# In[16]:


coordinates = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]

print(checkStraightLine(coordinates))


# In[ ]:





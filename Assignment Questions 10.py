#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Given an integer `n`, return *`true` if it is a power of three. Otherwise, return `false`*.

An integer `n` is a power of three, if there exists an integer `x` such that `n == 3x`.

**Example 1:**Input: n = 27
Output: true
Explanation: 27 = 33


# In[1]:


import math

def isPowerOfThree(n):
    if n <= 0:
        return False
    x = math.log10(n) / math.log10(3)
    return x.is_integer()

# Example usage
print(isPowerOfThree(27))  # Output: True
print(isPowerOfThree(9))  # Output: True
print(isPowerOfThree(45))  # Output: False


# In[ ]:





# In[ ]:





# Given a set represented as a string, write a recursive code to print all subsets of it. The subsets can be printed in any order.
# 
# **Example 1:**
# 
# Input :  set = “abc”
# 
# Output : { “”, “a”, “b”, “c”, “ab”, “ac”, “bc”, “abc”}
# 
# **Example 2:**
# 
# Input : set = “abcd”
# 
# Output : { “”, “a” ,”ab” ,”abc” ,”abcd”, “abd” ,”ac” ,”acd”, “ad” ,”b”, “bc” ,”bcd” ,”bd” ,”c” ,”cd” ,”d” }

# In[2]:


def generateSubsets(set_str, subset, index, result):
    if index == len(set_str):
        result.append(subset)
        return
    
    generateSubsets(set_str, subset + set_str[index], index + 1, result)
    generateSubsets(set_str, subset, index + 1, result)

def printAllSubsets(set_str):
    result = []
    generateSubsets(set_str, "", 0, result)
    return result

# Example usage
print(printAllSubsets("abc"))  # Output: ['', 'c', 'b', 'bc', 'a', 'ac', 'ab', 'abc']
print(printAllSubsets("abcd"))  # Output: ['', 'd', 'c', 'cd', 'b', 'bd', 'bc', 'bcd', 'a', 'ad', 'ac', 'acd', 'ab', 'abd', 'abc', 'abcd']


# Given a string calculate length of the string using recursion.
# 
# **Examples:**Input : str = "abcd"
# Output :4
# 
# Input : str = "GEEKSFORGEEKS"
# Output :13

# In[4]:


def stringLength(str):
    # Base case: if the string is empty, return 0
    if str == "":
        return 0

    # Recursive case: add 1 to the length of the string without the first character
    return 1 + stringLength(str[1:])

# Example usage
print(stringLength("abcd"))  # Output: 4
print(stringLength("GEEKSFORGEEKS"))  # Output: 13


# We are given a string S, we need to find count of all contiguous substrings starting and ending with same character.
# 
# **Examples :**Input  : S = "abcab"
# Output : 7
# There are 15 substrings of "abcab"
# a, ab, abc, abca, abcab, b, bc, bca
# bcab, c, ca, cab, a, ab, b
# Out of the above substrings, there
# are 7 substrings : a, abca, b, bcab,
# c, a and b.
# 
# Input  : S = "aba"
# Output : 4
# The substrings are a, b, a and aba

# In[5]:


def countSubstrings(S):
    count = 0
    for i in range(len(S)):
        for j in range(i, len(S)):
            if S[i] == S[j]:
                count += 1
    return count

# Example usage
print(countSubstrings("abcab"))  # Output: 7
print(countSubstrings("aba"))  # Output: 4


# In[ ]:


The tower of Hanoi is a famous puzzle where we have three rods and N disks. The objective of the puzzle is to move the entire stack to another rod. You are given the number of discs N. Initially, these discs are in the rod 1. You need to print all the steps of discs movement so that all the discs reach the 3rd rod. Also, you need to find the total moves.Note: The discs are arranged such that the top disc is numbered 1 and the bottom-most disc is numbered N. Also, all the discs have different sizes and a bigger disc cannot be put on the top of a smaller disc. Refer the provided link to get a better clarity about the puzzle.


# In[ ]:


Input:
N = 2
Output:
move disk 1 from rod 1 to rod 2
move disk 2 from rod 1 to rod 3
move disk 1 from rod 2 to rod 3
3
Explanation:For N=2 , steps will be
as follows in the example and total
3 steps will be taken.


# In[6]:


def towerOfHanoi(n, source, destination, auxiliary):
    if n == 1:
        print("move disk", n, "from rod", source, "to rod", destination)
        return 1
    else:
        moves = 0
        moves += towerOfHanoi(n-1, source, auxiliary, destination)
        print("move disk", n, "from rod", source, "to rod", destination)
        moves += 1
        moves += towerOfHanoi(n-1, auxiliary, destination, source)
        return moves

# Example usage
N = 2
total_moves = towerOfHanoi(N, 1, 3, 2)
print("Total moves:", total_moves)


# In[ ]:


N = 3
Output:
move disk 1 from rod 1 to rod 3
move disk 2 from rod 1 to rod 2
move disk 1 from rod 3 to rod 2
move disk 3 from rod 1 to rod 3
move disk 1 from rod 2 to rod 1
move disk 2 from rod 2 to rod 3
move disk 1 from rod 1 to rod 3
7
Explanation:For N=3 , steps will be
as follows in the example and total
7 steps will be taken.


# In[7]:


def towerOfHanoi(n, source, destination, auxiliary):
    if n == 1:
        print("move disk", n, "from rod", source, "to rod", destination)
        return 1
    else:
        moves = 0
        moves += towerOfHanoi(n-1, source, auxiliary, destination)
        print("move disk", n, "from rod", source, "to rod", destination)
        moves += 1
        moves += towerOfHanoi(n-1, auxiliary, destination, source)
        return moves

# Example usage
N = 3
total_moves = towerOfHanoi(N, 1, 3, 2)
print("Total moves:", total_moves)


# Given a string **str**, the task is to print all the permutations of **str**. A **permutation** is an arrangement of all or part of a set of objects, with regard to the order of the arrangement. For instance, the words ‘bat’ and ‘tab’ represents two distinct permutation (or arrangements) of a similar three letter word.
# 
# **Examples:**
# 
# > Input: str = “cd”
# > 
# > 
# > **Output:** cd dc
# > 
# > **Input:** str = “abb”
# > 
# > **Output:** abb abb bab bba bab bba
# >

# In[9]:


def permute(s, l, r):
    if l == r:
        print("".join(s))
    else:
        for i in range(l, r + 1):
            s[l], s[i] = s[i], s[l]
            permute(s, l + 1, r)
            s[l], s[i] = s[i], s[l]

# Function to print all permutations of a string
def printPermutations(str):
    n = len(str)
    s = list(str)
    permute(s, 0, n - 1)

# Example usage
str = "abb"
printPermutations(str)


# Given a string, count total number of consonants in it. A consonant is an English alphabet character that is not vowel (a, e, i, o and u). Examples of constants are b, c, d, f, and g.

# In[ ]:


Input : abc de
Output : 3
There are three consonants b, c and d.

Input : geeksforgeeks portal
Output : 12


# In[10]:


def countConsonants(str):
    consonants = 0
    vowels = ['a', 'e', 'i', 'o', 'u']

    for char in str:
        if char.isalpha() and char.lower() not in vowels:
            consonants += 1

    return consonants

# Example usage
str = "geeksforgeeks portal"
result = countConsonants(str)
print(result)


# In[ ]:





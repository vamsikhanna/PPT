#!/usr/bin/env python
# coding: utf-8

# Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.
# 
# Input: s = "leetcode"
# Output: 0
# Input: s = "loveleetcode"
# Output: 2
# 

# In[1]:


def firstUniqChar(s):
    char_freq = {}
    
    # Count the frequency of each character
    for char in s:
        if char in char_freq:
            char_freq[char] += 1
        else:
            char_freq[char] = 1
    
    # Find the first character with frequency 1
    for i in range(len(s)):
        if char_freq[s[i]] == 1:
            return i
    
    return -1


# In[2]:


# Test the function
s = "leetcode"
print(firstUniqChar(s))  # Output: 0

s = "loveleetcode"
print(firstUniqChar(s))  # Output: 2


# Given a **circular integer array** `nums` of length `n`, return *the maximum possible sum of a non-empty **subarray** of* `nums`.
# 
# A **circular array** means the end of the array connects to the beginning of the array. Formally, the next element of `nums[i]` is `nums[(i + 1) % n]` and the previous element of `nums[i]` is `nums[(i - 1 + n) % n]`.
# 
# A **subarray** may only include each element of the fixed buffer `nums` at most once. Formally, for a subarray `nums[i], nums[i + 1], ..., nums[j]`, there does not exist `i <= k1`, `k2 <= j` with `k1 % n == k2 % n`.
# 
# Input: nums = [1,-2,3,-2]
# Output: 3
# Explanation: Subarray [3] has maximum sum 3.
# 
#     Input: nums = [5,-3,5]
# Output: 10
# Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10.
# 
#     Input: nums = [-3,-2,-3]
# Output: -2
# Explanation: Subarray [-2] has maximum sum -2.

# In[3]:


def maxSubarraySumCircular(nums):
    def kadane(arr):
        max_sum = float('-inf')
        curr_sum = 0
        for num in arr:
            curr_sum = max(num, curr_sum + num)
            max_sum = max(max_sum, curr_sum)
        return max_sum

    max_sum_linear = kadane(nums)
    total_sum = sum(nums)
    inverted_nums = [-num for num in nums]
    max_sum_circular = total_sum + kadane(inverted_nums)

    # If all numbers are negative, the maximum sum will be the maximum element
    if max_sum_circular == 0:
        max_sum_circular = max(nums)

    return max(max_sum_linear, max_sum_circular)


# In[4]:


# Test the function
nums = [1, -2, 3, -2]
print(maxSubarraySumCircular(nums))  # Output: 3

nums = [5, -3, 5]
print(maxSubarraySumCircular(nums))  # Output: 10

nums = [-3, -2, -3]
print(maxSubarraySumCircular(nums))  # Output: -2


# <aside>
# 💡 **Question 3**
# 
# The school cafeteria offers circular and square sandwiches at lunch break, referred to by numbers `0` and `1` respectively. All students stand in a queue. Each student either prefers square or circular sandwiches.
# 
# The number of sandwiches in the cafeteria is equal to the number of students. The sandwiches are placed in a **stack**. At each step:
# 
# - If the student at the front of the queue **prefers** the sandwich on the top of the stack, they will **take it** and leave the queue.
# - Otherwise, they will **leave it** and go to the queue's end.
# 
# This continues until none of the queue students want to take the top sandwich and are thus unable to eat.
# 
# You are given two integer arrays `students` and `sandwiches` where `sandwiches[i]` is the type of the `ith` sandwich in the stack (`i = 0` is the top of the stack) and `students[j]` is the preference of the `jth` student in the initial queue (`j = 0` is the front of the queue). Return *the number of students that are unable to eat.*
# 
# 
# Input: students = [1,1,0,0], sandwiches = [0,1,0,1]
# Output: 0
# Explanation:
# - Front student leaves the top sandwich and returns to the end of the line making students = [1,0,0,1].
# - Front student leaves the top sandwich and returns to the end of the line making students = [0,0,1,1].
# - Front student takes the top sandwich and leaves the line making students = [0,1,1] and sandwiches = [1,0,1].
# - Front student leaves the top sandwich and returns to the end of the line making students = [1,1,0].
# - Front student takes the top sandwich and leaves the line making students = [1,0] and sandwiches = [0,1].
# - Front student leaves the top sandwich and returns to the end of the line making students = [0,1].
# - Front student takes the top sandwich and leaves the line making students = [1] and sandwiches = [1].
# - Front student takes the top sandwich and leaves the line making students = [] and sandwiches = [].
# Hence all students are able to eat.
# 
# 
# Input: students = [1,1,1,0,0,1], sandwiches = [1,0,0,0,1,1]
# Output: 3
# 
# 

# In[5]:


def countStudents(students, sandwiches):
    count = 0
    index = 0

    while index < len(students) and sandwiches:
        if students[index] == sandwiches[0]:
            students.pop(index)
            sandwiches.pop(0)
            count = 0  # Reset count to 0 since a student who can eat is encountered
        else:
            students.append(students[index])
            index += 1
            count += 1
        
        if count == len(students):  # All students in the queue are unable to eat
            break
    
    return len(students)  # Return the number of students unable to eat


# In[ ]:


# Test the function
students = [1, 1, 0, 0]
sandwiches = [0, 1, 0, 1]
print(countStudents(students, sandwiches))  # Output: 0


# In[ ]:


students = [1, 1, 1, 0, 0, 1]
sandwiches = [1, 0, 0, 0, 1, 1]
print(countStudents(students, sandwiches))  # Output: 3


# You have a `RecentCounter` class which counts the number of recent requests within a certain time frame.
# 
# Implement the `RecentCounter` class:
# 
# - `RecentCounter()` Initializes the counter with zero recent requests.
# - `int ping(int t)` Adds a new request at time `t`, where `t` represents some time in milliseconds, and returns the number of requests that has happened in the past `3000` milliseconds (including the new request). Specifically, return the number of requests that have happened in the inclusive range `[t - 3000, t]`.
# It is **guaranteed** that every call to `ping` uses a strictly larger value of `t` than the previous call.
# 
# 
# Input
# ["RecentCounter", "ping", "ping", "ping", "ping"]
# [[], [1], [100], [3001], [3002]]
# Output
# [null, 1, 2, 3, 3]
# 
# Explanation
# RecentCounter recentCounter = new RecentCounter();
# recentCounter.ping(1);     // requests = [1], range is [-2999,1], return 1
# recentCounter.ping(100);   // requests = [1,100], range is [-2900,100], return 2
# recentCounter.ping(3001);  // requests = [1,100,3001], range is [1,3001], return 3
# recentCounter.ping(3002);  // requests = [1,100,3001,3002], range is [2,3002], return 3

# In[1]:


from collections import deque

class RecentCounter:
    def __init__(self):
        self.requests = deque()

    def ping(self, t):
        self.requests.append(t)

        while self.requests[0] < t - 3000:
            self.requests.popleft()

        return len(self.requests)


# In[2]:


# Test the RecentCounter class
recentCounter = RecentCounter()
print(recentCounter.ping(1))     # Output: 1
print(recentCounter.ping(100))   # Output: 2
print(recentCounter.ping(3001))  # Output: 3
print(recentCounter.ping(3002))  # Output: 3


# In[ ]:


There are `n` friends that are playing a game. The friends are sitting in a circle and are numbered from `1` to `n` in **clockwise order**. More formally, moving clockwise from the `ith` friend brings you to the `(i+1)th` friend for `1 <= i < n`, and moving clockwise from the `nth` friend brings you to the `1st` friend.

The rules of the game are as follows:

1. **Start** at the `1st` friend.
2. Count the next `k` friends in the clockwise direction **including** the friend you started at. The counting wraps around the circle and may count some friends more than once.
3. The last friend you counted leaves the circle and loses the game.
4. If there is still more than one friend in the circle, go back to step `2` **starting** from the friend **immediately clockwise** of the friend who just lost and repeat.
5. Else, the last friend in the circle wins the game.

Given the number of friends, `n`, and an integer `k`, return *the winner of the game*Input: n = 5, k = 2
Output: 3
Explanation: Here are the steps of the game:
1) Start at friend 1.
2) Count 2 friends clockwise, which are friends 1 and 2.
3) Friend 2 leaves the circle. Next start is friend 3.
4) Count 2 friends clockwise, which are friends 3 and 4.
5) Friend 4 leaves the circle. Next start is friend 5.
6) Count 2 friends clockwise, which are friends 5 and 1.
7) Friend 1 leaves the circle. Next start is friend 3.
8) Count 2 friends clockwise, which are friends 3 and 5.
9) Friend 5 leaves the circle. Only friend 3 is left, so they are the winner.


# In[3]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def findTheWinner(n, k):
    # Create the circular linked list
    head = ListNode(1)
    prev = head
    for i in range(2, n + 1):
        curr = ListNode(i)
        prev.next = curr
        prev = curr
    prev.next = head  # Connect the last node with the first node to form a circular list

    # Simulate the game
    current = head
    while current.next != current:
        # Move k-1 steps clockwise
        for _ in range(k - 1):
            current = current.next

        # Remove the next friend
        current.next = current.next.next

        # Move to the friend after the removed friend
        current = current.next

    return current.val


# In[4]:


# Test the function
print(findTheWinner(5, 2))  # Output: 3


# In[5]:


print(findTheWinner(6, 5))  # Output: 1


# You are given an integer array `deck`. There is a deck of cards where every card has a unique integer. The integer on the `ith` card is `deck[i]`.
# 
# You can order the deck in any order you want. Initially, all the cards start face down (unrevealed) in one deck.
# 
# You will do the following steps repeatedly until all cards are revealed:
# 
# 1. Take the top card of the deck, reveal it, and take it out of the deck.
# 2. If there are still cards in the deck then put the next top card of the deck at the bottom of the deck.
# 3. If there are still unrevealed cards, go back to step 1. Otherwise, stop.
# 
# Return *an ordering of the deck that would reveal the cards in increasing order*.
# 
# Input: deck = [17,13,11,2,3,5,7]
# Output: [2,13,3,11,5,17,7]
# Explanation:
# We get the deck in the order [17,13,11,2,3,5,7] (this order does not matter), and reorder it.
# After reordering, the deck starts as [2,13,3,11,5,17,7], where 2 is the top of the deck.
# We reveal 2, and move 13 to the bottom.  The deck is now [3,11,5,17,7,13].
# We reveal 3, and move 11 to the bottom.  The deck is now [5,17,7,13,11].
# We reveal 5, and move 17 to the bottom.  The deck is now [7,13,11,17].
# We reveal 7, and move 13 to the bottom.  The deck is now [11,17,13].
# We reveal 11, and move 17 to the bottom.  The deck is now [13,17].
# We reveal 13, and move 17 to the bottom.  The deck is now [17].
# We reveal 17.
# Since all the cards revealed are in increasing order, the answer is correct.
# 
# 
# Input: deck = [1,1000]
# Output: [1,1000]

# In[6]:


from collections import deque

def deckRevealedIncreasing(deck):
    n = len(deck)
    deck.sort()  # Sort the deck in ascending order

    queue = deque(deck)  # Create a queue with the sorted deck
    result = []  # List to store the order of revealed cards

    while queue:
        result.append(queue.popleft())  # Take the top card from the queue
        if queue:  # If the queue is not empty
            queue.append(queue.popleft())  # Move the next top card to the bottom of the queue

    return result


# In[7]:


# Test the function
print(deckRevealedIncreasing([17, 13, 11, 2, 3, 5, 7]))  # Output: [2, 13, 3, 11, 5, 17, 7]
print(deckRevealedIncreasing([1, 1000]))  # Output: [1, 1000]


# Design a queue that supports `push` and `pop` operations in the front, middle, and back.
# 
# Implement the `FrontMiddleBack` class:
# 
# - `FrontMiddleBack()` Initializes the queue.
# - `void pushFront(int val)` Adds `val` to the **front** of the queue.
# - `void pushMiddle(int val)` Adds `val` to the **middle** of the queue.
# - `void pushBack(int val)` Adds `val` to the **back** of the queue.
# - `int popFront()` Removes the **front** element of the queue and returns it. If the queue is empty, return `1`.
# - `int popMiddle()` Removes the **middle** element of the queue and returns it. If the queue is empty, return `1`.
# - `int popBack()` Removes the **back** element of the queue and returns it. If the queue is empty, return `1`.
# 
# **Notice** that when there are **two** middle position choices, the operation is performed on the **frontmost** middle position choice. For example:
# 
# - Pushing `6` into the middle of `[1, 2, 3, 4, 5]` results in `[1, 2, 6, 3, 4, 5]`.
# - Popping the middle from `[1, 2, 3, 4, 5, 6]` returns `3` and results in `[1, 2, 4, 5, 6]`.
# 
# Input:
# ["FrontMiddleBackQueue", "pushFront", "pushBack", "pushMiddle", "pushMiddle", "popFront", "popMiddle", "popMiddle", "popBack", "popFront"]
# [[], [1], [2], [3], [4], [], [], [], [], []]
# Output:
# [null, null, null, null, null, 1, 3, 4, 2, -1]
# 
# Explanation:
# FrontMiddleBackQueue q = new FrontMiddleBackQueue();
# q.pushFront(1);   // [1]
# q.pushBack(2);    // [1,2]
# q.pushMiddle(3);  // [1,3, 2]
# q.pushMiddle(4);  // [1,4, 3, 2]
# q.popFront();     // return 1 -> [4, 3, 2]
# q.popMiddle();    // return 3 -> [4, 2]
# q.popMiddle();    // return 4 -> [2]
# q.popBack();      // return 2 -> []
# q.popFront();     // return -1 -> [] (The queue is empty)

# In[ ]:


class Node:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None


class FrontMiddleBackQueue:
    def __init__(self):
        self.front = None
        self.middle = None
        self.back = None

    def pushFront(self, val):
        new_node = Node(val)
        if self.front is None:
            self.front = new_node
            self.back = new_node
        else:
            new_node.next = self.front
            self.front.prev = new_node
            self.front = new_node
        self.updateMiddle()

    def pushMiddle(self, val):
        new_node = Node(val)
        if self.front is None:
            self.front = new_node
            self.back = new_node
        elif self.middle is None:
            self.middle = new_node
            new_node.prev = self.front
            self.front.next = new_node
            self.updateMiddle()
        else:
            prev_middle = self.middle
            next_middle = self.middle.next
            new_node.prev = prev


# For a stream of integers, implement a data structure that checks if the last `k` integers parsed in the stream are **equal** to `value`.
# 
# Implement the **DataStream** class:
# 
# - `DataStream(int value, int k)` Initializes the object with an empty integer stream and the two integers `value` and `k`.
# - `boolean consec(int num)` Adds `num` to the stream of integers. Returns `true` if the last `k` integers are equal to `value`, and `false` otherwise. If there are less than `k` integers, the condition does not hold true, so returns `false`.
# Input
# ["DataStream", "consec", "consec", "consec", "consec"]
# [[4, 3], [4], [4], [4], [3]]
# Output
# [null, false, false, true, false]
# 
# Explanation
# DataStream dataStream = new DataStream(4, 3); //value = 4, k = 3
# dataStream.consec(4); // Only 1 integer is parsed, so returns False.
# dataStream.consec(4); // Only 2 integers are parsed.
#                       // Since 2 is less than k, returns False.
# dataStream.consec(4); // The 3 integers parsed are all equal to value, so returns True.
# dataStream.consec(3); // The last k integers parsed in the stream are [4,4,3].
#                       // Since 3 is not equal to value, it returns False.

# In[8]:


from collections import deque

class DataStream:
    def __init__(self, value, k):
        self.value = value
        self.k = k
        self.queue = deque()

    def consec(self, num):
        self.queue.append(num)
        if len(self.queue) < self.k:
            return False
        if len(self.queue) > self.k:
            self.queue.popleft()
        return all(x == self.value for x in self.queue)


# In[9]:


dataStream = DataStream(4, 3)
print(dataStream.consec(4))  # Output: False
print(dataStream.consec(4))  # Output: False
print(dataStream.consec(4))  # Output: True
print(dataStream.consec(3))  # Output: False


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# Given an array, for each element find the value of the nearest element to the right which is having a frequency greater than that of the current element. If there does not exist an answer for a position, then make the value ‘-1’.                                                         
#  Input: a[] = [1, 1, 2, 3, 4, 2, 1]
# Output : [-1, -1, 1, 2, 2, 1, -1]
# 
# Explanation:
# Given array a[] = [1, 1, 2, 3, 4, 2, 1]
# Frequency of each element is: 3, 3, 2, 1, 1, 2, 3
# 
# Lets calls Next Greater Frequency element as NGF
# 1. For element a[0] = 1 which has a frequency = 3,
#    As it has frequency of 3 and no other next element
#    has frequency more than 3 so  '-1'
# 2. For element a[1] = 1 it will be -1 same logic
#    like a[0]
# 3. For element a[2] = 2 which has frequency = 2,
#    NGF element is 1 at position = 6  with frequency
#    of 3 > 2
# 4. For element a[3] = 3 which has frequency = 1,
#    NGF element is 2 at position = 5 with frequency
#    of 2 > 1
# 5. For element a[4] = 4 which has frequency = 1,
#    NGF element is 2 at position = 5 with frequency
#    of 2 > 1
# 6. For element a[5] = 2 which has frequency = 2,
#    NGF element is 1 at position = 6 with frequency
#    of 3 > 2
# 7. For element a[6] = 1 there is no element to its
#    right, hence -1
# 
# Input : a[] = [1, 1, 1, 2, 2, 2, 2, 11, 3, 3]
# Output : [2, 2, 2, -1, -1, -1, -1, 3, -1, -1]

# In[1]:


def find_nearest_greater_frequency(arr):
    frequency = {}
    stack = []
    result = [-1] * len(arr)

    # Step 1: Count the frequency of each element
    for num in arr:
        frequency[num] = frequency.get(num, 0) + 1

    # Step 2: Iterate through the array from right to left
    for i in range(len(arr) - 1, -1, -1):
        while stack and frequency[arr[i]] >= frequency[arr[stack[-1]]]:
            stack.pop()

        if stack:
            result[i] = arr[stack[-1]]

        stack.append(i)

    return result


# In[17]:


arr = [1, 1, 1, 2, 2, 2, 2, 11, 3, 3]
output = find_nearest_greater_frequency(arr)
print(output)


# Given a stack of integers, sort it in ascending order using another temporary stack.
# 
# 
# Input : [34, 3, 31, 98, 92, 23]
# Output : [3, 23, 31, 34, 92, 98]
# 
# Input : [3, 5, 1, 4, 2, 8]
# Output : [1, 2, 3, 4, 5, 8]

# In[16]:


def sort_stack(stack):
    temp_stack = []

    while stack:
        temp = stack.pop()

        while temp_stack and temp_stack[-1] > temp:
            stack.append(temp_stack.pop())

        temp_stack.append(temp)

    while temp_stack:
        stack.append(temp_stack.pop())

    return stack


# In[14]:


stack1 = [34, 3, 31, 98, 92, 23]
output1 = sort_stack(stack1)
print(output1)  # [3, 23, 31, 34, 92, 98]


# In[6]:


stack2 = [3, 5, 1, 4, 2, 8]
output2 = sort_stack(stack2)
print(output2)  # [1, 2, 3, 4, 5, 8]


# In[ ]:





# Given a stack with **push()**, **pop()**, and **empty()** operations, The task is to delete the **middle** element ****of it without using any additional data structure.
# 
# Input  : Stack[] = [1, 2, 3, 4, 5]
# 
# Output : Stack[] = [1, 2, 4, 5]
# 
# Input  : Stack[] = [1, 2, 3, 4, 5, 6]
# 
# Output : Stack[] = [1, 2, 4, 5, 6]

# In[25]:


def delete_middle(stack, index):
    if stack:
        if index == len(stack) // 2:
            stack.pop()
        else:
            temp = stack.pop()
            delete_middle(stack, index + 1)
            stack.append(temp)

def delete_middle_element(stack):
    delete_middle(stack, 0)


# In[26]:


# Testing the function
stack1 = [1, 2, 3, 4, 5]
delete_middle_element(stack1)
print(stack1)  # [1, 2, 4, 5]

stack2 = [1, 2, 3, 4, 5, 6]
delete_middle_element(stack2)
print(stack2)  # [1, 2, 4, 5, 6]


# In[ ]:





# Given a Queue consisting of first **n** natural numbers (in random order). The task is to check whether the given Queue elements can be arranged in increasing order in another Queue using a stack. The operation allowed are:
# 
# 1. Push and pop elements from the stack
# 2. Pop (Or Dequeue) from the given Queue.
# 3. Push (Or Enqueue) in the another Queue.
# 
# **Examples :**
# 
# Input : Queue[] = { 5, 1, 2, 3, 4 } 
# 
# Output : Yes 
# 
# Pop the first element of the given Queue 
# 
# i.e 5. Push 5 into the stack. 
# 
# Now, pop all the elements of the given Queue and push them to second Queue. 
# 
# Now, pop element 5 in the stack and push it to the second Queue.   
# 
# Input : Queue[] = { 5, 1, 2, 6, 3, 4 } 
# 
# Output : No 
# 
# Push 5 to stack. 
# 
# Pop 1, 2 from given Queue and push it to another Queue. 
# 
# Pop 6 from given Queue and push to stack. 
# 
# Pop 3, 4 from given Queue and push to second Queue. 
# 
# Now, from using any of above operation, we cannot push 5 into the second Queue because it is below the 6 in the stack.

# In[27]:


from queue import Queue

def check_queue_order(queue):
    stack = []
    second_queue = Queue()

    while not queue.empty():
        front_element = queue.queue[0]
        queue.get()

        if stack and stack[-1] > front_element:
            return "No"

        second_queue.put(front_element)

    while stack:
        popped_element = stack.pop()

        if not second_queue.empty() and popped_element > second_queue.queue[0]:
            return "No"

        second_queue.put(popped_element)

    # Check if the second queue is in increasing order
    for i in range(1, second_queue.qsize()):
        if second_queue.queue[i] < second_queue.queue[i - 1]:
            return "No"

    return "Yes"


# In[28]:


queue1 = Queue()
queue1.put(5)
queue1.put(1)
queue1.put(2)
queue1.put(3)
queue1.put(4)
output1 = check_queue_order(queue1)
print(output1)  # Yes

queue2 = Queue()
queue2.put(5)
queue2.put(1)
queue2.put(2)
queue2.put(6)
queue2.put(3)
queue2.put(4)
output2 = check_queue_order(queue2)
print(output2)  # No


# In[ ]:





# Given a number , write a program to reverse this number using stack.
# 
# Input : 365
# Output : 563
# 
# Input : 6899
# Output : 9986

# In[29]:


def reverse_number(num):
    num_str = str(num)
    stack = []

    # Push each character into the stack
    for char in num_str:
        stack.append(char)

    reversed_str = ""

    # Pop each character from the stack and append it to the string
    while stack:
        reversed_str += stack.pop()

    # Convert the string back to an integer
    reversed_num = int(reversed_str)

    return reversed_num


# In[30]:


num1 = 365
reversed_num1 = reverse_number(num1)
print(reversed_num1)  # 563

num2 = 6899
reversed_num2 = reverse_number(num2)
print(reversed_num2)  # 9986


# Given an integer k and a **[queue](https://www.geeksforgeeks.org/queue-data-structure/)** of integers, The task is to reverse the order of the first **k** elements of the queue, leaving the other elements in the same relative order.
# 
# Only following standard operations are allowed on queue.
# 
# - **enqueue(x) :** Add an item x to rear of queue
# - **dequeue() :** Remove an item from front of queue
# - **size() :** Returns number of elements in queue.
# - **front() :** Finds front item.

# In[31]:


from queue import Queue

def reverse_k_elements(queue, k):
    if k == 0 or queue.empty():
        return

    temp = queue.get()
    reverse_k_elements(queue, k - 1)
    queue.put(temp)

def reverse_first_k_elements(queue, k):
    if k <= 0 or k > queue.qsize():
        return "Invalid value of k"

    reverse_k_elements(queue, k)

    for _ in range(queue.qsize() - k):
        queue.put(queue.get())

    return queue


# In[32]:


# Testing the function
queue = Queue()
queue.put(1)
queue.put(2)
queue.put(3)
queue.put(4)
queue.put(5)

k = 3
result = reverse_first_k_elements(queue, k)
while not result.empty():
    print(result.get(), end=" ")  # Output: 3 2 1 4 5


# Given a sequence of n strings, the task is to check if any two similar words come together and then destroy each other then print the number of words left in the sequence after this pairwise destruction.
# 
# **Examples:**
# 
# Input : ab aa aa bcd ab
# 
# Output : 3
# 
# *As aa, aa destroys each other so,*
# 
# *ab bcd ab is the new sequence.*
# 
# Input :  tom jerry jerry tom
# 
# Output : 0
# 
# *As first both jerry will destroy each other.*
# 
# *Then sequence will be tom, tom they will also destroy*
# 
# *each other. So, the final sequence doesn’t contain any*
# 
# *word.*

# In[33]:


def count_words_after_destruction(sequence):
    stack = []

    for word in sequence:
        if stack and word == stack[-1]:
            stack.pop()
        else:
            stack.append(word)

    return len(stack)


# In[34]:


# Testing the function
sequence1 = ["ab", "aa", "aa", "bcd", "ab"]
result1 = count_words_after_destruction(sequence1)
print(result1)  # 3

sequence2 = ["tom", "jerry", "jerry", "tom"]
result2 = count_words_after_destruction(sequence2)
print(result2)  # 0


# Given an array of integers, the task is to find the maximum absolute difference between the nearest left and the right smaller element of every element in the array.
# 
# **Note:** If there is no smaller element on right side or left side of any element then we take zero as the smaller element. For example for the leftmost element, the nearest smaller element on the left side is considered as 0. Similarly, for rightmost elements, the smaller element on the right side is considered as 0.
# Input : arr[] = {2, 1, 8}
# Output : 1
# Left smaller  LS[] {0, 0, 1}
# Right smaller RS[] {1, 0, 0}
# Maximum Diff of abs(LS[i] - RS[i]) = 1
# 
# Input  : arr[] = {2, 4, 8, 7, 7, 9, 3}
# Output : 4
# Left smaller   LS[] = {0, 2, 4, 4, 4, 7, 2}
# Right smaller  RS[] = {0, 3, 7, 3, 3, 3, 0}
# Maximum Diff of abs(LS[i] - RS[i]) = 7 - 3 = 4
# 
# Input : arr[] = {5, 1, 9, 2, 5, 1, 7}
# Output : 1

# In[37]:


def max_absolute_difference(arr):
    n = len(arr)
    LS = [0] * n
    RS = [0] * n

    LS[0] = 0
    RS[n - 1] = 0

    for i in range(1, n):
        j = i - 1
        while j >= 0 and arr[j] >= arr[i]:
            j = LS[j]
        LS[i] = j

    for i in range(n - 2, -1, -1):
        j = i + 1
        while j < n and arr[j] >= arr[i]:
            j = RS[j]
        RS[i] = j

    max_diff = 0
    for i in range(n):
        max_diff = max(max_diff, abs(LS[i] - RS[i]))

    return max_diff


# In[ ]:


# Testing the function
arr1 = [2, 1, 8]
result1 = max_absolute_difference(arr1)
print(result1)  # 1

arr2 = [2, 4, 8, 7, 7, 9, 3]
result2 = max_absolute_difference(arr2)
print(result2)  # 4

arr3 = [5, 1, 9, 2, 5, 1, 7]
result3 = max_absolute_difference(arr3)
print(result3)  # 1


# In[ ]:





# In[ ]:





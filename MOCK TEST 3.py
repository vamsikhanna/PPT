#!/usr/bin/env python
# coding: utf-8

# Implement a stack using a list in Python. Include the necessary methods such as push, pop, and isEmpty.

# In[1]:


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.isEmpty():
            return self.stack.pop()
        else:
            print("Stack is empty.")
            return None

    def isEmpty(self):
        return len(self.stack) == 0


# In[2]:


stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)


# In[3]:


print(stack.pop())  # Output: 3
print(stack.pop())  # Output: 2
print(stack.isEmpty())  # Output: False
print(stack.pop())  # Output: 1
print(stack.isEmpty())  # Output: True
print(stack.pop())  # Output: Stack is empty. None


# Implement a queue using a list in Python. Include the necessary methods such as enqueue, dequeue, and isEmpty.

# In[5]:


class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.isEmpty():
            return self.queue.pop(0)
        else:
            print("Queue is empty.")
            return None

    def isEmpty(self):
        return len(self.queue) == 0


# In[6]:


queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)


# In[7]:


print(queue.dequeue())  # Output: 1
print(queue.dequeue())  # Output: 2
print(queue.isEmpty())  # Output: False
print(queue.dequeue())  # Output: 3
print(queue.isEmpty())  # Output: True
print(queue.dequeue())  # Output: Queue is empty. None


# In[ ]:





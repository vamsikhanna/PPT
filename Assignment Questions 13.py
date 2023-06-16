#!/usr/bin/env python
# coding: utf-8

# Given two linked list of the same size, the task is to create a new linked list using those linked lists. The condition is that the greater node among both linked list will be added to the new linked list.
# 
# Input: list1 = 5->2->3->8
# list2 = 1->7->4->5
# Output: New list = 5->7->4->8
# 
# Input:list1 = 2->8->9->3
# list2 = 5->3->6->4
# Output: New list = 5->8->9->4

# In[1]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeLists(list1, list2):
    head1 = list1
    head2 = list2
    newHead = None
    tail = None

    while head1 and head2:
        if head1.val >= head2.val:
            newNode = ListNode(head1.val)
            head1 = head1.next
        else:
            newNode = ListNode(head2.val)
            head2 = head2.next

        if not newHead:
            newHead = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = tail.next

    # Add remaining nodes from list1
    while head1:
        newNode = ListNode(head1.val)
        tail.next = newNode
        tail = tail.next
        head1 = head1.next

    # Add remaining nodes from list2
    while head2:
        newNode = ListNode(head2.val)
        tail.next = newNode
        tail = tail.next
        head2 = head2.next

    return newHead


# In[3]:


# Example 1
list1 = ListNode(5)
list1.next = ListNode(2)
list1.next.next = ListNode(3)
list1.next.next.next = ListNode(8)

list2 = ListNode(1)
list2.next = ListNode(7)
list2.next.next = ListNode(4)
list2.next.next.next = ListNode(5)

newList = mergeLists(list1, list2)
# Output: New list = 5->7->4->8
newList


# In[6]:


# Example 2
list3 = ListNode(2)
list3.next = ListNode(8)
list3.next.next = ListNode(9)
list3.next.next.next = ListNode(3)

list4 = ListNode(5)
list4.next = ListNode(3)
list4.next.next = ListNode(6)
list4.next.next.next = ListNode(4)

newList = mergeLists(list3, list4)
# Output: New list = 5->8->9->4


# Write a function that takes a list sorted in non-decreasing order and deletes any duplicate nodes from the list. The list should only be traversed once.
# 
# For example if the linked list is 11->11->11->21->43->43->60 then removeDuplicates() should convert the list to 11->21->43->60.
# 
# 
# Input:
# LinkedList: 
# 11->11->11->21->43->43->60
# Output:
# 11->21->43->60
# 
# 
# Input:
# LinkedList: 
# 10->12->12->25->25->25->34
# Output:
# 10->12->25->34

# In[9]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def removeDuplicates(head):
    if head is None:
        return head

    current = head
    while current.next is not None:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next

    return head


# In[11]:


# Example 1
list1 = ListNode(11)
list1.next = ListNode(11)
list1.next.next = ListNode(11)
list1.next.next.next = ListNode(21)
list1.next.next.next.next = ListNode(43)
list1.next.next.next.next.next = ListNode(43)
list1.next.next.next.next.next.next = ListNode(60)

newList = removeDuplicates(list1)
# Output: 11->21->43->60

newList


# In[7]:


# Example 2
list2 = ListNode(10)
list2.next = ListNode(12)
list2.next.next = ListNode(12)
list2.next.next.next = ListNode(25)
list2.next.next.next.next = ListNode(25)
list2.next.next.next.next.next = ListNode(25)
list2.next.next.next.next.next.next = ListNode(34)

newList = removeDuplicates(list2)
# Output: 10->12->25->34


# Given a linked list of size N. The task is to reverse every k nodes (where k is an input to the function) in the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should be considered as a group and must be reversed (See Example 2 for clarification).
# 
# Input:
# LinkedList: 1->2->2->4->5->6->7->8
# K = 4
# Output:4 2 2 1 8 7 6 5
# Explanation:
# The first 4 elements 1,2,2,4 are reversed first
# and then the next 4 elements 5,6,7,8. Hence, the
# resultant linked list is 4->2->2->1->8->7->6->5.
# 
# 
# Input:
# LinkedList: 1->2->3->4->5
# K = 3
# Output:3 2 1 5 4
# Explanation:
# The first 3 elements are 1,2,3 are reversed
# first and then elements 4,5 are reversed.Hence,
# the resultant linked list is 3->2->1->5->4.
# 
# 

# In[ ]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseKNodes(head, k):
    if head is None or k == 1:
        return head

    count = 0
    curr = head
    prev = None
    while curr is not None and count < k:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
        count += 1

    if curr is not None:
        head.next = reverseKNodes(curr, k)

    return prev

# Example
list1 = ListNode(1)
list1.next = ListNode(2)
list1.next.next = ListNode(2)
list1.next.next.next = ListNode(4)
list1.next.next.next.next = ListNode(5)
list1.next.next.next.next.next = ListNode(6)
list1.next.next.next.next.next.next = ListNode(7)
list1.next.next.next.next.next.next.next = ListNode(8)

newList = reverseKNodes(list1, 4)
# Output: 4->2->2->1->8->7->6->5

list2 = ListNode(1)
list2.next = ListNode(2)
list2.next.next = ListNode(3)
list2.next.next.next = ListNode(4)
list2.next.next.next.next = ListNode(5)

newList = reverseKNodes(list2, 3)
# Output: 3->2->1->5->4


# Given a linked list, write a function to reverse every alternate k nodes (where k is an input to the function) in an efficient way. Give the complexity of your algorithm.
# 
# Inputs:   1->2->3->4->5->6->7->8->9->NULL and k = 3
# Output:   3->2->1->4->5->6->9->8->7->NULL.

# In[ ]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseAlternateKNodes(head, k):
    if head is None or k == 1:
        return head

    count = 0
    curr = head
    prev = None
    next_node = None

    # Reverse k nodes
    while curr is not None and count < k:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
        count += 1

    # Recursive call on remaining nodes
    if next_node is not None:
        head.next = next_node

        # Skip k nodes for the alternate reversal
        for _ in range(k - 1):
            if next_node is None:
                break
            next_node = next_node.next

        if next_node is not None:
            next_node.next = reverseAlternateKNodes(next_node.next, k)

    return prev

# Example
list1 = ListNode(1)
list1.next = ListNode(2)
list1.next.next = ListNode(3)
list1.next.next.next = ListNode(4)
list1.next.next.next.next = ListNode(5)
list1.next.next.next.next.next = ListNode(6)
list1.next.next.next.next.next.next = ListNode(7)
list1.next.next.next.next.next.next.next = ListNode(8)
list1.next.next.next.next.next.next.next.next = ListNode(9)

newList = reverseAlternateKNodes(list1, 3)
# Output: 3->2->1->4->5->6->9->8->7


# Given a linked list and a key to be deleted. Delete last occurrence of key from linked. The list may have duplicates.
# 
# Input:   1->2->3->5->2->10, key = 2
# Output:  1->2->3->5->10

# In[12]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteLastOccurrence(head, key):
    if head is None:
        return head

    curr = head
    last_occurrence = None
    prev_last_occurrence = None

    # Find the last occurrence of the key
    while curr is not None:
        if curr.val == key:
            last_occurrence = curr
            prev_last_occurrence = None
        if curr.next is not None and curr.next.val == key:
            prev_last_occurrence = curr
        curr = curr.next

    # If the key is not found
    if last_occurrence is None:
        return head

    # If the last occurrence is the head node
    if prev_last_occurrence is None:
        head = head.next
    else:
        prev_last_occurrence.next = last_occurrence.next

    return head

# Example
list1 = ListNode(1)
list1.next = ListNode(2)
list1.next.next = ListNode(3)
list1.next.next.next = ListNode(5)
list1.next.next.next.next = ListNode(2)
list1.next.next.next.next.next = ListNode(10)

newList = deleteLastOccurrence(list1, 2)
# Output: 1->2->3->5->10


# Given two sorted linked lists consisting of N and M nodes respectively. The task is to merge both of the lists (in place) and return the head of the merged list.
# 
# Input: a: 5->10->15, b: 2->3->20
# 
# Output: 2->3->5->10->15->20
# 
# Input: a: 1->1, b: 2->4
# 
# Output: 1->1->2->4

# In[ ]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeLists(a, b):
    if a is None:
        return b
    if b is None:
        return a

    if a.val <= b.val:
        a.next = mergeLists(a.next, b)
        return a
    else:
        b.next = mergeLists(a, b.next)
        return b

# Example
list1 = ListNode(5)
list1.next = ListNode(10)
list1.next.next = ListNode(15)

list2 = ListNode(2)
list2.next = ListNode(3)
list2.next.next = ListNode(20)

mergedList = mergeLists(list1, list2)
# Output: 2->3->5->10->15->20


# Given a Doubly Linked List, the task is to reverse the given Doubly Linked List.
# Original Linked list 10 8 4 2
# Reversed Linked list 2 4 8 10

# In[13]:


class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

def reverseDLL(head):
    if head is None or head.next is None:
        return head

    current = head
    prev = None

    while current:
        next_node = current.next
        current.next = prev
        current.prev = next_node
        prev = current
        current = next_node

    head = prev
    return head

# Example
head = Node(10)
head.next = Node(8)
head.next.next = Node(4)
head.next.next.next = Node(2)

reversed_head = reverseDLL(head)

# Output: 2->4->8->10


# Given a doubly linked list and a position. The task is to delete a node from given position in a doubly linked list.
# 
# Input:
# LinkedList = 1 <--> 3 <--> 4
# x = 3
# Output:1 3
# Explanation:After deleting the node at
# position 3 (position starts from 1),
# the linked list will be now as 1->3.
# 
# Input:
# LinkedList = 1 <--> 5 <--> 2 <--> 9
# x = 1
# Output:5 2 9

# In[ ]:


class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

def deleteNode(head, position):
    if head is None:
        return head

    if position == 1:
        new_head = head.next
        if new_head:
            new_head.prev = None
        return new_head

    current = head
    count = 1

    while current and count < position:
        current = current.next
        count += 1

    if current:
        current.prev.next = current.next
        if current.next:
            current.next.prev = current.prev

    return head

# Example
head = Node(1)
head.next = Node(5)
head.next.next = Node(2)
head.next.next.next = Node(9)

new_head = deleteNode(head, 1)

# Output: 5 2 9


# In[ ]:





# In[ ]:





# In[ ]:





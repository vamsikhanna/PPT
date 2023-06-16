#!/usr/bin/env python
# coding: utf-8

# Given a linked list of **N** nodes such that it may contain a loop.
# 
# A loop here means that the last node of the link list is connected to the node at position X(1-based index). If the link list does not have any loop, X=0.
# 
# Remove the loop from the linked list, if it is present, i.e. unlink the last node which is forming the loop.
# 
# Input:
# N = 3
# value[] = {1,3,4}
# X = 2
# Output:1
# Explanation:The link list looks like
# 1 -> 3 -> 4
#      ^    |
#      |____|
# A loop is present. If you remove it
# successfully, the answer will be 1.
# 
# 
# Input:
# N = 4
# value[] = {1,8,3,4}
# X = 0
# Output:1
# Explanation:The Linked list does not
# contains any loop.
# 
# 
# Input:
# N = 4
# value[] = {1,2,3,4}
# X = 1
# Output:1
# Explanation:The link list looks like
# 1 -> 2 -> 3 -> 4
# ^              |
# |______________|
# A loop is present.
# If you remove it successfully,
# the answer will be 1.
# 
# 

# In[1]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def detectAndRemoveLoop(head):
    if not head or not head.next:
        return head

    slow = head
    fast = head

    # Find the meeting point of slow and fast pointers
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            break

    # If no loop is found
    if slow != fast:
        return head

    # Reset slow or fast pointer to head
    slow = head

    # Move both pointers one step at a time until they meet again
    while slow.next != fast.next:
        slow = slow.next
        fast = fast.next

    # Set the next pointer of the node before the meeting point to null
    fast.next = None

    return head


# In[2]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(3)
head1.next.next = ListNode(4)
head1.next.next.next = head1.next

result1 = detectAndRemoveLoop(head1)
print("Linked List after removing the loop:")
curr = result1
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")
# Output: 1 -> 3 -> 4 -> None


# In[3]:


# Example 2
head2 = ListNode(1)
head2.next = ListNode(8)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(4)

result2 = detectAndRemoveLoop(head2)
print("\nLinked List after removing the loop:")
curr = result2
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")
# Output: 1 -> 8 -> 3 -> 4 -> None


# In[4]:


# Example 3
head3 = ListNode(1)
head3.next = ListNode(2)
head3.next.next = ListNode(3)
head3.next.next.next = ListNode(4)
head3.next.next.next.next = head3

result3 = detectAndRemoveLoop(head3)
print("\nLinked List after removing the loop:")
curr = result3
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")
# Output: 1 -> 2 -> 3 -> 4 -> None


# A number N is represented in Linked List such that each digit corresponds to a node in linked list. You need to add 1 to it.
# 
# Input:
# LinkedList: 4->5->6
# Output:457
# 
# Input:
# LinkedList: 1->2->3
# Output:124

# In[5]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def addOne(head):
    # Reverse the linked list
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    head = prev

    carry = 1
    current = head
    prev = None

    # Traverse the linked list and add 1
    while current:
        sum = current.data + carry
        carry = sum // 10
        current.data = sum % 10
        prev = current
        current = current.next

    # If there is still a carry remaining, create a new node
    if carry > 0:
        new_node = Node(carry)
        prev.next = new_node

    # Reverse the linked list again to restore the original order
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    head = prev

    return head


# In[6]:


# Example 1
head1 = Node(4)
head1.next = Node(5)
head1.next.next = Node(6)

result1 = addOne(head1)
while result1:
    print(result1.data, end="")
    result1 = result1.next

# Output: 457


# In[7]:


# Example 2
head2 = Node(1)
head2.next = Node(2)
head2.next.next = Node(3)

result2 = addOne(head2)
while result2:
    print(result2.data, end="")
    result2 = result2.next

# Output: 124


# Given a Linked List of size N, where every node represents a sub-linked-list and contains two pointers:(i) a next pointer to the next node,(ii) a bottom pointer to a linked list where this node is head.Each of the sub-linked-list is in sorted order.Flatten the Link List such that all the nodes appear in a single level while maintaining the sorted order. Note: The flattened list will be printed using the bottom pointer instead of next pointer.
# 
# 
# Input:
# 5 -> 10 -> 19 -> 28
# |     |     |     |
# 7     20    22   35
# |           |     |
# 8          50    40
# |                 |
# 30               45
# Output: 5-> 7-> 8- > 10 -> 19-> 20->
# 22-> 28-> 30-> 35-> 40-> 45-> 50.
# Explanation:
# The resultant linked lists has every
# node in a single level.(Note:| represents the bottom pointer.)
# 
# 
# Input:
# 5 -> 10 -> 19 -> 28
# |          |
# 7          22
# |          |
# 8          50
# |
# 30
# Output: 5->7->8->10->19->22->28->30->50
# Explanation:
# The resultant linked lists has every
# node in a single level.
# 
# (Note:| represents the bottom pointer.)

# In[8]:


class ListNode:
    def __init__(self, val=0, next=None, bottom=None):
        self.val = val
        self.next = next
        self.bottom = bottom

def merge(a, b):
    dummy = ListNode()
    tail = dummy

    while a and b:
        if a.val <= b.val:
            tail.bottom = a
            a = a.bottom
        else:
            tail.bottom = b
            b = b.bottom
        tail = tail.bottom

    if a:
        tail.bottom = a
    else:
        tail.bottom = b

    return dummy.bottom

def flattenLinkedList(head):
    if not head or not head.next:
        return head

    head.next = flattenLinkedList(head.next)

    head = merge(head, head.next)

    return head


# In[9]:


# Example 1
head1 = ListNode(5)
head1.next = ListNode(10)
head1.next.next = ListNode(19)
head1.next.next.next = ListNode(28)

head1.bottom = ListNode(7)
head1.bottom.bottom = ListNode(8)
head1.bottom.bottom.bottom = ListNode(30)

head1.next.bottom = ListNode(20)
head1.next.next.bottom = ListNode(22)
head1.next.next.next.bottom = ListNode(35)
head1.next.next.next.bottom.bottom = ListNode(50)
head1.next.next.next.bottom.bottom.bottom = ListNode(45)

result1 = flattenLinkedList(head1)
print("Flattened Linked List:")
curr = result1
while curr:
    print(curr.val, end=" -> ")
    curr = curr.bottom
print("None")
# Output: 5 -> 7 -> 8 -> 10 -> 19 -> 20 -> 22 -> 28 -> 30 -> 35 -> 40 -> 45 -> 50 -> None


# In[10]:


# Example 2
head2 = ListNode(5)
head2.next = ListNode(10)
head2.next.next = ListNode(19)
head2.next.next.next = ListNode(28)

head2.bottom = ListNode(7)
head2.bottom.bottom = ListNode(22)
head2.bottom.bottom.bottom = ListNode(30)

head2.next.bottom = ListNode(8)
head2.next.bottom.bottom = ListNode(50)

result2 = flattenLinkedList(head2)
print("\nFlattened Linked List:")
curr = result2
while curr:
    print(curr.val, end=" -> ")
    curr = curr.bottom
print("None")
# Output: 5 -> 7 -> 8 -> 10 -> 19 -> 22 -> 28 -> 30 -> 50 -> None


# You are given a special linked list with **N** nodes where each node has a next pointer pointing to its next node. You are also given **M** random pointers, where you will be given **M** number of pairs denoting two nodes **a** and **b**  **i.e. a->arb = b** (arb is pointer to random node)**.**
# 
# Construct a copy of the given list. The copy should consist of exactly **N** new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.
# 
# For example, if there are two nodes **X** and **Y** in the original list, where **X.arb** **-->** **Y**, then for the corresponding two nodes **x** and **y** in the copied list, **x.arb --> y.**
# 
# Return the head of the copied linked list.
# 
# 
# Input:
# N = 4, M = 2
# value = {1,2,3,4}
# pairs = {{1,2},{2,4}}
# Output:1
# Explanation:In this test case, there
# are 4 nodes in linked list.  Among these
# 4 nodes,  2 nodes have arbitrary pointer
# set, rest two nodes have arbitrary pointer
# as NULL. Second line tells us the value
# of four nodes. The third line gives the
# information about arbitrary pointers.
# The first node arbitrary pointer is set to
# node 2.  The second node arbitrary pointer
# is set to node 4.
# 
# 
# Input:
# N = 4, M = 2
# value[] = {1,3,5,9}
# pairs[] = {{1,1},{3,4}}
# Output:1
# Explanation:In the given testcase ,
# applying the method as stated in the
# above example, the output will be 1.

# In[11]:


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.random = None

def cloneList(head):
    if not head:
        return None

    # Step 1: Create a copy of each node and insert it between the current node and its next node
    curr = head
    while curr:
        new_node = Node(curr.data)
        new_node.next = curr.next
        curr.next = new_node
        curr = new_node.next

    # Step 2: Set the random pointers of the copied nodes
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # Step 3: Separate the original list and the copied list
    curr = head
    copy_head = head.next
    copy_curr = copy_head
    while curr:
        curr.next = curr.next.next
        if copy_curr.next:
            copy_curr.next = copy_curr.next.next
        curr = curr.next
        copy_curr = copy_curr.next

    return copy_head

def printList(head):
    curr = head
    while curr:
        random_data = curr.random.data if curr.random else "None"
        print(f"{curr.data}({random_data})", end=" -> ")
        curr = curr.next
    print("None")


# In[12]:


# Example 1
head1 = Node(1)
head1.next = Node(2)
head1.next.next = Node(3)
head1.next.next.next = Node(4)

head1.random = head1.next
head1.next.random = head1.next.next.next

print("Original Linked List:")
printList(head1)
# Output: 1(None) -> 2(3) -> 3(4) -> 4(2) -> None

copy_head1 = cloneList(head1)
print("\nCopied Linked List:")
printList(copy_head1)
# Output: 1(None) -> 2(3) -> 3(4) -> 4(2) -> None


# In[13]:


# Example 2
head2 = Node(1)
head2.next = Node(3)
head2.next.next = Node(5)
head2.next.next.next = Node(9)

head2.random = head2
head2.next.next.random = head2.next.next.next

print("\nOriginal Linked List:")
printList(head2)
# Output: 1(1) -> 3(None) -> 5(None) -> 9(5) -> None

copy_head2 = cloneList(head2)
print("\nCopied Linked List:")
printList(copy_head2)
# Output: 1(1) -> 3(None) -> 5(None) -> 9(5) -> None


# In[ ]:


Given the `head` of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return *the reordered list*.

The **first** node is considered **odd**, and the **second** node is **even**, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.

You must solve the problem in `O(1)` extra space complexity and `O(n)` time complexity.


Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]

Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]


# In[14]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def oddEvenList(head):
    if not head or not head.next:
        return head

    odd = head
    even = head.next
    even_head = even

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = even_head
    return head

def printList(head):
    curr = head
    while curr:
        print(curr.val, end=" -> ")
        curr = curr.next
    print("None")


# In[15]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(3)
head1.next.next.next = ListNode(4)
head1.next.next.next.next = ListNode(5)

print("Original Linked List:")
printList(head1)
# Output: 1 -> 2 -> 3 -> 4 -> 5 -> None

reordered_head1 = oddEvenList(head1)
print("\nReordered Linked List:")
printList(reordered_head1)
# Output: 1 -> 3 -> 5 -> 2 -> 4 -> None


# In[16]:


# Example 2
head2 = ListNode(2)
head2.next = ListNode(1)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(5)
head2.next.next.next.next = ListNode(6)
head2.next.next.next.next.next = ListNode(4)
head2.next.next.next.next.next.next = ListNode(7)

print("\nOriginal Linked List:")
printList(head2)
# Output: 2 -> 1 -> 3 -> 5 -> 6 -> 4 -> 7 -> None

reordered_head2 = oddEvenList(head2)
print("\nReordered Linked List:")
printList(reordered_head2)
# Output: 2 -> 3 -> 6 -> 7 -> 1 -> 5 -> 4 -> None


# In[ ]:


Given a singly linked list of size N. The task is to left-shift the linked list by k nodes, where k is a given positive integer smaller than or equal to length of the linked list.

Input:
N = 5
value[] = {2, 4, 7, 8, 9}
k = 3
Output:8 9 2 4 7
Explanation:Rotate 1:4 -> 7 -> 8 -> 9 -> 2
Rotate 2: 7 -> 8 -> 9 -> 2 -> 4
Rotate 3: 8 -> 9 -> 2 -> 4 -> 7
    
Input:
N = 8
value[] = {1, 2, 3, 4, 5, 6, 7, 8}
k = 4
Output:5 6 7 8 1 2 3 4


# In[ ]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def leftShift(head, k):
    if not head or not head.next or k == 0:
        return head

    # Calculate length of the linked list
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next

    # Calculate actual number of shifts needed
    k = k % length

    if k == 0:
        return head

    # Initialize pointers
    first = head
    second = head

    # Move second pointer k positions ahead
    for _ in range(k):
        second = second.next

    # Iterate until second pointer reaches last node
    while second.next:
        first = first.next
        second = second.next

    # Connect last node to the head of the original list
    second.next = head

    # Set new head of modified linked list
    new_head = first.next

    # Break the circular structure
    first.next = None

    return new_head

def printList(head):
    curr = head
    while curr:
        print(curr.val, end=" -> ")
        curr = curr.next
    print("None")


# In[ ]:


# Example 1
head1 = ListNode(2)
head1.next = ListNode(4)
head1.next.next = ListNode(7)
head1.next.next.next = ListNode(8)
head1.next.next.next.next = ListNode(9)

print("Original Linked List:")
printList(head1)
# Output: 2 -> 4 -> 7 -> 8 -> 9 -> None

k1 = 3
shifted_head1 = leftShift(head1, k1)
print(f"\nLeft-shifted Linked List by {k1}:")
printList(shifted_head1)
# Output: 8 -> 9 -> 2 -> 4 -> 7 -> None

# Example 2
head2 = ListNode(1)
head2.next = ListNode(2)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(4)
head2.next.next.next.next = ListNode(5)
head2.next.next.next.next.next = ListNode(6)
head2.next.next.next.next.next.next


# You are given the `head` of a linked list with `n` nodes.
# 
# For each node in the list, find the value of the **next greater node**. That is, for each node, find the value of the first node that is next to it and has a **strictly larger** value than it.
# 
# Return an integer array `answer` where `answer[i]` is the value of the next greater node of the `ith` node (**1-indexed**). If the `ith` node does not have a next greater node, set `answer[i] = 0`.
# 
# 
# Input: head = [2,1,5]
# Output: [5,5,0]
# 
# 
# Input: head = [2,7,4,3,5]
# Output: [7,0,5,5,0]

# In[ ]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def nextLargerNodes(head):
    # Convert linked list to array
    arr = []
    curr = head
    while curr:
        arr.append(curr.val)
        curr = curr.next

    n = len(arr)
    stack = []
    result = [0] * n

    # Iterate through array in reverse order
    for i in range(n - 1, -1, -1):
        while stack and arr[i] >= stack[-1]:
            stack.pop()

        if stack:
            result[i] = stack[-1]

        stack.append(arr[i])

    return result

# Example 1
head1 = ListNode(2)
head1.next = ListNode(1)
head1.next.next = ListNode(5)

print("Linked List:")
curr = head1
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")

result1 = nextLargerNodes(head1)
print("\nNext Greater Nodes:")
print(result1)
# Output: [5, 5, 0]


# In[ ]:


# Example 2
head2 = ListNode(2)
head2.next = ListNode(7)
head2.next.next = ListNode(4)
head2.next.next.next = ListNode(3)
head2.next.next.next.next = ListNode(5)

print("\nLinked List:")
curr = head2
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")

result2 = nextLargerNodes(head2)
print("\nNext Greater Nodes:")
print(result2)
# Output: [7, 0, 5, 5, 0]


# Given the `head` of a linked list, we repeatedly delete consecutive sequences of nodes that sum to `0` until there are no such sequences.
# 
# After doing so, return the head of the final linked list.  You may return any such answer.
# 
# (Note that in the examples below, all sequences are serializations of `ListNode` objects.)
# 
# Input: head = [1,2,-3,3,1]
# Output: [3,1]
# Note: The answer [1,2,1] would also be accepted.
# 
# 
# Input: head = [1,2,3,-3,4]
# Output: [1,2,4]
# 
# Input: head = [1,2,3,-3,-2]
# Output: [1]

# In[17]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def removeZeroSumSublists(head):
    # Create a dummy node
    dummy = ListNode(0)
    dummy.next = head

    runningSum = 0
    stack = [(0, dummy)]  # Initialize stack with dummy node

    curr = head
    while curr:
        runningSum += curr.val

        if runningSum == 0:
            # Remove nodes from stack until node before curr
            while stack[-1][1] != curr:
                stack.pop()

            # Update next pointer of previous node
            stack[-1][1].next = curr.next
        else:
            stack.append((runningSum, curr))

        curr = curr.next

    return dummy.next


# In[18]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(-3)
head1.next.next.next = ListNode(3)
head1.next.next.next.next = ListNode(1)

print("Linked List:")
curr = head1
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")

result1 = removeZeroSumSublists(head1)
print("\nUpdated Linked List:")
curr = result1
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")
# Output: [3, 1]


# In[ ]:


# Example 2
head2 = ListNode(1)
head2.next = ListNode(2)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(-3)
head2.next.next.next.next = ListNode(4)

print("\nLinked List:")
curr = head2
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")

result2 = removeZeroSumSublists(head2)
print("\nUpdated Linked List:")
curr = result2
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")
# Output: [1, 2, 4]


# In[ ]:


# Example 3
head3 = ListNode(1)
head3.next = ListNode(2)
head3.next.next = ListNode(3)
head3.next.next.next = ListNode(-3)
head3.next.next.next.next = ListNode(-2)

print("\nLinked List:")
curr = head3
while curr:
    print(curr.val, end=" -> ")
    curr = curr.next
print("None")

result3 = remove


# In[ ]:





# Given a singly linked list of size N. The task is to left-shift the linked list by k nodes, where k is a given positive integer smaller than or equal to length of the linked list.
# 
# Input:
# N = 5
# value[] = {2, 4, 7, 8, 9}
# k = 3
# Output:8 9 2 4 7
# Explanation:Rotate 1:4 -> 7 -> 8 -> 9 -> 2
# Rotate 2: 7 -> 8 -> 9 -> 2 -> 4
# Rotate 3: 8 -> 9 -> 2 -> 4 -> 7
# 
#     
# Input:
# N = 8
# value[] = {1, 2, 3, 4, 5, 6, 7, 8}
# k = 4
# Output:5 6 7 8 1 2 3 4

# In[ ]:





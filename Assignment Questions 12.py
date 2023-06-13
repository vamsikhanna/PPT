#!/usr/bin/env python
# coding: utf-8

# Given a singly linked list, delete middle of the linked list. For example, if given linked list is 1->2->3->4->5 then linked list should be modified to 1->2->4->5.If there are even nodes, then there would be two middle nodes, we need to delete the second middle element. For example, if given linked list is 1->2->3->4->5->6 then it should be modified to 1->2->3->5->6.If the input linked list is NULL or has 1 node, then it should return NULL
# 
# Input:
# LinkedList: 1->2->3->4->5
# Output:1 2 4 5
# 
# Input:
# LinkedList: 2->4->6->7->5->1
# Output:2 4 6 5 1

# In[1]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteMiddleNode(head):
    if head is None or head.next is None:
        return None

    slow = head
    fast = head
    prev = None

    while fast is not None and fast.next is not None:
        fast = fast.next.next
        prev = slow
        slow = slow.next

    prev.next = slow.next

    return head

def printLinkedList(head):
    current = head
    while current:
        print(current.val, end=" ")
        current = current.next


# In[2]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(3)
head1.next.next.next = ListNode(4)
head1.next.next.next.next = ListNode(5)
result1 = deleteMiddleNode(head1)
printLinkedList(result1)  # Output: 1 2 4 5
print()


# In[3]:


# Example 2
head2 = ListNode(2)
head2.next = ListNode(4)
head2.next.next = ListNode(6)
head2.next.next.next = ListNode(7)
head2.next.next.next.next = ListNode(5)
head2.next.next.next.next.next = ListNode(1)
result2 = deleteMiddleNode(head2)
printLinkedList(result2)  # Output: 2 4 6 5 1
print()


# In[ ]:


Given a linked list of N nodes. The task is to check if the linked list has a loop. Linked list can contain self loop.

Input:
N = 3
value[] = {1,3,4}
x(position at which tail is connected) = 2
Output:True
Explanation:In above test case N = 3.
The linked list with nodes N = 3 is
given. Then value of x=2 is given which
means last node is connected with xth
node of linked list. Therefore, there
exists a loop.

Input:
N = 4
value[] = {1,8,3,4}
x = 0
Output:False
Explanation:For N = 4 ,x = 0 means
then lastNode->next = NULL, then
the Linked list does not contains
any loop.


# In[29]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def hasLoop(head):
    if head is None or head.next is None:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if fast is None or fast.next is None:
            return False
        slow = slow.next
        fast = fast.next.next

    return True


# In[30]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(3)
head1.next.next = ListNode(4)
head1.next.next.next = head1.next
print(hasLoop(head1))  # Output: True


# In[31]:


# Example 2
head2 = ListNode(1)
head2.next = ListNode(8)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(4)
print(hasLoop(head2))  # Output: False


# Given a linked list consisting of L nodes and given a number N. The task is to find the Nth node from the end of the linked list.
# Input:
# N = 2
# LinkedList: 1->2->3->4->5->6->7->8->9
# Output:8
# Explanation:In the first example, there
# are 9 nodes in linked list and we need
# to find 2nd node from end. 2nd node
# from end is 8
# 
# Input:
# N = 5
# LinkedList: 10->5->100->5
# Output:-1
# Explanation:In the second example, there
# are 4 nodes in the linked list and we
# need to find 5th from the end. Since 'n'
# is more than the number of nodes in the
# linked list, the output is -1.

# In[26]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def findNthFromEnd(head, N):
    if head is None:
        return -1

    fast = head
    slow = head

    # Move the fast pointer N steps ahead
    for _ in range(N):
        if fast is None:
            return -1
        fast = fast.next

    # Move both pointers until fast reaches the end
    while fast is not None:
        fast = fast.next
        slow = slow.next

    return slow.val


# In[27]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(3)
head1.next.next.next = ListNode(4)
head1.next.next.next.next = ListNode(5)
head1.next.next.next.next.next = ListNode(6)
head1.next.next.next.next.next.next = ListNode(7)
head1.next.next.next.next.next.next.next = ListNode(8)
head1.next.next.next.next.next.next.next.next = ListNode(9)
print(findNthFromEnd(head1, 2))  # Output: 8


# In[28]:


# Example 2
head2 = ListNode(10)
head2.next = ListNode(5)
head2.next.next = ListNode(100)
head2.next.next.next = ListNode(5)
print(findNthFromEnd(head2, 5))  # Output: -1


# Given a singly linked list of characters, write a function that returns true if the given list is a palindrome, else false.
# 
# > Input: R->A->D->A->R->NULL
# > 
# > 
# > **Output:** Yes
# > 
# > **Input:** C->O->D->E->NULL
# > 
# > **Output:** No
# >

# In[23]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def isPalindrome(head):
    if head is None or head.next is None:
        return True

    # Find the midpoint and reverse the second half
    slow = fast = head
    prev = None

    while fast is not None and fast.next is not None:
        fast = fast.next.next
        next_node = slow.next
        slow.next = prev
        prev = slow
        slow = next_node

    if fast is not None:  # Linked list has odd number of nodes
        slow = slow.next

    # Compare the values of the first half with the reversed second half
    while prev is not None:
        if prev.val != slow.val:
            return False
        prev = prev.next
        slow = slow.next

    return True


# In[24]:


# Example 1
head1 = ListNode('R')
head1.next = ListNode('A')
head1.next.next = ListNode('D')
head1.next.next.next = ListNode('A')
head1.next.next.next.next = ListNode('R')
print(isPalindrome(head1))  # Output: True


# In[25]:


# Example 2
head2 = ListNode('C')
head2.next = ListNode('O')
head2.next.next = ListNode('D')
head2.next.next.next = ListNode('E')
print(isPalindrome(head2))  # Output: False


# Given a linked list of **N** nodes such that it may contain a loop.
# 
# A loop here means that the last node of the link list is connected to the node at position X(1-based index). If the link list does not have any loop, X=0.
# 
# Remove the loop from the linked list, if it is present, i.e. unlink the last node which is forming the loop.
# 
# 
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

# In[19]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def detectAndRemoveLoop(head):
    if head is None or head.next is None:
        return head

    slow = head
    fast = head

    # Detect the loop using Floyd's Algorithm
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            break

    if slow != fast:  # No loop
        return head

    # Move slow to the head and keep fast at the meeting point
    slow = head
    while slow.next != fast.next:
        slow = slow.next
        fast = fast.next

    # Break the loop
    fast.next = None

    return head

def printLinkedList(head):
    current = head
    while current:
        print(current.val, end=" ")
        current = current.next


# In[20]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(3)
head1.next.next = ListNode(4)
head1.next.next.next = head1.next
head1 = detectAndRemoveLoop(head1)
printLinkedList(head1)  # Output: 1


# In[21]:


# Example 2
head2 = ListNode(1)
head2.next = ListNode(8)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(4)
head2 = detectAndRemoveLoop(head2)
printLinkedList(head2)  # Output: 1


# In[22]:


# Example 3
head3 = ListNode(1)
head3.next = ListNode(2)
head3.next.next = ListNode(3)
head3.next.next.next = ListNode(4)
head3.next.next.next.next = head3  # Creating a loop by connecting the last node to the first node
head3 = detectAndRemoveLoop(head3)
printLinkedList(head3)  # Output: 1 2 3 4


# Given a linked list and two integers M and N. Traverse the linked list such that you retain M nodes then delete next N nodes, continue the same till end of the linked list.
# 
# Difficulty Level: Rookie
# 
# 
# Input:
# M = 2, N = 2
# Linked List: 1->2->3->4->5->6->7->8
# Output:
# Linked List: 1->2->5->6
# 
# Input:
# M = 3, N = 2
# Linked List: 1->2->3->4->5->6->7->8->9->10
# Output:
# Linked List: 1->2->3->6->7->8
# 
# 
# Input:
# M = 1, N = 1
# Linked List: 1->2->3->4->5->6->7->8->9->10
# Output:
# Linked List: 1->3->5->7->9

# In[16]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def deleteNodes(head, M, N):
    if head is None or M <= 0 or N <= 0:
        return head

    current = head
    previous = None

    while current is not None:
        # Skip M nodes
        for _ in range(M):
            if current is None:
                return head
            previous = current
            current = current.next

        # Delete N nodes
        for _ in range(N):
            if current is None:
                break
            current = current.next

        # Connect the previous node to the current node
        previous.next = current

    return head

def printLinkedList(head):
    current = head
    while current:
        print(current.val, end=" ")
        current = current.next


# In[17]:


# Example 1
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(3)
head1.next.next.next = ListNode(4)
head1.next.next.next.next = ListNode(5)
head1.next.next.next.next.next = ListNode(6)
head1.next.next.next.next.next.next = ListNode(7)
head1.next.next.next.next.next.next.next = ListNode(8)
M1 = 2
N1 = 2
head1 = deleteNodes(head1, M1, N1)
printLinkedList(head1)  # Output: 1 2 5 6


# In[ ]:


# Example 2
head2 = ListNode(1)
head2.next = ListNode(2)
head2.next.next = ListNode(3)
head2.next.next.next = ListNode(4)
head2.next.next.next.next = ListNode(5)
head2.next.next.next.next.next = ListNode(6)
head2.next.next.next.next.next.next = ListNode(7)
head2.next.next.next.next.next.next.next = ListNode(8)
head2.next.next.next.next.next.next.next.next = ListNode(9)
head2.next.next.next.next.next.next.next.next.next = ListNode(10)
M2 = 3
N2 = 2
head2 = deleteNodes(head2, M2, N2)
printLinkedList(head2)  # Output: 1 2 3 6 7 8


# In[18]:


# Example 3
head3 = ListNode(1)
head3.next = ListNode(2)
head3.next.next = ListNode(3)
head3.next.next.next = ListNode(4)
head3.next.next.next.next = ListNode(5)
head3.next.next.next.next.next = ListNode(6)
head3.next.next.next.next.next.next = ListNode(7)
head3.next.next.next.next.next.next.next = ListNode(8)
head3.next.next.next.next.next.next.next.next = ListNode(9)
head3.next.next.next.next.next.next.next.next.next = ListNode(10)
M3 = 1
N3 = 1
head3 = deleteNodes(head3, M3, N3)
printLinkedList(head3)  # Output: 1 3 5 7 9


# Given two linked lists, insert nodes of second list into first list at alternate positions of first list.
# For example, if first list is 5->7->17->13->11 and second is 12->10->2->4->6, the first list should become 5->12->7->10->17->2->13->4->11->6 and second list should become empty. The nodes of second list should only be inserted when there are positions available. For example, if the first list is 1->2->3 and second list is 4->5->6->7->8, then first list should become 1->4->2->5->3->6 and second list to 7->8.
# 
# Use of extra space is not allowed (Not allowed to create additional nodes), i.e., insertion must be done in-place. Expected time complexity is O(n) where n is number of nodes in first list

# In[13]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeLists(first, second):
    if first is None:
        return second
    if second is None:
        return first

    # Initialize pointers for both lists
    current_first = first
    current_second = second

    while current_first is not None and current_second is not None:
        # Save the next pointers for both lists
        first_next = current_first.next
        second_next = current_second.next

        # Insert the current node from the second list into the first list
        current_first.next = current_second
        current_second.next = first_next

        # Move the pointers ahead
        current_first = first_next
        current_second = second_next

    # Update the head of the second list to None
    second = None

    return first

def printLinkedList(head):
    current = head
    while current:
        print(current.val, end=" ")
        current = current.next


# In[14]:


# Example 1
first1 = ListNode(5)
first1.next = ListNode(7)
first1.next.next = ListNode(17)
first1.next.next.next = ListNode(13)
first1.next.next.next.next = ListNode(11)

second1 = ListNode(12)
second1.next = ListNode(10)
second1.next.next = ListNode(2)
second1.next.next.next = ListNode(4)
second1.next.next.next.next = ListNode(6)

first1 = mergeLists(first1, second1)
printLinkedList(first1)  # Output: 5 12 7 10 17 2 13 4 11
printLinkedList(second1)  # Output: Empty


# In[15]:


# Example 2
first2 = ListNode(1)
first2.next = ListNode(2)
first2.next.next = ListNode(3)

second2 = ListNode(4)
second2.next = ListNode(5)
second2.next.next = ListNode(6)
second2.next.next.next = ListNode(7)
second2.next.next.next.next = ListNode(8)

first2 = mergeLists(first2, second2)
printLinkedList(first2)  # Output: 1 4 2 5 3 6
printLinkedList(second2)  # Output: 7 8


# Given a singly linked list, find if the linked list is [circular](https://www.geeksforgeeks.org/circular-linked-list/amp/) or not.
# 
# > A linked list is called circular if it is not NULL-terminated and all nodes are connected in the form of a cycle. Below is an example of a circular linked list.
# >

# In[7]:


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def isCircular(head):
    if head is None:
        return False

    slow = head
    fast = head.next

    while fast and fast.next:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next.next

    return False


# In[8]:


# Example 1: Circular linked list
head1 = ListNode(1)
head1.next = ListNode(2)
head1.next.next = ListNode(3)
head1.next.next.next = head1  # Pointing back to the head

print(isCircular(head1))  # Output: True


# In[9]:


# Example 2: Non-circular linked list
head2 = ListNode(1)
head2.next = ListNode(2)
head2.next.next = ListNode(3)

print(isCircular(head2))  # Output: False


# In[ ]:





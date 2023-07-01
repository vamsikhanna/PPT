#!/usr/bin/env python
# coding: utf-8

# <aside>
# 💡 Question-1
# 
# You are given a binary tree. The binary tree is represented using the TreeNode class. Each TreeNode has an integer value and left and right children, represented using the TreeNode class itself. Convert this binary tree into a binary search tree.
# 
# Input:
# 
#         10
# 
#        /   \
# 
#      2      7
# 
#    /   \
# 
#  8      4
# 
# Output:
# 
#         8
# 
#       /   \
# 
#     4     10
# 
#   /   \
# 
# 2      7
# 
# </aside>

# In[1]:


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorderTraversal(node, values):
    if node is not None:
        inorderTraversal(node.left, values)
        values.append(node.val)
        inorderTraversal(node.right, values)


def assignValuesInorder(node, values):
    if node is not None:
        assignValuesInorder(node.left, values)
        node.val = values[0]
        values.pop(0)
        assignValuesInorder(node.right, values)


def convertToBST(root):
    if root is None:
        return None

    values = []
    inorderTraversal(root, values)
    values.sort()

    assignValuesInorder(root, values)

    return root


# In[2]:


# Test the algorithm with the given example

# Construct the binary tree
root = TreeNode(10)
root.left = TreeNode(2)
root.right = TreeNode(7)
root.left.left = TreeNode(8)
root.left.right = TreeNode(4)

# Convert the binary tree to a binary search tree
convertedRoot = convertToBST(root)

# Function to print the BST in inorder traversal
def inorderTraversal(node):
    if node is not None:
        inorderTraversal(node.left)
        print(node.val, end=" ")
        inorderTraversal(node.right)

print("Converted BST:")
inorderTraversal(convertedRoot)  # Output: 2 4 7 8 10


# <aside>
# 💡 Question-2:
# 
# Given a Binary Search Tree with all unique values and two keys. Find the distance between two nodes in BST. The given keys always exist in BST.
# 
# Example:
# 
# Consider the following BST:
# 
# ![1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2455039-7e12-43fc-a7d3-b5be24931c1c/1.png)
# 
# **Input-1:**
# 
# n = 9
# 
# values = [8, 3, 1, 6, 4, 7, 10, 14,13]
# 
# node-1 = 6
# 
# node-2 = 14
# 
# **Output-1:**
# 
# The distance between the two keys = 4
# 
# **Input-2:**
# 
# n = 9
# 
# values = [8, 3, 1, 6, 4, 7, 10, 14,13]
# 
# node-1 = 3
# 
# node-2 = 4
# 
# **Output-2:**
# 
# The distance between the two keys = 2
# 
# </aside>

# In[3]:


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def findPath(root, node, path):
    if root is None:
        return False

    path.append(root.val)

    if root.val == node:
        return True

    if (root.left is not None and findPath(root.left, node, path)) or (
        root.right is not None and findPath(root.right, node, path)
    ):
        return True

    path.pop()
    return False


def findLCA(root, node1, node2):
    path1 = []
    path2 = []

    if not findPath(root, node1, path1) or not findPath(root, node2, path2):
        return None

    i = 0
    while i < len(path1) and i < len(path2):
        if path1[i] != path2[i]:
            break
        i += 1

    return path1[i - 1]


def findDistance(root, node, distance):
    if root is None:
        return -1

    if root.val == node:
        return distance

    leftDistance = findDistance(root.left, node, distance + 1)
    if leftDistance != -1:
        return leftDistance

    rightDistance = findDistance(root.right, node, distance + 1)
    return rightDistance


def findNodeDistance(root, node1, node2):
    lca = findLCA(root, node1, node2)

    distance1 = findDistance(root, node1, 0)
    distance2 = findDistance(root, node2, 0)
    lcaDistance = findDistance(root, lca, 0)

    return distance1 + distance2 - 2 * lcaDistance


# Test the algorithm with the given examples

# Construct the BST
root = TreeNode(8)
root.left = TreeNode(3)
root.right = TreeNode(10)
root.left.left = TreeNode(1)
root.left.right = TreeNode(6)
root.left.right.left = TreeNode(4)
root.left.right.right = TreeNode(7)
root.right.right = TreeNode(14)
root.right.right.left = TreeNode(13)


# In[4]:


# Example 1
node1 = 6
node2 = 14
distance = findNodeDistance(root, node1, node2)
print("The distance between the two keys:", distance)  # Output: 4


# In[5]:


# Example 2
node1 = 3
node2 = 4
distance = findNodeDistance(root, node1, node2)
print("The distance between the two keys:", distance)  # Output: 2


# <aside>
# 💡 Question-3:
# 
# Write a program to convert a binary tree to a doubly linked list.
# 
# Input:
# 
#         10
# 
#        /   \
# 
#      5     20
# 
#            /   \
# 
#         30     35
# 
# Output:
# 
# 5 10 30 20 35
# 
# </aside>

# In[6]:


class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.prev = None
        self.next = None


def convertToDoublyLinkedList(root):
    if root is None:
        return None

    prev = None
    head = None

    def convertNode(node):
        nonlocal prev, head

        if node is None:
            return

        convertNode(node.left)

        if prev is None:
            head = node
        else:
            node.prev = prev
            prev.next = node

        prev = node

        convertNode(node.right)

    convertNode(root)
    prev.next = None

    return head


# In[7]:


# Test the algorithm with the given example

# Construct the binary tree
root = Node(10)
root.left = Node(5)
root.right = Node(20)
root.right.left = Node(30)
root.right.right = Node(35)

# Convert the binary tree to a doubly linked list
head = convertToDoublyLinkedList(root)

# Print the doubly linked list in forward traversal
def printDoublyLinkedListForward(head):
    curr = head
    while curr is not None:
        print(curr.data, end=" ")
        curr = curr.next
    print()

print("Doubly linked list (forward traversal):")
printDoublyLinkedListForward(head)  # Output: 5 10 30 20 35


# <aside>
# 💡 Question-4:
# 
# Write a program to connect nodes at the same level.
# 
# Input:
# 
#         1
# 
#       /   \
# 
#     2      3
# 
#   /   \   /   \
# 
# 4     5 6    7
# 
# Output:
# 
# 1 → -1
# 
# 2 → 3
# 
# 3 → -1
# 
# 4 → 5
# 
# 5 → 6
# 
# 6 → 7
# 
# 7 → -1
# 
# </aside>

# In[8]:


from collections import deque

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.nextRight = None


def connectNodes(root):
    if root is None:
        return None

    queue = deque()
    queue.append(root)

    while queue:
        size = len(queue)

        for i in range(size):
            node = queue.popleft()

            if i < size - 1:
                node.nextRight = queue[0]

            if node.left:
                queue.append(node.left)

            if node.right:
                queue.append(node.right)

    return root


# Test the algorithm with the given example

# Construct the binary tree
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)

# Connect nodes at the same level
connectedRoot = connectNodes(root)

# Print the connected nodes
def printConnectedNodes(root):
    while root:
        curr = root
        while curr:
            print(curr.data, end=" → ")
            curr = curr.nextRight
        print("-1")
        root = root.left

print("Connected nodes at the same level:")
printConnectedNodes(connectedRoot)


# In[ ]:





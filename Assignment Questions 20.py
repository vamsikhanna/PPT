#!/usr/bin/env python
# coding: utf-8

# <aside>
# 💡 Question-1
# 
# Given a binary tree, your task is to find subtree with maximum sum in tree.
# 
# Examples:
# 
# Input1 :       
# 
#        1
# 
#      /   \
# 
#    2      3
# 
#   / \    / \
# 
# 4   5  6   7
# 
# Output1 : 28
# 
# As all the tree elements are positive, the largest subtree sum is equal to sum of all tree elements.
# 
# Input2 :
# 
#        1
# 
#      /    \
# 
#   -2      3
# 
#   / \    /  \
# 
# 4   5  -6   2
# 
# Output2 : 7
# 
# Subtree with largest sum is :
# 
#  -2
# 
#  / \
# 
# 4   5
# 
# Also, entire tree sum is also 7.
# 
# </aside>

# In[1]:


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxSubtreeSum(root):
    if root is None:
        return 0, 0
    
    leftMaxSum, leftTotalSum = maxSubtreeSum(root.left)
    rightMaxSum, rightTotalSum = maxSubtreeSum(root.right)
    
    totalSum = root.val + leftTotalSum + rightTotalSum
    maxSum = max(root.val, leftMaxSum, rightMaxSum)
    maxSum = max(maxSum, root.val + leftMaxSum + rightMaxSum)
    
    return maxSum, totalSum


# In[2]:


# Test the algorithm with the provided examples

# Example 1
root1 = TreeNode(1)
root1.left = TreeNode(2)
root1.right = TreeNode(3)
root1.left.left = TreeNode(4)
root1.left.right = TreeNode(5)
root1.right.left = TreeNode(6)
root1.right.right = TreeNode(7)

maxSum1, _ = maxSubtreeSum(root1)
print("Maximum subtree sum in Example 1:", maxSum1)  # Output: 28


# In[3]:


# Example 2
root2 = TreeNode(1)
root2.left = TreeNode(-2)
root2.right = TreeNode(3)
root2.left.left = TreeNode(4)
root2.left.right = TreeNode(5)
root2.right.left = TreeNode(-6)
root2.right.right = TreeNode(2)

maxSum2, _ = maxSubtreeSum(root2)
print("Maximum subtree sum in Example 2:", maxSum2)  # Output: 7


# <aside>
# 💡 Question-2
# 
# Construct the BST (Binary Search Tree) from its given level order traversal.
# 
# Example:
# 
# Input: arr[] = {7, 4, 12, 3, 6, 8, 1, 5, 10}
# 
# Output: BST:
# 
#             7
# 
#          /    \
# 
#        4     12
# 
#      /  \     /
# 
#     3   6  8
# 
#    /    /     \
# 
#  1    5      10
# 
# </aside>
# 
# <aside>
# 💡 Question-3
# 
# </aside>

# In[4]:


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def constructBST(levelOrder):
    if levelOrder is None or len(levelOrder) == 0:
        return None
    
    root = TreeNode(levelOrder[0])
    leftSubtree = []
    rightSubtree = []
    
    for i in range(1, len(levelOrder)):
        if levelOrder[i] < root.val:
            leftSubtree.append(levelOrder[i])
        else:
            rightSubtree.append(levelOrder[i])
    
    root.left = constructBST(leftSubtree)
    root.right = constructBST(rightSubtree)
    
    return root


# In[5]:


# Test the algorithm with the given example

levelOrder = [7, 4, 12, 3, 6, 8, 1, 5, 10]
root = constructBST(levelOrder)

# Function to print the BST in inorder traversal
def inorderTraversal(node):
    if node is not None:
        inorderTraversal(node.left)
        print(node.val, end=" ")
        inorderTraversal(node.right)

print("Constructed BST:")
inorderTraversal(root)  # Output: 1 3 4 5 6 7 8 10 12


# <aside>
# 💡 Question-3
# 
# Given an array of size n. The problem is to check whether the given array can represent the level order traversal of a Binary Search Tree or not.
# 
# Examples:
# 
# Input1 : arr[] = {7, 4, 12, 3, 6, 8, 1, 5, 10}
# 
# Output1 : Yes
# 
# For the given arr[], the Binary Search Tree is:
# 
#             7
# 
#          /    \
# 
#        4     12
# 
#      /  \     /
# 
#     3   6  8
# 
#    /    /     \
# 
#  1    5      10
# 
# Input2 : arr[] = {11, 6, 13, 5, 12, 10}
# 
# Output2 : No
# 
# The given arr[] does not represent the level order traversal of a BST.
# 
# </aside>

# In[10]:


def isLevelOrderBST(levelOrder):
    if levelOrder is None or len(levelOrder) <= 1:
        return True
    
    stack = []
    root = levelOrder[0]
    
    for i in range(1, len(levelOrder)):
        if levelOrder[i] < root:
            return False
        
        while len(stack) > 0 and levelOrder[i] > stack[-1]:
            root = stack.pop()
            
        stack.append(levelOrder[i])
        
    return True


# In[11]:


# Test the algorithm with the given examples

# Example 1
levelOrder1 = [7, 4, 12, 3, 6, 8, 1, 5, 10]
result1 = isLevelOrderBST(levelOrder1)
print("Can represent level order traversal of a BST? ", result1)  # Output: Yes


# In[9]:


# Example 2
levelOrder2 = [11, 6, 13, 5, 12, 10]
result2 = isLevelOrderBST(levelOrder2)
print("Can represent level order traversal of a BST? ", result2)  # Output: No


# In[ ]:





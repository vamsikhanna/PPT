#!/usr/bin/env python
# coding: utf-8

# . Write a Python program to reverse a string without using any built-in string reversal functions.
# 

# In[1]:


def reverse_string(s):
    # Convert the string to a list of characters
    chars = list(s)
    
    # Initialize two pointers, one at the start and the other at the end of the list
    start = 0
    end = len(chars) - 1
    
    # Swap characters symmetrically from both ends until the pointers meet in the middle
    while start < end:
        chars[start], chars[end] = chars[end], chars[start]
        start += 1
        end -= 1
    
    # Convert the list of characters back to a string
    reversed_string = ''.join(chars)
    
    return reversed_string

# Test the function
string = "Hello, World!"
reversed_string = reverse_string(string)
print(reversed_string)


# 2. Implement a function to check if a given string is a palindrome.
# 

# In[2]:


def is_palindrome(s):
    # Convert the string to lowercase and remove non-alphanumeric characters
    s = ''.join(c.lower() for c in s if c.isalnum())
    
    # Initialize two pointers, one at the start and the other at the end of the string
    start = 0
    end = len(s) - 1
    
    # Compare characters symmetrically from both ends until the pointers meet in the middle
    while start < end:
        if s[start] != s[end]:
            return False
        start += 1
        end -= 1
    
    return True

# Test the function
string = "A man, a plan, a canal: Panama"
print(is_palindrome(string))  # Output: True

string = "Hello, World!"
print(is_palindrome(string))  # Output: False


# 3. Write a program to find the largest element in a given list.

# In[3]:


def find_largest_element(lst):
    if not lst:
        return None
    
    largest = lst[0]
    for num in lst:
        if num > largest:
            largest = num
    
    return largest

# Test the program
numbers = [5, 9, 3, 2, 7, 10]
largest_number = find_largest_element(numbers)
print(largest_number)  # Output: 10

empty_list = []
largest_number = find_largest_element(empty_list)
print(largest_number)  # Output: None


# 4. Implement a function to count the occurrence of each element in a list.
# 

# In[4]:


def count_occurrences(lst):
    occurrence_count = {}
    
    for element in lst:
        if element in occurrence_count:
            occurrence_count[element] += 1
        else:
            occurrence_count[element] = 1
    
    return occurrence_count

# Test the function
numbers = [1, 2, 3, 2, 1, 3, 3, 4, 5, 4, 4]
occurrences = count_occurrences(numbers)
print(occurrences)
# Output: {1: 2, 2: 2, 3: 3, 4: 3, 5: 1}

characters = ['a', 'b', 'c', 'a', 'b', 'c', 'c', 'd', 'e', 'd', 'd']
occurrences = count_occurrences(characters)
print(occurrences)
# Output: {'a': 2, 'b': 2, 'c': 3, 'd': 3, 'e': 1}


# In[ ]:





# In[ ]:


5. Write a Python program to find the second largest number in a list.


# In[5]:


def find_second_largest(lst):
    if len(lst) < 2:
        return None
    
    largest = float('-inf')
    second_largest = float('-inf')
    
    for num in lst:
        if num > largest:
            second_largest = largest
            largest = num
        elif num > second_largest and num != largest:
            second_largest = num
    
    if second_largest == float('-inf'):
        return None
    
    return second_largest

# Test the program
numbers = [5, 9, 3, 2, 7, 10]
second_largest_number = find_second_largest(numbers)
print(second_largest_number)  # Output: 9

numbers = [5, 9, 3, 2, 7, 10, 10]
second_largest_number = find_second_largest(numbers)
print(second_largest_number)  # Output: 9

numbers = [1, 1, 1, 1]
second_largest_number = find_second_largest(numbers)
print(second_largest_number)  # Output: None

empty_list = []
second_largest_number = find_second_largest(empty_list)
print(second_largest_number)  # Output: None


# In[ ]:





# In[ ]:


6. Implement a function to remove duplicate elements from a list.


# In[6]:


def remove_duplicates(lst):
    seen = set()
    result = []
    
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    
    return result

# Test the function
numbers = [1, 2, 3, 2, 1, 3, 3, 4, 5, 4, 4]
unique_numbers = remove_duplicates(numbers)
print(unique_numbers)
# Output: [1, 2, 3, 4, 5]

characters = ['a', 'b', 'c', 'a', 'b', 'c', 'c', 'd', 'e', 'd', 'd']
unique_characters = remove_duplicates(characters)
print(unique_characters)
# Output: ['a', 'b', 'c', 'd', 'e']


# In[ ]:





# In[ ]:


7. Write a program to calculate the factorial of a given number.


# In[7]:


def factorial(n):
    if n < 0:
        return None
    if n == 0:
        return 1
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return result

# Test the program
num = 5
factorial_num = factorial(num)
print(factorial_num)  # Output: 120

num = 0
factorial_num = factorial(num)
print(factorial_num)  # Output: 1

num = -3
factorial_num = factorial(num)
print(factorial_num)  # Output: None


# In[ ]:


. Implement a function to check if a given number is prime.


# In[8]:


def is_prime(n):
    if n < 2:
        return False
    
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    
    return True

# Test the function
num = 17
print(is_prime(num))  # Output: True

num = 10
print(is_prime(num))  # Output: False

num = 1
print(is_prime(num))  # Output: False


# In[ ]:





# 9. Write a Python program to sort a list of integers in ascending order.
# 

# In[9]:


def sort_list(lst):
    lst.sort()
    return lst

# Test the program
numbers = [5, 2, 8, 1, 9, 3]
sorted_numbers = sort_list(numbers)
print(sorted_numbers)  # Output: [1, 2, 3, 5, 8, 9]


# 10. Implement a function to find the sum of all numbers in a list.
# 

# In[10]:


def sum_of_numbers(lst):
    total = 0
    for num in lst:
        total += num
    return total

# Test the function
numbers = [1, 2, 3, 4, 5]
sum_of_nums = sum_of_numbers(numbers)
print(sum_of_nums)  # Output: 15

empty_list = []
sum_of_nums = sum_of_numbers(empty_list)
print(sum_of_nums)  # Output: 0


# 11. Write a program to find the common elements between two lists.
# 

# In[11]:


def find_common_elements(list1, list2):
    common_elements = []
    for item in list1:
        if item in list2:
            common_elements.append(item)
    return common_elements

# Test the program
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
common_elements = find_common_elements(list1, list2)
print(common_elements)  # Output: [4, 5]

list3 = [10, 20, 30]
list4 = [40, 50, 60]
common_elements = find_common_elements(list3, list4)
print(common_elements)  # Output: []

list5 = [1, 2, 3, 4]
list6 = [4, 4, 3, 3, 2, 2]
common_elements = find_common_elements(list5, list6)
print(common_elements)  # Output: [2, 3, 4]


# 12. Implement a function to check if a given string is an anagram of another string.
# 

# In[13]:


def is_anagram(str1, str2):
    # Remove spaces and convert to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()

    # Check if the lengths of the strings are equal
    if len(str1) != len(str2):
        return False

    # Create dictionaries to store character frequencies
    char_freq_1 = {}
    char_freq_2 = {}

    # Count the frequency of each character in str1
    for char in str1:
        if char in char_freq_1:
            char_freq_1[char] += 1
        else:
            char_freq_1[char] = 1

    # Count the frequency of each character in str2
    for char in str2:
        if char in char_freq_2:
            char_freq_2[char] += 1
        else:
            char_freq_2[char] = 1

    # Compare the character frequencies
    return char_freq_1 == char_freq_2

# Test the function
string1 = "listen"
string2 = "silent"
print(is_anagram(string1, string2))  # Output: True

string3 = "hello"
string4 = "world"
print(is_anagram(string3, string4))  # Output: False


# 3. Write a Python program to generate all permutations of a given string

# In[14]:


def generate_permutations(s):
    # Base case: If the string is empty or contains only one character,
    # return a list containing the string itself as the only permutation
    if len(s) <= 1:
        return [s]

    # List to store all permutations
    permutations = []

    # Iterate over each character in the string
    for i in range(len(s)):
        # Choose the current character as the first character
        first_char = s[i]

        # Generate all permutations of the remaining characters
        remaining_chars = s[:i] + s[i+1:]
        sub_permutations = generate_permutations(remaining_chars)

        # Add the current character to each sub-permutation
        for sub_permutation in sub_permutations:
            permutation = first_char + sub_permutation
            permutations.append(permutation)

    return permutations

# Test the program
string = "abc"
permutations = generate_permutations(string)
print(permutations)


# 14. Implement a function to calculate the Fibonacci sequence up to a given number of terms.
# 

# In[15]:


def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        sequence = [0, 1]
        while len(sequence) < n:
            next_number = sequence[-1] + sequence[-2]
            sequence.append(next_number)
        return sequence

# Test the function
terms = 10
fib_sequence = fibonacci(terms)
print(fib_sequence)  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

terms = 1
fib_sequence = fibonacci(terms)
print(fib_sequence)  # Output: [0]

terms = 0
fib_sequence = fibonacci(terms)
print(fib_sequence)  # Output: []


# 15. Write a program to find the median of a list of numbers.
# 

# In[16]:


def find_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    if n % 2 == 0:
        mid1 = n // 2
        mid2 = mid1 - 1
        median = (sorted_numbers[mid1] + sorted_numbers[mid2]) / 2
    else:
        mid = n // 2
        median = sorted_numbers[mid]

    return median

# Test the program
num_list = [5, 2, 9, 1, 7, 4, 8]
median_value = find_median(num_list)
print(median_value)  # Output: 5

num_list = [3, 1, 6, 2, 4, 5]
median_value = find_median(num_list)
print(median_value)  # Output: 3.5


# 16. Implement a function to check if a given list is sorted in non-decreasing order.
# 

# In[17]:


def is_sorted(lst):
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
    return True

# Test the function
numbers1 = [1, 2, 3, 4, 5]
print(is_sorted(numbers1))  # Output: True

numbers2 = [5, 2, 8, 1, 9, 3]
print(is_sorted(numbers2))  # Output: False

numbers3 = [1, 1, 2, 3, 4, 4, 5]
print(is_sorted(numbers3))  # Output: True


# 17. Write a Python program to find the intersection of two lists.
# 

# In[18]:


def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return list(intersection)

# Test the program
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
intersection = find_intersection(list1, list2)
print(intersection)  # Output: [4, 5]

list3 = [10, 20, 30]
list4 = [40, 50, 60]
intersection = find_intersection(list3, list4)
print(intersection)  # Output: []

list5 = [1, 2, 3, 4]
list6 = [4, 4, 3, 3, 2, 2]
intersection = find_intersection(list5, list6)
print(intersection)  # Output: [2, 3, 4]


# 18. Implement a function to find the maximum subarray sum in a given list.
# 

# In[19]:


def find_maximum_subarray_sum(lst):
    max_sum = float('-inf')  # Initialize with negative infinity
    current_sum = 0

    for num in lst:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum

# Test the function
numbers = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_subarray_sum = find_maximum_subarray_sum(numbers)
print(max_subarray_sum)  # Output: 6

numbers = [-1, -2, -3, -4, -5]
max_subarray_sum = find_maximum_subarray_sum(numbers)
print(max_subarray_sum)  # Output: -1


# 19. Write a program to remove all vowels from a given string

# In[20]:


def remove_vowels(string):
    vowels = 'aeiouAEIOU'
    without_vowels = ''
    for char in string:
        if char not in vowels:
            without_vowels += char
    return without_vowels

# Test the program
text = "Hello, World!"
result = remove_vowels(text)
print(result)  # Output: Hll, Wrld!

text = "Python Programming"
result = remove_vowels(text)
print(result)  # Output: Pythn Prgrmmng


# 20. Implement a function to reverse the order of words in a given sentence.

# In[21]:


def reverse_sentence(sentence):
    words = sentence.split()
    reversed_words = words[::-1]
    reversed_sentence = ' '.join(reversed_words)
    return reversed_sentence

# Test the function
text = "Hello, world! Welcome to Python."
reversed_text = reverse_sentence(text)
print(reversed_text)  # Output: Python. to Welcome world! Hello,

text = "This is a sample sentence."
reversed_text = reverse_sentence(text)
print(reversed_text)  # Output: sentence. sample a is This


# 21. Write a Python program to check if two strings are anagrams of each other.
# 

# In[22]:


def are_anagrams(str1, str2):
    # Remove spaces and convert to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()

    # Check if the lengths of the strings are equal
    if len(str1) != len(str2):
        return False

    # Create dictionaries to store character frequencies
    char_freq_1 = {}
    char_freq_2 = {}

    # Count the frequency of each character in str1
    for char in str1:
        if char in char_freq_1:
            char_freq_1[char] += 1
        else:
            char_freq_1[char] = 1

    # Count the frequency of each character in str2
    for char in str2:
        if char in char_freq_2:
            char_freq_2[char] += 1
        else:
            char_freq_2[char] = 1

    # Compare the character frequencies
    return char_freq_1 == char_freq_2

# Test the program
string1 = "listen"
string2 = "silent"
print(are_anagrams(string1, string2))  # Output: True

string3 = "hello"
string4 = "world"
print(are_anagrams(string3, string4))  # Output: False


# 22. Implement a function to find the first non-repeating character in a string.
# 

# In[23]:


def find_first_non_repeating_character(string):
    char_count = {}

    # Count the frequency of each character
    for char in string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # Find the first non-repeating character
    for char in string:
        if char_count[char] == 1:
            return char

    # If no non-repeating character found, return None
    return None

# Test the function
text = "abaccdeff"
first_non_repeating_char = find_first_non_repeating_character(text)
print(first_non_repeating_char)  # Output: "b"

text = "hello"
first_non_repeating_char = find_first_non_repeating_character(text)
print(first_non_repeating_char)  # Output: "h"

text = "aabbcc"
first_non_repeating_char = find_first_non_repeating_character(text)
print(first_non_repeating_char)  # Output: None


# 23. Write a program to find the prime factors of a given number

# In[24]:


def find_prime_factors(n):
    factors = []
    divisor = 2

    while divisor <= n:
        if n % divisor == 0:
            factors.append(divisor)
            n = n // divisor
        else:
            divisor += 1

    return factors

# Test the program
number = 36
prime_factors = find_prime_factors(number)
print(prime_factors)  # Output: [2, 2, 3, 3]

number = 56
prime_factors = find_prime_factors(number)
print(prime_factors)  # Output: [2, 2, 2, 7]

number = 17
prime_factors = find_prime_factors(number)
print(prime_factors)  # Output: [17]


# 4. Implement a function to check if a given number is a power of two.
# 

# In[25]:


def is_power_of_two(n):
    if n <= 0:
        return False
    while n > 1:
        if n % 2 != 0:
            return False
        n = n // 2
    return True

# Test the function
number1 = 16
print(is_power_of_two(number1))  # Output: True

number2 = 10
print(is_power_of_two(number2))  # Output: False

number3 = 128
print(is_power_of_two(number3))  # Output: True

number4 = 0
print(is_power_of_two(number4))  # Output: False


# 25. Write a Python program to merge two sorted lists into a single sorted list.
# 

# In[26]:


def merge_sorted_lists(list1, list2):
    merged_list = []
    i = 0  # Index for list1
    j = 0  # Index for list2

    # Merge the lists while both have elements
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1

    # Append remaining elements of list1, if any
    while i < len(list1):
        merged_list.append(list1[i])
        i += 1

    # Append remaining elements of list2, if any
    while j < len(list2):
        merged_list.append(list2[j])
        j += 1

    return merged_list

# Test the program
nums1 = [1, 3, 5, 7]
nums2 = [2, 4, 6, 8]
merged_nums = merge_sorted_lists(nums1, nums2)
print(merged_nums)  # Output: [1, 2, 3, 4, 5, 6, 7, 8]

nums3 = [10, 20, 30]
nums4 = [5, 15, 25]
merged_nums = merge_sorted_lists(nums3, nums4)
print(merged_nums)  # Output: [5, 10, 15, 20, 25, 30]


# 26. Implement a function to find the mode of a list of numbers.
# 

# In[27]:


from collections import Counter

def find_mode(numbers):
    frequency = Counter(numbers)
    max_count = max(frequency.values())
    mode = [num for num, count in frequency.items() if count == max_count]
    return mode

# Test the function
nums1 = [1, 2, 3, 4, 5, 2, 3, 2]
mode_nums = find_mode(nums1)
print(mode_nums)  # Output: [2]

nums2 = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
mode_nums = find_mode(nums2)
print(mode_nums)  # Output: [4]

nums3 = [1, 2, 3, 4, 5]
mode_nums = find_mode(nums3)
print(mode_nums)  # Output: [1, 2, 3, 4, 5]


# 27. Write a program to find the greatest common divisor (GCD) of two numbers.
# 

# In[28]:


def find_gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Test the program
num1 = 24
num2 = 36
gcd = find_gcd(num1, num2)
print(gcd)  # Output: 12

num3 = 17
num4 = 23
gcd = find_gcd(num3, num4)
print(gcd)  # Output: 1

num5 = 60
num6 = 48
gcd = find_gcd(num5, num6)
print(gcd)  # Output: 12


# 28. Implement a function to calculate the square root of a given number

# In[29]:


def calculate_square_root(n):
    if n < 0:
        raise ValueError("Square root is not defined for negative numbers.")
    if n == 0:
        return 0

    # Initialize the guess
    guess = n / 2

    # Iterate until convergence
    while True:
        new_guess = (guess + n / guess) / 2
        if abs(new_guess - guess) < 1e-9:  # Check for convergence
            return new_guess
        guess = new_guess

# Test the function
number1 = 16
sqrt1 = calculate_square_root(number1)
print(sqrt1)  # Output: 4.0

number2 = 2
sqrt2 = calculate_square_root(number2)
print(sqrt2)  # Output: 1.4142135623730951

number3 = 0
sqrt3 = calculate_square_root(number3)
print(sqrt3)  # Output: 0.0


# In[ ]:





# 29. Write a Python program to check if a given string is a valid palindrome ignoring 

# In[30]:


def is_valid_palindrome(s):
    # Remove non-alphanumeric characters and convert to lowercase
    s = ''.join(c.lower() for c in s if c.isalnum())

    # Check if the string is a palindrome
    return s == s[::-1]

# Test the program
string1 = "A man, a plan, a canal: Panama"
print(is_valid_palindrome(string1))  # Output: True

string2 = "race a car"
print(is_valid_palindrome(string2))  # Output: False

string3 = "Never odd or even"
print(is_valid_palindrome(string3))  # Output: True


# 30. Implement a function to find the minimum element in a rotated sorted list

# In[31]:


def find_minimum(nums):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        # Check if mid element is greater than the rightmost element
        if nums[mid] > nums[right]:
            left = mid + 1
        # Check if mid element is smaller than the rightmost element
        else:
            right = mid

    return nums[left]

# Test the function
rotated_nums1 = [4, 5, 6, 7, 0, 1, 2]
min_num1 = find_minimum(rotated_nums1)
print(min_num1)  # Output: 0

rotated_nums2 = [3, 4, 5, 1, 2]
min_num2 = find_minimum(rotated_nums2)
print(min_num2)  # Output: 1

rotated_nums3 = [1]
min_num3 = find_minimum(rotated_nums3)
print(min_num3)  # Output: 1


# 31. Write a program to find the sum of all even numbers in a list.
# 

# In[32]:


def sum_even_numbers(numbers):
    sum_even = 0
    for num in numbers:
        if num % 2 == 0:
            sum_even += num
    return sum_even

# Test the program
nums1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sum_even1 = sum_even_numbers(nums1)
print(sum_even1)  # Output: 30

nums2 = [2, 4, 6, 8, 10]
sum_even2 = sum_even_numbers(nums2)
print(sum_even2)  # Output: 30

nums3 = [1, 3, 5, 7, 9]
sum_even3 = sum_even_numbers(nums3)
print(sum_even3)  # Output: 0


# 32. Implement a function to calculate the power of a number using recursion.
# 

# In[33]:


def power(base, exponent):
    if exponent == 0:
        return 1
    elif exponent > 0:
        return base * power(base, exponent - 1)
    else:
        return 1 / base * power(base, exponent + 1)

# Test the function
base1 = 2
exponent1 = 3
result1 = power(base1, exponent1)
print(result1)  # Output: 8

base2 = 5
exponent2 = -2
result2 = power(base2, exponent2)
print(result2)  # Output: 0.04

base3 = 0
exponent3 = 5
result3 = power(base3, exponent3)
print(result3)  # Output: 0


# 33. Write a Python program to remove duplicates from a list while preserving the order.

# In[34]:


def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Test the program
numbers1 = [1, 2, 3, 2, 4, 3, 5]
result1 = remove_duplicates(numbers1)
print(result1)  # Output: [1, 2, 3, 4, 5]

numbers2 = [1, 1, 1, 1, 1]
result2 = remove_duplicates(numbers2)
print(result2)  # Output: [1]

numbers3 = [1, 2, 3, 4, 5]
result3 = remove_duplicates(numbers3)
print(result3)  # Output: [1, 2, 3, 4, 5]


# 34. Implement a function to find the longest common prefix among a list of strings.
# 

# In[35]:


def find_longest_common_prefix(strs):
    if not strs:
        return ""
    common_prefix = strs[0]
    for string in strs[1:]:
        while not string.startswith(common_prefix):
            common_prefix = common_prefix[:-1]
            if not common_prefix:
                return ""
    return common_prefix

# Test the function
words1 = ["flower", "flow", "flight"]
prefix1 = find_longest_common_prefix(words1)
print(prefix1)  # Output: "fl"

words2 = ["dog", "racecar", "car"]
prefix2 = find_longest_common_prefix(words2)
print(prefix2)  # Output: ""

words3 = ["apple", "apple", "apple"]
prefix3 = find_longest_common_prefix(words3)
print(prefix3)  # Output: "apple"


# 35. Write a program to check if a given number is a perfect square.

# In[36]:


def is_perfect_square(num):
    if num < 0:
        return False
    if num == 0:
        return True

    left = 0
    right = num

    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid

        if square == num:
            return True
        elif square < num:
            left = mid + 1
        else:
            right = mid - 1

    return False

# Test the program
number1 = 16
print(is_perfect_square(number1))  # Output: True

number2 = 14
print(is_perfect_square(number2))  # Output: False

number3 = 0
print(is_perfect_square(number3))  # Output: True

number4 = -9
print(is_perfect_square(number4))  # Output: False


# 36. Implement a function to calculate the product of all elements in a list.
# 

# In[37]:


def calculate_product(numbers):
    product = 1
    for num in numbers:
        product *= num
    return product

# Test the function
nums1 = [1, 2, 3, 4, 5]
product1 = calculate_product(nums1)
print(product1)  # Output: 120

nums2 = [2, 4, 6, 8, 10]
product2 = calculate_product(nums2)
print(product2)  # Output: 3840

nums3 = [1, 3, 5, 7, 9]
product3 = calculate_product(nums3)
print(product3)  # Output: 945


# 37. Write a Python program to reverse the order of words in a sentence while preserving the word order.
# 

# In[38]:


def reverse_sentence(sentence):
    words = sentence.split()
    reversed_words = words[::-1]
    reversed_sentence = ' '.join(reversed_words)
    return reversed_sentence

# Test the program
sentence1 = "Hello World"
reversed_sentence1 = reverse_sentence(sentence1)
print(reversed_sentence1)  # Output: "World Hello"

sentence2 = "I love Python programming"
reversed_sentence2 = reverse_sentence(sentence2)
print(reversed_sentence2)  # Output: "programming Python love I"

sentence3 = "This is a sentence"
reversed_sentence3 = reverse_sentence(sentence3)
print(reversed_sentence3)  # Output: "sentence a is This"


# 38. Implement a function to find the missing number in a given list of consecutive numbers.
# 

# In[39]:


def find_missing_number(numbers):
    n = len(numbers) + 1
    total_sum = (n * (n + 1)) // 2
    actual_sum = sum(numbers)
    missing_number = total_sum - actual_sum
    return missing_number

# Test the function
nums1 = [1, 2, 3, 5, 6, 7]
missing_num1 = find_missing_number(nums1)
print(missing_num1)  # Output: 4

nums2 = [10, 12, 13, 14, 15]
missing_num2 = find_missing_number(nums2)
print(missing_num2)  # Output: 11

nums3 = [2, 3, 4, 5, 6]
missing_num3 = find_missing_number(nums3)
print(missing_num3)  # Output: 1


# In[ ]:


39. Write a program to find the sum of digits of a given number.


# In[40]:


def sum_of_digits(number):
    digit_sum = 0
    while number > 0:
        digit = number % 10
        digit_sum += digit
        number //= 10
    return digit_sum

# Test the program
num1 = 12345
sum1 = sum_of_digits(num1)
print(sum1)  # Output: 15

num2 = 987654321
sum2 = sum_of_digits(num2)
print(sum2)  # Output: 45

num3 = 0
sum3 = sum_of_digits(num3)
print(sum3)  # Output: 0


# 40. Implement a function to check if a given string is a valid palindrome considering case sensitivity.
# 

# In[41]:


def is_valid_palindrome(string):
    left = 0
    right = len(string) - 1

    while left < right:
        if string[left] != string[right]:
            return False
        left += 1
        right -= 1

    return True

# Test the function
word1 = "racecar"
print(is_valid_palindrome(word1))  # Output: True

word2 = "level"
print(is_valid_palindrome(word2))  # Output: True

word3 = "Hello"
print(is_valid_palindrome(word3))  # Output: False


# 41. Write a Python program to find the smallest missing positive integer in a list.
# 

# In[42]:


def find_smallest_missing_positive(nums):
    n = len(nums)

    # Step 1: Move all positive numbers to the left side
    left = 0
    for i in range(n):
        if nums[i] > 0:
            nums[left], nums[i] = nums[i], nums[left]
            left += 1

    # Step 2: Mark visited numbers
    for i in range(left):
        num = abs(nums[i])
        if num <= left:
            nums[num - 1] = -abs(nums[num - 1])

    # Step 3: Find the first positive number
    for i in range(left):
        if nums[i] > 0:
            return i + 1

    # Step 4: All positive numbers are present, return the next number
    return left + 1

# Test the program
numbers1 = [3, 4, -1, 1]
smallest1 = find_smallest_missing_positive(numbers1)
print(smallest1)  # Output: 2

numbers2 = [1, 2, 0]
smallest2 = find_smallest_missing_positive(numbers2)
print(smallest2)  # Output: 3

numbers3 = [7, 8, 9, 11, 12]
smallest3 = find_smallest_missing_positive(numbers3)


# 42. Implement a function to find the longest palindrome substring in a given string.

# In[43]:


def longest_palindrome_substring(s):
    n = len(s)
    if n < 2:
        return s

    start = 0
    max_length = 1

    def expand_around_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for i in range(n):
        len1 = expand_around_center(i, i)
        len2 = expand_around_center(i, i + 1)
        length = max(len1, len2)
        if length > max_length:
            max_length = length
            start = i - (length - 1) // 2

    return s[start : start + max_length]

# Test the function
string1 = "babad"
print(longest_palindrome_substring(string1))  # Output: "bab"

string2 = "cbbd"
print(longest_palindrome_substring(string2))  # Output: "bb"

string3 = "a"
print(longest_palindrome_substring(string3))  # Output: "a"


# 43. Write a program to find the number of occurrences of a given element in a list.
# 

# In[44]:


def count_occurrences(numbers, element):
    count = 0
    for num in numbers:
        if num == element:
            count += 1
    return count

# Test the program
nums = [1, 2, 3, 4, 2, 2, 5, 2]
element = 2
occurrences = count_occurrences(nums, element)
print(occurrences)  # Output: 4


# 44. Implement a function to check if a given number is a perfect number.
# 

# In[45]:


def is_perfect_number(number):
    if number <= 0:
        return False

    divisors_sum = 0
    for i in range(1, number):
        if number % i == 0:
            divisors_sum += i

    return divisors_sum == number

# Test the function
num1 = 6
print(is_perfect_number(num1))  # Output: True

num2 = 28
print(is_perfect_number(num2))  # Output: True

num3 = 12
print(is_perfect_number(num3))  # Output: False


# 45. Write a Python program to remove all duplicates from a string.

# In[46]:


def remove_duplicates(string):
    unique_chars = []
    for char in string:
        if char not in unique_chars:
            unique_chars.append(char)
    return ''.join(unique_chars)

# Test the program
text = "Hello World"
result = remove_duplicates(text)
print(result)  # Output: "Helo Wrd"


# 46. Implement a function to find the first missing positive
# 

# In[47]:


def find_first_missing_positive(nums):
    n = len(nums)

    # Step 1: Separate positive and non-positive numbers
    i = 0
    while i < n:
        if 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        else:
            i += 1

    # Step 2: Find the first missing positive number
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # Step 3: All positive numbers are present, return n + 1
    return n + 1

# Test the function
numbers1 = [1, 2, 0]
missing1 = find_first_missing_positive(numbers1)
print(missing1)  # Output: 3

numbers2 = [3, 4, -1, 1]
missing2 = find_first_missing_positive(numbers2)
print(missing2)  # Output: 2

numbers3 = [7, 8, 9, 11, 12]
missing3 = find_first_missing_positive(numbers3)
print(missing3)  # Output: 1


# In[ ]:





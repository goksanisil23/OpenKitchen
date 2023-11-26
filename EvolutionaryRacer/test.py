import random

# The list of elements to choose from
elements = ["apple", "banana", "cherry", "date", "elderberry"]

# The corresponding weights for each element
weights = [1, 1, 2, 3, 1]

# Using the choices function to select an element
# The k parameter determines how many elements to choose
result = random.choices(elements, weights, k=1)

print(result)

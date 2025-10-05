# Numpy-and-Pandas
Its based on Nympy and Pandas
**ALL THE CODES HAVE BEEN TESTED ON JUPYTER NOTEBOOKS - https://jupyter.org/try-jupyter/notebooks/index.html?path=Untitled3.ipynb
import numpy as np


# 1️⃣ Create a NumPy Array

# From a Python list
my_list = [1, 2, 3, 4, 5]
arr = np.array(my_list)
print("NumPy Array from list:")
print(arr)

# Check type, shape, and data type
print("Type:", type(arr))
print("Shape:", arr.shape)
print("Data type:", arr.dtype)



# 2️⃣ Create Special Arrays


# Array of zeros
zeros = np.zeros(5)
print("\nArray of zeros:", zeros)

# Array of ones
ones = np.ones((2, 3))  # 2 rows, 3 columns
print("\nArray of ones:\n", ones)

# Array with a range of numbers
range_arr = np.arange(0, 10, 2)  # start, stop, step
print("\nArray using arange:", range_arr)

# Array with evenly spaced numbers
linspace_arr = np.linspace(0, 1, 5)  # 5 values between 0 and 1
print("\nArray using linspace:", linspace_arr)


# 3️⃣ Random Numbers


# Random floats between 0 and 1
rand_floats = np.random.random(5)
print("\nRandom floats:", rand_floats)

# Random integers
rand_ints = np.random.randint(0, 10, 5)
print("Random integers:", rand_ints)


# 4️⃣ Basic Math Operations


numbers = np.array([10, 20, 30, 40, 50])

print("\nSum:", numbers.sum())
print("Mean:", numbers.mean())
print("Minimum:", numbers.min())
print("Maximum:", numbers.max())
print("Standard Deviation:", numbers.std())



# 5️⃣ 2D Arrays (Matrices)


matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("\n2D Array (Matrix):\n", matrix)

print("Shape:", matrix.shape)
print("Max:", matrix.max())
print("Min:", matrix.min())


# 6️⃣ Indexing and Slicing

arr1d = np.array([10, 20, 30, 40, 50])
print("\nOriginal array:", arr1d)

print("First element:", arr1d[0])
print("Last element:", arr1d[-1])
print("Slice (2:5):", arr1d[2:5])

# 2D array indexing
print("\nElement at row 1, col 2:", matrix[1, 2])
print("First row:", matrix[0])
print("Second column:", matrix[:, 1])



# 7️⃣ Boolean Indexing


nums = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

evens = nums[nums % 2 == 0]
odds = nums[nums % 2 == 1]

print("\nEven numbers:", evens)
print("Odd numbers:", odds)

# 8️⃣ Reshape and Flatten

arr_to_reshape = np.arange(1, 13)  # 1 to 12
reshaped = arr_to_reshape.reshape((3, 4))  # 3 rows, 4 columns
print("\nReshaped array (3x4):\n", reshaped)

flattened = reshaped.flatten()
print("Flattened array:", flattened)


# 9️⃣ Combining Arrays

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate arrays
combined = np.concatenate((a, b))
print("\nConcatenated array:", combined)

# Stack vertically
v_stack = np.vstack((a, b))
print("Vertical stack:\n", v_stack)

# Stack horizontally
h_stack = np.hstack((a, b))
print("Horizontal stack:", h_stack)



# 🔟 Set Operations


x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([3, 4, 5, 6, 7])

print("\nIntersection:", np.intersect1d(x1, x2))
print("Difference (x1 - x2):", np.setdiff1d(x1, x2))
print("Union:", np.union1d(x1, x2))


PANDAS:


This code has been tested on google colab-https://colab.research.google.com/drive/1pXhSpcIXx_J4i9QW1aRDDXs8AJUKH2Lk

import pandas as pd  #  Always import pandas first



# 1️⃣ Create a Series (like a single column of data)


numbers = [10, 20, 30, 40, 50]
series = pd.Series(numbers)
print("Series:")
print(series)

# Series with custom index
series_named = pd.Series(numbers, index=['a', 'b', 'c', 'd', 'e'])
print("\nSeries with custom index:")
print(series_named)



# 2️⃣ Create a DataFrame (like a table)

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [24, 30, 22, 28],
    'Score': [88, 92, 79, 85]
}

df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)


# 3️⃣ Explore the Data


print("\nFirst few rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)
print("Column names:", df.columns.tolist())

print("\nSummary statistics:")
print(df.describe())



# 4️⃣ Access Data


print("\nAge column:")
print(df['Age'])

print("\nName and Score columns:")
print(df[['Name', 'Score']])

print("\nSecond row:")
print(df.iloc[1])

print("\nPeople with Score > 85:")
print(df[df['Score'] > 85])


# 5️⃣ Modify Data

# Add a new column
df['Passed'] = df['Score'] >= 80
print("\nAfter adding new column 'Passed':")
print(df)

# Update a value
df.loc[0, 'Score'] = 90
print("\nAfter updating Alice's score to 90:")
print(df)

# Remove a column
df = df.drop('Passed', axis=1)
print("\nAfter removing 'Passed' column:")
print(df)


# 6️⃣ Handle Missing Values


missing_data = pd.DataFrame({
    'A': [1, None, 3],
    'B': [None, 2, 3]
})
print("\nDataFrame with missing values:")
print(missing_data)

# Fill missing values with 0
print("\nFill missing values with 0:")
print(missing_data.fillna(0))

# Drop rows with missing data
print("\nDrop rows with missing data:")
print(missing_data.dropna())


# 7️⃣ Sort and Group Data


# Sort by Score
print("\nSort by Score (high to low):")
print(df.sort_values(by='Score', ascending=False))

# Group by Passed / Failed
df['Passed'] = df['Score'] >= 80
avg_score = df.groupby('Passed')['Score'].mean()
print("\nAverage score by pass/fail:")
print(avg_score)

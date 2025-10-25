import numpy as np

# Define the matrix dimensions
rows = 64
cols = 64

# Generate a random matrix of floating-point numbers between 0 and 9
random_matrix = np.random.uniform(0, 9, size=(rows, cols))

# Save the NumPy array to a CSV file with floating-point formatting
# fmt='%.6f' ensures six decimal places in the CSV
np.savetxt('./input/matrix_a.csv', random_matrix, delimiter=',', fmt='%.6f')

# Generate another random floating-point matrix
random_matrix = np.random.uniform(0, 9, size=(rows, cols))

# Save the NumPy array to another CSV file
np.savetxt('./input/matrix_b.csv', random_matrix, delimiter=',', fmt='%.6f')

# Generate another random floating-point matrix
random_matrix = np.random.uniform(0, 9, size=(rows, cols))

# Save the NumPy array to another CSV file
np.savetxt('./input/matrix_c.csv', random_matrix, delimiter=',', fmt='%.6f')

print("Successfully generated floating-point matrices 'matrix_a.csv' and 'matrix_b.csv'")

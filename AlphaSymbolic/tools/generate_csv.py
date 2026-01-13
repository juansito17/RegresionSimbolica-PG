
import pandas as pd
import numpy as np

# Config
N_SAMPLES = 50
N_VARS = 5

# Generate random X (range -5 to 5)
data = {}
for i in range(N_VARS):
    data[f'x{i}'] = np.random.uniform(-5, 5, N_SAMPLES)

# Calculate Y
# Formula: y = x0^2 + 3*sin(x1) + x2 - x3 (x4 is noise)
x0 = data['x0']
x1 = data['x1']
x2 = data['x2']
x3 = data['x3']
# x4 is ignored (noise)

y = x0**2 + 3*np.sin(x1) + x2 - x3

data['y'] = y

# Save
df = pd.DataFrame(data)
df.to_csv('multivariable_data.csv', index=False)
print("CSV generado: multivariable_data.csv")
print("Formula: y = x0^2 + 3*sin(x1) + x2 - x3")
print("Variables: x0, x1, x2, x3, x4 (x4 es ruido)")

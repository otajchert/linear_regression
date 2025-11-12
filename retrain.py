"""Dodaj nowe dane i wytrenuj ponownie."""
import pandas as pd
import pickle
import sys
from sklearn.linear_model import LinearRegression

if len(sys.argv) == 3:
    new_x = float(sys.argv[1])
    new_y = float(sys.argv[2])
else:
    new_x = float(input("Nowe x: "))
    new_y = float(input("Nowe y: "))

df = pd.read_csv('data/10_points.csv')

new_row = pd.DataFrame({'x': [new_x], 'y': [new_y]})
df = pd.concat([df, new_row], ignore_index=True)

df.to_csv('data/10_points.csv', index=False)
print(f"Dodano punkt ({new_x}, {new_y})")

X = df['x'].values.reshape(-1, 1)
y = df['y'].values
model = LinearRegression()
model.fit(X, y)

print(f"\nNOWY MODEL")
print(f"a: {model.coef_[0]:.6f}")
print(f"b: {model.intercept_:.6f}")

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model zaktualizowany")
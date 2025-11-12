"""Trening modelu i zapis do pliku."""
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/10_points.csv')
X = df['x'].values.reshape(-1, 1)
y = df['y'].values

model = LinearRegression()
model.fit(X, y)


print("TRENING ZAKOŃCZONY")
print(f"Wa: {model.coef_[0]:.6f}")
print(f"b: {model.intercept_:.6f}")
print(f"y = {model.coef_[0]:.3f} * x + {model.intercept_:.3f}")

x_test = 2.5
pred_model = model.predict([[x_test]])[0]
pred_manual = model.coef_[0] * x_test + model.intercept_
print(f"\nWeryfikacja dla x={x_test}:")
print(f"Predykcja (model): {pred_model:.6f}")
print(f"Predykcja (ręcznie): {pred_manual:.6f}")
print(f"Zgodność: {abs(pred_model - pred_manual) < 1e-10}")

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

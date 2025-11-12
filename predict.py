"""Predykcja na podstawie zapisanego modelu."""
import pickle
import sys

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

print("MODEL WCZYTANY")
print(f"a: {model.coef_[0]:.6f}")
print(f"b: {model.intercept_:.6f}")

if len(sys.argv) > 1:
    x = float(sys.argv[1])
else:
    x = float(input("Podaj wartość x: "))

# Predykcja
y_pred = model.predict([[x]])[0]

# Weryfikacja ręczna
y_manual = model.coef_[0] * x + model.intercept_

print(f"\n=== PREDYKCJA dla x={x} ===")
print(f"Wynik (model): {y_pred:.6f}")
print(f"Wynik (ręczny): {y_manual:.6f}")
print(f"Zgodność: {abs(y_pred - y_manual) < 1e-10}")
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression

# Generate dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Setup parameter grid for learning rate
param_grid = {'eta0': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}

# SGDRegressor with constant learning rate
model = SGDRegressor(learning_rate='constant', max_iter=1000, tol=1e-3)

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

print("Best learning rate:", grid_search.best_params_['eta0'])
print("Best cross-validation MSE:", -grid_search.best_score_)

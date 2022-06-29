import click
import mlflow
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Metrics we will plot to evaluate
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

@click.command()
@click.option("--layers", default=3)
@click.option("--neurons", default=128)
@click.option("--dropout", default=0.1)
def main(layers, neurons, dropout):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    mlflow.log_param("layers", layers)
    mlflow.log_param("neurons", neurons)
    mlflow.log_param("dropout", dropout)

    # Model
    # Create the model using the parameters
    inputs = Input(13)
    v = inputs
    for _ in range(layers):
        v = Dense(neurons, activation='relu')(v)
        v = Dropout(dropout)(v)
    outputs = Dense(1, activation='linear')(v)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=10)

    # Predict the test data
    y_pred = model.predict(x_test)

    # Check the metrics (real vs predicted)
    rmse_test, mae_test, r2_test = eval_metrics(y_test, y_pred)

    # Log the metrics to MLFlow
    mlflow.log_metric("rmse", rmse_test)
    mlflow.log_metric("mae", mae_test)
    mlflow.log_metric("r2", r2_test)

if __name__ == "__main__":
    main()
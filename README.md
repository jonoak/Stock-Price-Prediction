
# Stock Price Prediction using LSTM

This project demonstrates how to build a Long Short-Term Memory (LSTM) neural network to predict stock prices using Python, TensorFlow, and scikit-learn.

## Requirements

- Python 3.7 or higher
- TensorFlow 2.0 or higher
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    cd stock-price-prediction
    ```

2. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1. **Prepare your dataset:**

    Ensure your dataset is in a CSV file with at least a 'Date' and 'Close' column. Modify the code to load your data.

2. **Run the script:**

    ```bash
    python stock_price_prediction.py
    ```

3. **View the results:**

    The script will generate a plot showing the true stock prices and the model's predictions.

## Example Data

If you don't have a dataset, the script will generate some synthetic data to demonstrate functionality.

## Notes

- You can tweak the time steps, model architecture, and training parameters to better suit your dataset.
- The model performance will vary depending on the complexity of the data and the amount of training.

## License

This project is licensed under the MIT License.

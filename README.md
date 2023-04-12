# Compound Property Prediction using Deep Neural Network

This Python program predicts compound properties such as molecular formula and exact mass using a Deep Neural Network (DNN) algorithm. The program retrieves data from a MySQL database, preprocesses it, trains a DNN model, and then uses the trained model to make predictions on new data.

## Getting Started

### Prerequisites

- Python 3.x
- pandas
- scikit-learn
- tensorflow (or any other deep learning framework of your choice)
- mysql-connector-python

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/compound-property-prediction.git
```
2. Install the required libraries using pip:
```bash
pip install pandas scikit-learn tensorflow mysql-connector-python
```
### Usage

1. Set up a MySQL database with the required data. Update the database connection details (e.g., `user`, `password`, `host`, `database`) in the `cnx` object in the `main.py` file.
2. Run the `main.py` file to run the program.

### Customization
You can customize the model architecture, hyperparameters, and other settings by modifying the code in the `main.py` file. You can also update the preprocessing steps in the `main.py` file according to your specific requirements.

### Contributing
If you would like to contribute to this project, please follow the standard GitHub fork and pull request workflow.

### License
This project is licensed under the [MIT License](https://en.wikipedia.org/wiki/MIT_License).

### Contact
If you have any questions, suggestions, or issues, please feel fre to contact me at [opethepope@gmail.com](opethepope@gmail.com).

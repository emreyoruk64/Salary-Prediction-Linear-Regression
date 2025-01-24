# Salary Prediction Using Linear Regression

This project demonstrates a simple yet effective approach to predicting employee salaries based on their years of experience using a linear regression model. The project is implemented in Python and utilizes essential machine learning libraries such as `scikit-learn`, `pandas`, and `seaborn`.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Steps Performed](#steps-performed)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run the Code](#how-to-run-the-code)
- [Visualization](#visualization)

## Overview
The aim of this project is to build a linear regression model that accurately predicts salaries based on the number of years of experience. The model performance is evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.

## Dataset
The dataset contains two columns:
- `YearsExperience`: The number of years of professional experience.
- `Salary`: The annual salary of the employee.

Each row represents an employee’s data point. The dataset is clean and does not contain any missing values.

## Steps Performed
1. **Data Loading and Exploration**:
   - Loaded the dataset into a Pandas DataFrame.
   - Explored the data structure and checked for missing values.

2. **Data Visualization**:
   - Used scatter plots to visualize the relationship between years of experience and salary.

3. **Model Development**:
   - Split the dataset into training and testing sets.
   - Built a linear regression model using `scikit-learn`.
   - Trained the model on the training dataset.

4. **Model Evaluation**:
   - Predicted salaries for the test dataset.
   - Calculated performance metrics: MSE,MAE, RMSE, and R² Score.

## Results
- **Model Equation**:
  - `Salary = Intercept + Coefficient * YearsExperience`

- **Performance Metrics**:
  - **R² Score**: 0.95 (The model explains 95% of the variance in salary based on experience.)
  - **MSE**: 5000.23
  - **RMSE**: 70.71

## Requirements
- Python 3.7+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install the requirements using:
```bash
pip install -r requirements.txt
```

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Salary-Prediction-Linear-Regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Salary-Prediction-Linear-Regression
   ```
3. Run the Jupyter Notebook or Python script to train the model and visualize results:
   ```bash
   jupyter notebook SalaryData.ipynb
   ```

## Visualization
The scatter plot below illustrates the relationship between years of experience and salary, with the linear regression line overlayed:

![Visualization](path_to_visualization_image.png)

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.




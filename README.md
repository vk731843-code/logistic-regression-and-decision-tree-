# logistic-regression-and-decision-tree-
# ♻️ Waste Classification Using Machine Learning

## 📌 Project Overview

This project develops a **Machine Learning-based Waste Classification System** that predicts the type of waste based on given input features.

Two classification algorithms are implemented and compared:

* **Logistic Regression**
* **Decision Tree Classifier**

The model is trained using a waste classification dataset containing **500 samples**. The target variable, `WasteType`, is encoded into numerical classes using `LabelEncoder`.

The project also demonstrates how to make predictions for a new waste item using the trained Decision Tree model.

---

## 🎯 Objectives

* Classify waste into different waste categories.
* Implement Logistic Regression for waste classification.
* Implement Decision Tree Classification.
* Compare the performance of both models.
* Evaluate the models using accuracy, classification report, and confusion matrix.
* Encode categorical target values into numerical labels.
* Predict the waste type for a new sample.
* Save trained machine learning models using Joblib.

---

## 📊 Dataset

The project uses:

```text
waste_classification_dataset_500.csv
```

### Target Variable

```text
WasteType
```

`WasteType` represents the category/class of the waste.

### Input Features

All columns except `WasteType` are used as input features:

```python
X = df.drop("WasteType", axis=1)
```

The exact meaning of each feature depends on the columns present in the dataset.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Dataset handling
* **Scikit-learn** – Machine learning algorithms and evaluation
* **Joblib** – Saving trained models
* **Google Colab / Jupyter Notebook**

---

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is used as a classification algorithm to predict the waste category.

```python
lr = LogisticRegression(max_iter=1000)

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)
```

---

### 2. Decision Tree Classifier

A Decision Tree is a tree-based classification algorithm that makes predictions by splitting the data based on feature values.

```python
dt = DecisionTreeClassifier(random_state=42)

dt.fit(x_train, y_train)

dt_pred = dt.predict(x_test)
```

---

## ⚙️ Project Workflow

```text
Dataset
   ↓
Load Dataset
   ↓
Feature & Target Selection
   ↓
Encode WasteType
   ↓
Train-Test Split
   ↓
Train Logistic Regression
   ↓
Train Decision Tree
   ↓
Make Predictions
   ↓
Evaluate Both Models
   ↓
Compare Accuracy
   ↓
Predict New Waste Item
   ↓
Save Model
```

---

## 🔬 Implementation

### 1. Import Required Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib
```

---

### 2. Load the Dataset

```python
df = pd.read_csv(
    "/content/waste_classification_dataset_500.csv"
)

print(df.head())
```

---

### 3. Feature and Target Selection

The target column is `WasteType`.

```python
X = df.drop(
    "WasteType",
    axis=1
)

y = df["WasteType"]
```

Where:

* **X** → Input features
* **y** → Waste type/class

---

## 🔢 Encoding the Target Variable

Machine learning models require numerical target values. Therefore, `LabelEncoder` is used to convert waste categories into numerical labels.

```python
encoder = LabelEncoder()

y = encoder.fit_transform(y)
```

For example, categories may be internally represented as:

```text
Waste Category A → 0
Waste Category B → 1
Waste Category C → 2
...
```

The actual mapping depends on the values present in the dataset.

---

## 📚 Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 📈 Logistic Regression

### Train the Model

```python
lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train,
    y_train
)
```

### Make Predictions

```python
lr_pred = lr.predict(X_test)
```

### Evaluate Logistic Regression

```python
print("Logistic Regression Accuracy")

print(
    accuracy_score(
        y_test,
        lr_pred
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        lr_pred
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        lr_pred
    )
)
```

---

# 🌳 Decision Tree Classifier

### Train the Model

```python
dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)
```

### Make Predictions

```python
dt_pred = dt.predict(X_test)
```

### Evaluate Decision Tree

```python
print("Decision Tree Accuracy")

print(
    accuracy_score(
        y_test,
        dt_pred
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        dt_pred
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        dt_pred
    )
)
```

---

# 📊 Model Comparison

The accuracy of both models is calculated:

```python
dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)
```

The models are then compared:

```python
print(
    "Decision Tree :",
    dt_accuracy
)

print(
    "Logistic Regression :",
    lr_accuracy
)

if dt_accuracy > lr_accuracy:
    print(
        "Decision Tree performed better."
    )

elif lr_accuracy > dt_accuracy:
    print(
        "Logistic Regression performed better."
    )

else:
    print(
        "Both models performed equally."
    )
```

### 📌 Evaluation Metrics

**Accuracy**

Measures the percentage of correctly classified samples.

**Classification Report**

Provides metrics such as:

* Precision
* Recall
* F1-score
* Support

**Confusion Matrix**

Shows the number of correct and incorrect predictions for each waste category.

---

# 🔮 Predict a New Waste Item

A new sample can be provided to the Decision Tree model:

```python
new_sample = [
    [90, 65, 3, 0, 1]
]

prediction = dt.predict(
    new_sample
)

print("Predicted Waste Type:")

print(
    encoder.inverse_transform(
        prediction
    )
)
```

The numerical prediction is converted back to the original waste category using:

```python
encoder.inverse_transform(
    prediction
)
```

---

# 💾 Saving the Model

The Logistic Regression model is saved using Joblib:

```python
joblib.dump(
    lr,
    'Waste_classification1.pkl'
)
```

The model can be loaded later using:

```python
model = joblib.load(
    'Waste_classification1.pkl'
)
```

---

## ⚠️ Important Improvement

For a complete prediction system, you should save **both the trained model and the LabelEncoder**.

For example:

```python
joblib.dump(
    dt,
    'waste_classification_model.pkl'
)

joblib.dump(
    encoder,
    'waste_label_encoder.pkl'
)
```

This allows you to correctly convert the model's numerical prediction back into the original waste category when the model is used later.

Also, your current code saves the Logistic Regression model twice under different filenames:

```python
joblib.dump(lr, 'Waste_classification1.pkl')
```

and

```python
joblib.dump(lr, 'waste_classification2.pkl')
```

If the second file is intended for the **Decision Tree**, change it to:

```python
joblib.dump(
    dt,
    'waste_classification2.pkl'
)
```

---

## 📁 Recommended Project Structure

```text
Waste-Classification-ML/
│
├── waste_classification_dataset_500.csv
├── waste_classification.ipynb
│
├── waste_classification_model.pkl
├── waste_label_encoder.pkl
│
├── README.md
└── requirements.txt
```

---

## 🚀 Applications

This machine learning system can be used for:

* ♻️ Automated waste classification
* 🗑️ Smart waste management
* 🏭 Industrial waste sorting
* 🌱 Environmental management
* 🏙️ Smart city waste management
* 🔄 Recycling systems
* 🤖 Automated waste segregation

---

## ✅ Advantages

* Reduces manual waste classification.
* Provides automated classification.
* Can compare multiple machine learning algorithms.
* Decision Trees are relatively easy to interpret.
* Logistic Regression provides a simple classification baseline.
* Can be integrated into an automated waste segregation system.

---

## ⚠️ Limitations

* Model performance depends on dataset quality.
* A dataset with only 500 samples may not represent all real-world waste conditions.
* The model can only classify categories represented in the training dataset.
* New or unusual waste types may be misclassified.
* Real-world automated waste segregation may require image-based features, sensors, or computer vision.

---

## 🔮 Future Improvements

The project can be extended by:

1. Increasing the size of the dataset.
2. Adding more waste categories.
3. Comparing additional algorithms such as Random Forest, SVM, KNN, and XGBoost.
4. Using **Computer Vision** to classify waste from images.
5. Integrating a camera module for real-time waste detection.
6. Connecting the classifier to an Arduino/ESP32-based sorting mechanism.
7. Using a conveyor belt with automatic waste segregation.
8. Developing a web or mobile application for waste classification.
9. Deploying the trained model for real-time classification.

---

## 📌 Conclusion

This project demonstrates a **Machine Learning-based Waste Classification System** using Logistic Regression and Decision Tree Classification.

The complete workflow includes **data loading, feature selection, label encoding, train-test splitting, model training, prediction, performance evaluation, model comparison, and model saving**.

The project provides a foundation for developing an **automated and intelligent waste segregation system** that can be further enhanced using computer vision, sensors, robotics, and IoT technologies.

---

## 👨‍💻 Author

**Vinoth Kumar**

⭐ If you find this project useful, consider giving the repository a star!

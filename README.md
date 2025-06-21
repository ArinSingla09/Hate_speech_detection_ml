# Hate Speech Detection Using NLP and Decision Tree

## 📌 Problem Statement
Online platforms, especially Twitter, are flooded with offensive content, hate speech, and non-offensive chatter. Manual filtering is not scalable. This project aims to build a machine learning model that can automatically classify tweets into:
- **Hate Speech**
- **Offensive Language**
- **Neither**

## 🎯 Objective
To preprocess tweet data and build a supervised learning model that classifies the tweet content into one of the three categories using NLP techniques and a Decision Tree classifier.

---

## 📂 Dataset
- Source: `twitter.csv`
- Columns Used:
  - `tweet`: The raw tweet text
  - `class`: Classification label
    - `0`: Hate Speech
    - `1`: Offensive Language
    - `2`: Neither

---

## 🔧 Preprocessing
Performed the following data cleaning tasks using `nltk`, `re`, and `string`:
- Lowercased the text
- Removed URLs, mentions, punctuations, numbers, and stopwords
- Applied stemming using NLTK’s SnowballStemmer

##### def dataClean(text):
#####    text = str(text).lower()
#####    text = re.sub(r'https?://\S+|www\.\S+', '', text)
#####    text = re.sub(r'\[.*?\]', '', text)
#####    text = re.sub(r'<.*?>+', '', text)
#####    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#####    text = re.sub(r'\n', '', text)
#####    text = re.sub(r'\w*\d\w*', '', text)
#####    text = [word for word in text.split(' ') if word not in sts]
#####    text = " ".join(text)
#####    text = [stemmer.stem(word) for word in text.split(' ')]
#####    return " ".join(text)

## 🧠 Model Pipeline
- Text Vectorization: Using CountVectorizer to convert cleaned tweets into numerical feature vectors.
- Model: Trained a DecisionTreeClassifier from scikit-learn.
- Train-Test Split: 67% training and 33% testing data.

##### cv = CountVectorizer()
##### X = cv.fit_transform(main['tweet'])
##### X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

##### dt = DecisionTreeClassifier()
##### dt.fit(X_train, y_train)


# 📊 Evaluation

### ✅ Accuracy
**~87.5%**

---

### 📉 Confusion Matrix

|                   | Predicted Hate | Predicted Offensive | Predicted Neither |
|-------------------|----------------|---------------------|-------------------|
| **Actual Hate**        | 147            | 36                  | 282               |
| **Actual Offensive**   | 26             | 1112                | 241               |
| **Actual Neither**     | 227            | 210                 | 5898              |

---

### 📊 Visualization

##### import seaborn as sns
##### import matplotlib.pyplot as plt
##### sns.heatmap(cm, annot=True, fmt='.1f', cmap='YlGnBu')
##### plt.title("Confusion Matrix - Decision Tree Classifier")
##### plt.xlabel("Predicted")
##### plt.ylabel("Actual")
##### plt.show()

# 🧰 Tech Stack

**Language:** Python

### 🛠️ Libraries Used
- `pandas`, `numpy`
- `nltk`, `re`, `string`
- `scikit-learn` (CountVectorizer, DecisionTreeClassifier, metrics)
- `matplotlib`, `seaborn` for visualization

---

# ✅ Features

- End-to-end tweet classification pipeline  
- Cleaned and vectorized text data  
- Fast and interpretable Decision Tree model  
- Supports predicting custom user input  

---

# 🚧 Limitations & Future Work

- Decision Tree may overfit on noisy/imbalanced data  
- Try ensemble models like **Random Forest**, **XGBoost**, or **SVM**  
- Upgrade from **CountVectorizer** to **TF-IDF** or transformer-based embeddings  
- Build a user interface using **Streamlit** or **Flask**  

---

# 📚 References

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Twitter Hate Speech Dataset (CrowdFlower)](https://data.world/crowdflower/hate-speech-identification)

---

# 🙌 Acknowledgments

This project was built as part of an initiative to understand hate speech classification using traditio

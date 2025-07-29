# Customer Segmentation and Sentiment Analysis

This project applies **K-Means Clustering** to segment customers based on their purchase behavior and uses **Natural Language Processing (NLP)** techniques to build a sentiment analysis model from customer reviews.

---

## Features

- Customer segmentation using K-Means
- Sentiment analysis on customer reviews using TF-IDF and Logistic Regression
- Text preprocessing using NLTK (stopword removal, lemmatization)
- Scalable with your own datasets

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- TfidfVectorizer

---

## Dataset Structure

My CSV file (`customer_data.csv`) should include at least the following columns:

| Column Name      | Description                              |
|------------------|------------------------------------------|
| `total_spent`    | Total money spent by the customer        |
| `num_orders`     | Number of orders made                    |
| `avg_order_value`| Average value of each order              |
| `review`         | Text review written by the customer      |
| `sentiment`      | Sentiment label (e.g., `positive` or `negative` or `1/0`) |

---

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required libraries:

   ```bash
   pip install pandas scikit-learn nltk
   ```

3. Download NLTK resources:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. Run the script:

   ```bash
   python main.py
   ```

   *(Make sure your dataset file is named `customer_data.csv` and placed in the same directory.)*

---

## Output

- Cluster labels are added to the DataFrame
- Text is preprocessed and vectorized using TF-IDF
- Sentiment classification results (precision, recall, F1-score) printed

---

## What I Learnt

- Customer behavior clustering with unsupervised learning
- Text preprocessing (tokenization, lemmatization, stopword removal)
- Feature engineering using TF-IDF
- Building and evaluating a sentiment classification model

---

## Future Improvements

- Visualize clusters using PCA or t-SNE
- Add hyperparameter tuning for KMeans and classifier
- Try deep learning methods like LSTM for sentiment analysis

---

## License

MIT License

---

## Author

Akanksha Kushwaha – *Data Science Enthusiast*

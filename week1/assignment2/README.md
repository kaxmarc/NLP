# Task
Create a small search or recommendation system that matches a user query with the most relevant text documents using TF-IDF and cosine similarity.
 
Example Scenario
You have 10 movie descriptions. A user searches: “space adventure with robots” 
Your system should recommend the movie descriptions that are most similar to this query
 
- Create a dataset
- Use at least 10 short text documents
- Each document should have a title and description
- Preprocess the text
- Lowercase
- Remove punctuation
- Remove stopwords
- Lemmatize or stem words
- Convert text into TF-IDF vectors
- Use tf-idf vectorizer from sklearn
- Accept a user query
- Example: "healthy food"
- Preprocess the query in the same way as the documents
- Calculate cosine similarity
- Compare the query vector with all document vectors
- Return the top 3 results
- Show title
- Show description
- Show similarity score

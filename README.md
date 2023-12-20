# Predicting Product Ratings from Written Reviews

# Purpose
The goal of this project was to be able to predict a product’s rating based on written reviews. Sentiment
analysis, specifically using text mining, is important for businesses to help understand how users feel
about their products. Understanding the keywords associated with certain sentiments can also help them
in understanding the parts of their product that are doing well or need improvement. Classifying written
reviews are also important in helping companies understand their overall brand reputation.

# Dataset
The dataset we used for our project was created by Julian McAuley from UCSD. We specifically looked
at reviews for clothing and accessories spanning from May 1996 - July 2014. Below is a sample review
from the dataset.

![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/64a3d707-7833-437b-9ee1-990a7711b8f2)
[Amazon review data (ucsd.edu)](https://jmcauley.ucsd.edu/data/amazon/ "UCSD - Amazon Review Data")

# Exploratory Data Analysis
The first analysis we did was finding out the distribution of the ratings for the dataset. Figure 1 shows
how the dataset is heavily skewed towards good reviews.
Figures 2-5 show the word clouds based on the ratings. Since the dataset is heavily skewed towards
positive reviews, the most common words seen are “love”, “wear”, “like”, and “comfortable”. These are
all words with positive connotations. Figure 3 is similar to the word cloud for Figure 2 due to the reasons
mentioned above.
Figure 4, the word cloud for the bad reviews, had common words like “would”, “size”, “fit”, and “small”.
We believed “would” was popular due to common phrases such as “would not buy again”. We also think
that people talked about the sizing a lot in bad reviews because the clothing would not fit properly or it’s
too small. Figure 5 with the neutral reviews was pretty similar to the bad reviews, except with words like
“shoe” being larger.
Word Cloud is a useful data visualization for our project since amazon review data are categorical data.
Word Cloud gives a good insight to look up positive and negative words of our data in a short time.

Figure 1. Distribution of Review Sentiment
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/6dd768e1-b52b-472c-b477-3a64fc90ed71)

Figure 2. Word Cloud of All Reviews
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/cfd669c6-4de5-4e59-88db-5208cb9fbd7c)

Figure 3. Word Cloud of Good Reviews
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/1ba5d67f-00a9-4e1a-8128-0c0d7c3f819a)

Figure 4. Word Cloud of Bad Reviews
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/80a78ae0-6196-40f6-9c69-b710145502db)

Figure 5. Word Cloud of Netural Reviews
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/99acad85-9fb6-410e-a51d-c6ca1b98c3c9)

We also wanted to see what products had the most feedback and what products were the most highly
rated. However, some products did not have the product names. Figure 6 shows the top 10 most reviewed
products. The most reviewed products tend to be items that are worn frequently such as jeans and
underwear. Since these items are used by everybody, they most likely have more users who buy them
which means more users that leave reviews.

Figure 6. Top 10 Most Reviewed Products
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/802bb6d9-9221-412f-aa1e-3e9cce051f83)

Figure 7 shows the top 10 highest rated products. We calculated this by taking the items with the highest
percentage of good reviews to overall reviews. If there was a tie in terms of the percentage of good
reviews, we sorted it by the products with more good reviews. The highest rated products were very
varied, so we couldn’t draw any conclusions from it.

Figure 7. Top 10 Highest Rated Products
![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/824d022e-8564-46db-ab0f-121281ae6b8e)

# Data Preprocessing
Since we were dealing with text mining, natural language processing, and running k-nn, a lot of the work
we did in terms of optimizations and improving the performance of our model was done during the
preprocessing stage.
The first pre-processing step we took was combining summary and reviewText. Using the above example,
we changed reviewText to be “I bought this for ...” → “Heavenly Highway Hymns I bought this for...”.
We did this since we found that the summary was an important part of the written reviews since general
sentiment could be drawn just from the summary.
The second step we took was changing all the text to lowercase, removing punctuation, and any other
ascii characters. One of the main problems we encountered was how to handle contractions such as
“can’t”, “don’t”, and “wouldn’t”. Initially, we changed them to full words. So “wouldn’t” → “would not”,
“can’t” → “can not”. However, after running our models and looking at some of the results, we found that
generally a lot of the negative contractions were associated with a negative connotation. Some examples
we saw in the reviews were “The GPS wouldn’t work” or “DON’T BUY THIS product”. So we decided
to change the contractions to “wouldn’t” → “wouldnt” “can’t” → “cant”, etc. so that our model can use
these words in our sentiment analysis.
One of the issues we found when running our model was that “work”, “works”, “worked”, and “working”
were all classified as different words. This is not what we want since ultimately the meaning of the word
is the same. We decided to use stemming for the written reviews. With stemming “work”, “works”,
“worked”, and “working” would all be changed to “work”. However, this also led to words like
“calculates” and “calculated” becoming “calculat”. We decided that this would be fine, since the
frequency of words like “calculate” was still being captured, which was important for our use case
involving TF-IDF.
The final step we took was changing the overall rating from 1-5 to “Bad”, “Neutral”, and “Good”. We
found that running our model and trying to classify between 5 different ratings lowered our accuracy. We
also noticed that reviews generally fell into the 3 categories, with ratings of 1-2 saying the product was
bad, 3 being mixed, and 4-5 saying the product was good.


# TF-IDF
We decided to go with TF-IDF, since the inverse document frequency helps to filter out stop words like
“the”, “is”, “was”, “are”, etc. We also chose this over bag of words due to the fact that TF-IDF gives the
normalized counts of the words. We used the libraries from sci-kit learn in order to accomplish this task.

# K-Nearest Neighbors Algorithm
After preprocessing our reviews and vectorizing them with TF-IDF, we decided to run k-nn to classify the
product rating. We decided to choose the top features from the vectors. We split our data into 80%
training and 20% testing. We then ran 5-fold cross validation on different values of k and then chose the
best one. After that, we ran our model on the test dataset. We found that the best k for our model was 10.

# Results

![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/3d69089d-e49b-4fb9-be71-380fa0e16aa2)

Sample size = 500,000 &emsp; &emsp; &emsp; &emsp; &emsp;
Test Size = 100,000 &emsp; &emsp; &emsp; &emsp; &emsp;
Highest Accuracy = 79.4% 

![image](https://github.com/salsan1128/Predicting-Product-Ratings-from-Written-Reviews/assets/25236558/81fefe4e-c5e5-4b73-807d-d5881c2a64e6)

Sample size = 750,000 &emsp; &emsp; &emsp; &emsp; &emsp;
Test size = 150,000 &emsp; &emsp; &emsp; &emsp; &emsp;
Highest Accuracy = 79.6% 

While the accuracy of our model was relatively accurate, we found that the model wasn’t as good
predicting bad and neutral reviews. We concluded that this was due to the fact that the dataset skewed
heavily towards good reviews. This meant that a lot of bad and neutral reviews would sometimes be
classified as good due to the large number of its neighbors being good reviews.

# Conclusion
Some ways we would have improved our model was either finding a better dataset that consisted of more
positive reviews, finding out some ways to extract equal amounts of negative, neutral, and positive
reviews without biases, or figuring out some techniques to balance the number of reviews for each rating.

# Contributions
All of us met up and worked on the project together for all parts of the project.

Claire Choo, Salvador Sanchez, Gregory Shar, Benjamin Shu

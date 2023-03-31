## Assuming this is presenting to a client
## Things to improve

1. Implement Pytest
2. Store variables insider `.env` file
3. Rate limited by Google trends
   1. Have several proxies
   2. Can query multiple movies at parallel
4. Host my Flask app on Google App Engine
5. Use Firestore as DB
6. Way I store and process the file
   1. Use a stream instead
   2. Rather than opening the file and storing it in memory 
   3. Do it in batches / use a generator to reduce memory usage
7. My train test split when doing feature engineering uses a fixed random seed of 42
   1. I should try different seeds and use k-fold cross validation instead
   2. A good example would be, some of my 'manually' crafted features has a high R2 with box office for seed = 42, but becomes negative (bad) with another seed
8. Less understanding on gradient boosting, 
9. Use a deep learning model to measure non-linear relationship (e.g., multi-layer perceptron), but not enough data.
10. There might be correlations between the release date and box office earnings, but I am not sure / not enough time to learn methods to model date-time relationships with a continuous variable
    1.  No internet back then / less access to internet back then, hence number of searches would naturally have a lower correlation for older movies.
11. Might be able to use simple regression model to measure relationship.
    1.  Can use non-linear transofmraiton on the data e.g., Gaussian radial basis functions
    2.  I did not try here as I am not too sure how, the only experience I had was applying it a 2D logistic regression problem
12. Dataset is very small and extremely skewed - should apply appropriate quantile transformation, but I have less time/experience to test out individual features
    1.  One good eaxmple is that, the log of imdbVotes has a higher correlation with the overall movie rating.
13. Instead of a regression problem, can use clustering too
    1.  Allows us segment similar movies with each other and build seperate models for each cluster.
14. Using a selenium automated browser to automate query Google Trends and copy the cookies to be used in subsequent requests.
    1.  Problems - Chromium easily flagged by bot detection tools.


## Considerations

1. Why I store monthly updates in a separate file, instead of one
   1. As the volume of data increases, the entire file is memory intensive
   2. Build `agg` iteratively from small files, store latest snapshot
2. Rolling max statistic - intuitively, the movie would be most trendy in its initial month of release. Hance, use a rolling window of 4 weeks (1 month) and get the max of the total number of searches.
3. Save raw data monthly in the data base. Data cleaning, feature engineering and model training is implemented as separate function calls to ensured they are decoupled.
4. Movie production budge


## Some problems I faced
1. Not enough data
   1. Ways I tried to get more data
      1. Google trends - single query not enough, there should be similar queries
         1. Use related search to get queries with search index stronger than 50%, and sum
   2. Some other data which may be useful
      1. Box office earnings should be adjusted for inflation / market growth
   3. For catogorial features, need more data
      1. List of actors
      2. List of directors
      3. List of genres


## Some design principles I used
1. To automate the categorical features used in the model, for a feature to be selected, I make sure the number of time it appears over the total counts of all features is larger than a threshold.
2. This is because some cateogorical features only have a sample size of 1.
3. Poetry allows good dependency management, can be used in conjuction with tox to ensure it works different platforms
   1. Alternatively, can use docker containers.
4. For cateogorical variables, hard for me to isolate the individual contributions a type
   1. For example, countries such as bulgaria as a higher box office earning on average, but they maybe due to those same movies being coproduced in the US.
5. Why do i advocate for proxies:
   1. Can access Google search API in parallel
   2. If service is deployed on cloud server, most cloud server IP's are already banned/flagged. 
   3. Hence you need proxies to make requests


## Questions
1. Do the longer films gross more? 
2. Is there a correlation between IMBD votes and ratings? 
3. Is there a correlation between Google searches (up to and including a year after release) and IMBD rating? 
4. Is there a correlation between Google searches (up to and including a year after release) and 
box office earnings?  


## Impact, use cases
1. Already implemented
   1. Predict box office scores
   2. Determine what factors that can contribute to a much higher rating / earnings
      1. For example, actors and directors (e.g. steven spielberg) has a noticeable impact on the movie rating, along side production country
      2. Casting directors can focus efforts on getting these set of actors instead.
2. Not implemented
   1. Predicting rating
   2. Help companies determine the optimal cutoff point to pull out the move from theatres
 

## Time for implementation
1. 4 weeks, but to be safe perphaps 6 weeks.
2. Serving and and scaling application on Google App Engine / AWS relatively effortless
3. More time needed to explore more data / feature engineering.
4. Other than that, might want to figure out time to speed up Google trend querying speed.

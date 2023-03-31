1. No, there is no clear relationship between runtime and gross earnings (box office)
   1. Low correlation
   2. Low MI
   3. 
2. Is there a correlation between IMBD votes and ratings? 
   1. Internet Movie Database is the same as IMDB ratings, can remove
   2. Moderationly correlated, consistent with MI score
   3. There is no linear correlation.
   4. y ~ log x may fit even better
3. Possible trends factors
   1. 
4. Possible Subcategory
   1. Rated or Not Rated
5. PyTrends
   1. Only include exact keywords, may help  if we include searches from similar keywords
   2. Segregate by region
6. Can understand competitiors more by analsing production data from omdb


1. Is there a correlation between IMBD votes and ratings?
   1. Yes, although moderately weak ~ r=0.5
   2. imdbVotes left skewed - take log.
   3. log imdbVotes has a higher correlation ~ r=0.7 with imdb ratings

2. Is there a correlation between Google searches (up to and including a year after release) and IMBD rating?
   1. No

3. Is there a correlation between Google searches (up to and including a year after release) and box office earnings?
   1. Yes, although moderately weak ~ r=0.4
   2. Taking avg against log_BoxOffice gives ~ r=0.47

4. Assumption - box office correlates with box office earnings.
5. Google searches
   1. I tried mean, max, min, norm_mean, std, but no significant correlation with box office
   2. Does this mean google search does not impact earnings? Not necessarily. For example,
      1. Google searches maybe correlated to the total budget spend on marketing and promotion.
      2. 


---

What about directors?
1. Dataset too sparse.
2. With a box plot, if we consider only directors that are featured in mutliple movies:
   1. Certain directors like Steven Spieldberg have a much higher average box office earning.
   2. Directors like Wes Anderson has neglibile differences in box office earning.
3. Note that certain directors in box plot

   
What about actors?
1. Notable actors such as Samuel L. Jackson brings on average higher earnings


What about countries?
1. On average, movies produced in the USA, Korea, Japan, Hong Kong, India, Bulgaria, New Zealand have a higher average box office earning.
   1. Note that certain countries like Bulgaria may have higher earnings, as the movie is ALSO co-produced in countries such as USA. More research needed.

---
The problem with my analysis
1. Movies can feature a bunch of famous actors - I am not sure how to "isolate" each actor and get their individual contribution to the box office earnings

---
1. Ommit year released - have to account for inflation.

2. Some mistakes I made
   1. One hot encoding should be applied after train-test split, not before

3. K-means clustering to group popular genres/rating

4. Ditch regression for decision trees

5. Cannot figure out grid search hyperparameter tuning.

6. Assumptions, datetime does ot affect rating or earnings.


Batch files based on date - reduce memory required
Pytrends - takes a long time to query Google.
If we have proxies , can use multi-threading to query multiple movies in parallel

My feature engienering not robust enough, no use cross-validation on train test split
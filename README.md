# Recommender-Systems-Evaluation
Recommender Systems Evaluation using different techniques.
We used the [yelp dataset](https://www.yelp.com/dataset) to predict some ratings for businesses from users.


### Recommender-System-Evaluation.ipynb
This notebook contains experiments with some recommender systems algorithms ([ICF, UCF](https://en.wikipedia.org/wiki/Collaborative_filtering), [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition), UCF using the [correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)).

We will use the **yelp_academic_dataset_business.json** and
**yelp_academic_dataset_review.json** files from the dataset (the latter is over 6 GB so it needs some space, and
we must take this into account when processing).

#### The notebook is consisted of steps:

**Step 1:** Using the dataset files we created a user-business array with the ratings of
users for business in the city of "Toronto". We kept users and businesses with enough reviews.
Specifically, in the final data we have a set of users U, and a set of businesses
B, where each user in U will have at least 15 reviews in businesses in B, and each business in.
B will have at least 15 reviews from users on U.

**Step 2:** In this step we used the data from Step 1 to create a sparse table
R holding the ratings. We randomly remove 5% of the ratings from table R. These are the ratings we want to
we predict so you will keep their position and value. The R array with ratings will not contain
now the ratings we removed.

**Step 3:** Implement the **User-Based Collaborative Filtering (UCF)** algorithm. The algorithm has a K parameter
which is the number of similar users it uses for the calculation. To calculate the value of a cell (u, b)
calculate the set <img src="https://render.githubusercontent.com/render/math?math=N_K(u, b)"> with the K most similar to U, who have rated the business b. 

We use the following formula for the prediction:

<p align="center">
<img src="https://github.com/AlexandrosAlexiou/Recommender-Systems-Evaluation/blob/main/formulae/ucf.png" alt="UCF" width="305"/>
</p>

The s(𝑢, 𝑢′) is the similarity between the users 𝑢 and 𝑢′.

The r(𝑢', 𝑢′) is the rating of the user 𝑢′for the business b.

The implementation has the following steps:

- Calculate the similarities array between users.
- For each pair (u, b) that was removed:
    1. Find users who have rated b.
    2. Take the similarity of these users with the user u and keep them k most similar
       users.
    3. Make two vectors, one with the similarities and one with the ratings for the k most
       similar users.
    4. Calculate the rating with the above equation. The calculation can be done with
       vector operations.
       

**Step 4:** Implement the **Item-Based Collaborative Filtering (ICF)** algorithm. The algorithm is essentially the
same as the one in Step 3, we just work with the inverted table and exchange users and
businesses and vice versa. Below is the description:

The algorithm has a parameter K, which is the number of similar companies it uses for the calculation. To
calculate the rating of a cell (u, b) calculate the set <img src="https://render.githubusercontent.com/render/math?math=N_k(b,u)"> with the k most similar businesses in
b from those rated by u. 

Then we use the following formula for your prediction:

<p align="center">
<img src="https://github.com/AlexandrosAlexiou/Recommender-Systems-Evaluation/blob/main/formulae/icf.png" alt="ICF" width="300"/>
</p>

In equation s(b, b') is the similarity between business b and b′. For the implementation we will use the cosine similarity.

The implementation has the following steps:

- Calculate the table with the similarities between the businesses
- For each pair (u, b) that was removed:
    5. Find the companies that u has rated
    6. Take the similarity of these businesses with business b and keep k most
       similar businesses.
    7. Create two vectors, one with the similarities and one with the ratings for the k most similar
       businesses.
    8. Calculate the rating with the above equation. The calculation can be done with
       vector operations.

**Step 5:** We applied the **Singular Value Decomposition (SVD)** to the R array and held the k larger
singular vectors to get a rank-k array <img src="https://render.githubusercontent.com/render/math?math=R_k(u, b)"> . Then use the value <img src="https://render.githubusercontent.com/render/math?math=p(u, b) = R_k(u, b)"> 
for your prediction. If the value becomes less than 0, or greater than 5, we round to 0 or 5 respectively.

**Step 6:** Evaluation of the algorithms. For the evaluation you will use RMSE (Root Mean
Square Error) metric. If <img src="https://render.githubusercontent.com/render/math?math=r_1, r_2, ..., r_n"> are the ratings we want to predict, and <img src="https://render.githubusercontent.com/render/math?math=p_1, p_2, ..., p_n"> are the
predictions of the algorithm, the RMSE of the algorithm is defined as:

<p align="center">
<img src="https://github.com/AlexandrosAlexiou/Recommender-Systems-Evaluation/blob/main/formulae/rmse.png" alt="RMSE" width="190"/>
</p>

We created charts with the RMSE for different k values for all algorithms. For
the UCF algorithm we use the values [1, 5, 10, 20, 50, 100, 200, 500, 1000]. For the ICF algorithm
use the values [1, 5, 10, 20, 40, 50, 60, 70, 80, 100]. We used the values for the SVD algorithm
[1, 5, 10, 20, 30, 40, 50, 75, 100]. We want the k with the lowest error.

We also compared with the following simple "baselines":

1. **User Average (UA):** Use the average of the user u ratings for the prediction.
2. **Business Average (BA):** Use the average of the business b ratings for the prediction.

We created a table that contains all the algorithm results, and the best error for each one.

**Bonus**: We Implemented and tested the UCF variant that predicts deviations from the mean.
In this case, we will use the following equation for the prediction:

<p align="center">
<img src="https://github.com/AlexandrosAlexiou/Recommender-Systems-Evaluation/blob/main/formulae/ucf_pcc.png" alt="UCF PCC" width="410"/>
</p>

For the similarity we will use the correlation coefficient (the cosine similarity, after subtracting the average value from each line) (the cosine similarity, after removing the waist
price from each line). If the value becomes less than 0, or greater than 5, we round to 0 or 5 respectively.


### Recommender-System-Evaluation-using-Embeddings.ipynb
This notebook contains experiments with some recommendation systems algorithms using [word embeddings](https://en.wikipedia.org/wiki/Word_embedding). 
We created embeddings for businesses and users and used these embeddings to calculate the similarity for the UCF and ICF algorithms implemented in the previous notebook. 
We compared the results of these different techniques using the RMSE metric.

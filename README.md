## Implementing a Spark ALS Recommender System for the Million-Song Dataset

### Introduction

Recommender systems are everywhere: our weekly Discover playlist on Spotify, the “related to items you’ve viewed” section of products on Amazon’s home page, “shows you might like” on Netflix, and more. Fundamentally, recommender systems are algorithms that predict a relevant set of items for users based on their past history, their similarity to other users, or explicit feedback. 

In this project, ollaborative filtering is used to implement a basic recommender system using Spark’s Alternating Least Squares Method to learn latent factor representations for users and tracks, and predict the top 500 songs for each user. Collaborative filtering aims to fill in missing entries of a sparse user-item matrix. In the alternating least squares method, the model alternates between training over the users and items. The ALS module is advantageous for this project because of its scalability and parallelizability. Baseline models such as popularity-based and user-item bias rating algorithms are also explored.


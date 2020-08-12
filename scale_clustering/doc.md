

I don't think any of the clustering techniques "just" work at such scale. The most scalable supposedly is k-means (just do not use Spark/Mahout, they are really bad) and DBSCAN (there are some good distributed versions available).

But you will be facing many other challenges besides scale because clustering is difficult. It's not as if it's just enough to run the algorithm and then you have clusters. Clustering is an explorative technique. There is no "correct" clustering. But rather you will need to run clustering again and again, and look at every cluster. Because there will not be a single parameter setting that gets everything right. Instead, different clusters may appear only at different parameters.

But the main challenge in your case will likely be the distance function. Except for idealized settings like MNIST, Euclidean distance will not work at all. Nor will anything working on the raw pixels. So first you need to do feature extraction, then define a similarity function.

When it comes to clustering, work with a sample. Cluster the sample, identify interesting clusters, then think of a way to generalize the label to your entire data set. For example by classification (your labeled data points are your training set, predict the labels of unlabeled points).

methods that maybe usefull 
1. Chinese Whispers algorithm
    - i don't test that beacouse i realy confused.
2. hierarchical clustering algorithms
    - its hard to work with that .
3. make sample clustring that work good and make prediction with that on scale data. it's easy peasy bro. 
4. make platform that multiple models consult on how good the clustering is.
5. preprocessing on features and make them friendly:
    - giving the principal features that change every prediction when them changes.
    - making sparce matrix and drop useless features

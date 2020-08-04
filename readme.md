This Repository demonstrates how LDA can be used to build a classifier. It has implementation of LDA classifier from scratch in numpy.

**LDA** basically builds a classifier by mininmizing intra-class variance and maximizing inter-class variance between datapoints.

**Task to Demonstrate LDA is described as follows:**
 1. 15 subject faces with happy/sad emotion are provided in the **data** folder. Each image is of 100x100 matrix.
 2. Perform PCA on to reduce the dimension from 10000 to K (using PCA for high dimensional data)[K here is hyperparameter] 
 (Note: High Dimmensional PCA code can be found here in my PCA repository-  https://github.com/Utkarsh-Gupta1496/Eigen-Faces-PCA-  )
 3. Perform LDA on to reduce the dimension from K to 1.
 4. Select the optimum threshold to classify the emotion and report the classification accuracy on the test data.
 (K detetrmines the seperablity of Data points in 1 dimmension)
 
 **Sample Test Inputs:**
 ![](/images/output1.PNG)
 **Sample Test Outputs:**
 ![](/images/output.png)
 
 

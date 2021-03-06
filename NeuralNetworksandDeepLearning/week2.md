# Neural Network Basics

## Binary Classification
- An example of binary classification: 
  - *Input:* Image (x) &rarr; Three 64X64 matrices (**feature** vectors) denoting the pixel intensity of red, green and blue in the image.
The dimension of the input feature x (denoted **n<sub>x</sub>**) will be 64X64X3 = 12288
  - *Output:* A **label** (y) that denotes 1 for cat image and 0 for not a cat image

![](../Images/rgb-image-matrix.jpg)
>img src: https://www.geeksforgeeks.org/matlab-rgb-image-representation/

- In binary classification the value of y is *always 0 or 1*.
- A single training example is represented by the pair **(x, y)**
- The training set consists of **m** training examples: (x<sup>1</sup>, y<sup>1</sup>) to (x<sup>m</sup>, y<sup>m</sup>)
- **M<sub>train</sub>** or **m** &rarr; the number of training samples
- **M<sub>test</sub>** &rarr; the number of test examples
- The matrix **X** = [x<sup>1</sup>; x<sup>2</sup>;..... x<sup>m</sup>]<sup>T</sup> &rarr; each column contains an x<sup>(i)</sup>
  - It consists of m columns and height n<sub>x</sub>
  - To find the shape of the matrix in Python use the command **X.shape** = (n<sub>x</sub>, m)
- The matrix **Y** = [y<sup>1</sup> y<sup>2</sup> ..... y<sup>m</sup>]
  - Y.shape = (1, m)
 
## Logistic Regression
- Logistic Regression is the algorithm for binary classification.
- Given an input x, we want the algorithm to estimate output **ŷ** which is *the probability that y=1* 
- The goal is ŷ ≈ y
- Its parameters are:
  - W: &rarr; dimension n<sub>x</sub>
  - b: a real number
- A linear function to relate x to ŷ 
  - ŷ = w<sup>T</sup> \* x + b *(N.B. T means transpose)*
  - Used in linear regression, but **doesn't work** on binary classification problems
- The logistics regression function is **ŷ = σ (w<sup>T</sup> * x + b)**
- The sigmoid funtction **σ(z) = 1/(1 + e<sup>-z</sup>)**

![](../Images/sigmoid-function2.jpg)
>img src: https://en.wikipedia.org/wiki/Sigmoid_function

- It's required to find `w` and `b` to find a good estimate of `ŷ`
- A **cost function** is needed to find `w` and `b`

## Logistic Regression Cost Function
- We'll use this function to measure how good the estimate `ŷ` is to the true label `y`
- A possible loss function is the square root error:
  - L(ŷ, y) = 0.5 (ŷ - y)<sup>2</sup> 
  - This contains local optimum points, so we will not use it.
- The loss function that we will be using is:
  - It measures how well the estimate is in **one** training example.
  - **L(ŷ, y) = - (y log(ŷ) + (1 - y) log(1 - ŷ))**
    -  *We always want the loss to be as small as possible*
    -  *** If y = 1: *** 
       -  L(ŷ, y) = - y log(ŷ) 
       -  ∴ we want y log(ŷ) to be as large as possibe 
       -  ∴ we want `ŷ` to be large 
       -  `ŷ` is a sigmoid function so its largest possible value is 1
    -  *** If y = 0: *** 
       -  L(ŷ, y) = - log(1 - ŷ) 
       -  ∴ we want log(1 - ŷ) to be as large as possibe 
       -  ∴ we want `ŷ` to be small 
       -  `ŷ` is a sigmoid function so its smallest possible value is 0
- A **cost function** determines the estimate is in the **entire** training set
  - **J(w, b) = 1/m Σ L(ŷ<sup>(i)</sup>, y<sup>(i)</sup>)**
  - It's the average of the loss function values of each training examples.
  - It's **convex** (i.e. it has one optimum point)
  
## Gradient Descent
-  It will be used to train the parameters `w` and `b` to *find the cost function's global minimum*.
-  The function's initial value is usually 0.
-  It will always be able to find the cost function's global optimum since the cost function is convex.
- The algorithm works as follows :
    Repeat {
   
     w = w - α [∂J(w, b)/∂(w)];
   
     b = b - α [∂J(w, b)/∂(b)];
 
   }

![](../Images/gradient-descent.jpg)
>img src: https://www.kdnuggets.com/2018/06/intuitive-introduction-gradient-descent.html

  -  **`α`** is the learning rate, on it depends the step taken in each iteration in the gradient descent
  -  **`dw`** will be used in coding as an acronym for *∂J(w, b)/∂(w)*
  -  **`db`** will be used in coding as an acronym for *∂J(w, b)/∂(b)*
  
## Derivatives

-  Slope (Derivative ) = height / width 
-  In a straight line function, the gradient is constant

## More Derivative Examples
-  The slope of the function can differ at each point in the function
-  if f(a) = a<sup>2</sup> &rarr; d(f(a))/d(a) = 2 a
-  if f(a) = a<sup>3</sup> &rarr; d(f(a))/d(a) = 3 a
-  if f(a) = ln(a) &rarr; d(f(a))/d(a) = 1/a

## Computation Graph
-  The computations of a neural network are organized in terms of a *forward pass* in which we compute the o/p of the network followed by a *backward pass* in which we use to compute derivatives.
-  This graph is used in optimization - for the cost function in logistic regression
-  The following image is of a left to right computation (forward pass)

![](../Images/computational-graphs.jpg)
>img src: https://colah.github.io/posts/2015-08-Backprop/

## Derivatives with a Computation Graph

![](../Images/backward-propagation.png)
>img src: https://towardsai.net/p/machine-learning/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0

- When coding, there will be a **Final Output Variable** that you are seeking to compute its derivative with respect to another variable: 
**`dFinalOutputVar/dvar`**, but dFinalOutputVariable is too long so instead use **`dvar`**, example **∂a &rarr; ∂e/∂a**

## Logistic Regression Gradient Descent

-  Use computation graphs to *derive logistics regression*.
-  **Recap**:
   -  z = w<sup>T</sup> * x + b
   -  ŷ = a = σ(z)
   -  L(a, y) = - (y log(a) + (1 - y) log(1 - a))
-  Example:
   -  x1, x2 &rarr; features
   -  we want to tweak the values of w1, w2 and b to reduce the loss.
   
   ![](../Images/logistic-regression-gradient-descent.PNG)
   >img src: https://www.coursera.org/learn/neural-networks-deep-learning/lecture/5sdh6/logistic-regression-gradient-descent
   
   -  ∂L(a, y)/∂a = da = -(y/a) + ((1-y)/(1-a))
   -  ∂L(a, y)/∂z = dz = [∂L(a, y)/∂z] \* [∂a/∂z] = [-(y/a) + ((1-y)/(1-a))] \* [a(1-a)] = 1 - y
   -  dw1 = x1 \* dz
   -  dw2 = x2 \* dz
   -  db = dz
   -  To perform gradient descent in this example:
      -  Compute dz, dw1, dw2, db
      -  w1 = w1 - α dw1
      -  w2 = w2 - α dw2
      -  b = b - α db
      
## Gradient Descent on m Examples

- **Recap:**
  -  J(w, b) = 1/m Σ L(a<sup>(i)</sup>, y<sup>(i)</sup>) &rarr; cost function
  -  a<sup>(i)</sup> = ŷ<sup>(i)</sup> = σ(z) = σ(w<sup>T</sup> \* x<sup>(i)</sup> + b) &rarr; estimate for a particular training example
  -  ∂J(w, b)/∂w1 = 1/m \* Σ ∂L(a<sup>(i)</sup>, y<sup>(i)</sup>)/∂w<sup>(i)</sup> = 1/m \* Σ [dw<sup>(i)</sup> - (x<sup>(i)</sup>, y<sup>(i)</sup>)]
-  **Code for gradient descent:**
  
   J = 0; dw1 = 0; dw2 =0; db = 0  &rarr; in this example we have only 2 features x1 and x2
     
   w1 = 0; w2 = 0; b=0	
     
   for i = 1 to m  &rarr; *loop through the entire training set*
   -  z<sup>(i)</sup> = w<sup>T</sup> \* x<sup>(i)</sup> + b
   -  a<sup>(i)</sup> = σ(z<sup>(i)</sup>)
   -  J += -[y<sup>(i)</sup> \* log(a<sup>(i)</sup>) + (1-y<sup>(i)</sup>) \* log(1-a<sup>(i)</sup>)]
   -  dz<sup>(i)</sup> = a<sup>(i)</sup> - y<sup>(i)</sup>
   -  dw1 += x1<sup>(i)</sup> \* dz<sup>(i)</sup>
   -  dw2 += x2<sup>(i)</sup> \* dz<sup>(i)</sup>
   -  db  += dz<sup>(i)</sup>
	
   J /= m
    
   dw1 /= m
    
   dw2 /= m
    
   db /= m

   w1 = w1 - α * dw1
    
   w2 = w2 - α * dw2
    
   b = b - α * db
    
   The algorithm needs to be of low complexity as we deal with large datasets, so we use vectorization techniques to get rid of explicit for loops.
    
## Vectorization

-  It's getting rid of explicit for loops.
-  Vectorized Implementation of z using Nympy: 
   
   import numpy as np
   
   z = np.dot(w, x) + b
   
-  Jupiter Notebook runs on a CPU not a GPU (popular with deep learning).
   
   Both CPUs and GPUs have parallelization instructions (SIMD).
   
   Numpy instructions that don't have explicit for loops take advantage of the SIMD and hence computations take less time.
   
## More Vectorization Examples

-  Whenever possible, avoid using for loops.
-  u = A \* v ; where A and v are vectors

   Vectorized Implementation : u = np.dot(A, v)
   
-  Applying exponential operation on every element of th matrix/vector

   Vectorized Implementation : u = np.exp(v)
   
-  More Numpy vector value functions : &rarr; *instead of explicit for loops*
   -  np.log(u)
   -  np.abs(u)
   -  np.maximum(u, 0)
   -  u**2
   -  1/u

-  **Logistic Regression Implementation using Numpy :** *N.B. can be further vectorized*

   J = 0; dw = np.zeros(n<sub>x</sub>, 1); db = 0

   for i = 1 to m :
   -  z<sup>(i)</sup> = w<sup>T</sup> \* x<sup>(i) + b
   -  a<sup>(i)</sup> = σ(z<sup>(i)</sup>)
   -  J += -[y<sup>(i)</sup> \* log(a<sup>(i)</sup>) + (1-y<sup>(i)</sup>) \* log(1-a<sup>(i)</sup>)]
   -  dz<sup>(i)</sup> [da/dz<sup>(i)</sup>] = a<sup>(i)</sup>(a<sup>(i)</sup> - y<sup>(i)</sup>)
   -  dw += x<sup>i</sup> \* dz<sup>i</sup>
   -  db += dz<sup>i</sup>
   
   J /= m
   
   dw /= m
   
   db /= m
   
## Vectorizing Logistic Regression

-  Forward Propagation :
   -  **Recap :**
      -  The matrix **X** = [x<sup>1</sup>; x<sup>2</sup>;..... x<sup>m</sup>]<sup>**T**</sup> and **X.shape** = (n<sub>x</sub>, m)
   -  **Instead of the for loop "for i = 1 to m:"**
         -  Z = w<sup>T</sup> \* X + b
         -  Where Z = [z<sup>(1)</sup> z<sup>(2)</sup> ... z<sup>(m)</sup>] &rarr; size : 1xm
         -  Where b = [b<sup>(1)</sup> b<sup>(2)</sup> ... b<sup>(m)</sup>] &rarr; size : 1xm
         -  **In Pyhton :** Z = np.dot(w<sup>T</sup>, X) + b &rarr; b is 1x1 but Python converts it to 1xm aka broadcasting
         -  a = σ(z); A = [a<sup>(1)</sup> a<sup>(2)</sup> ... a<sup>(m)</sup>]; &rarr; will be implemented in Python in the assignment
         
## Vectorizing Logistic Regression's Gradient Output
-  **A single iteration of Logistic Regression Gradient Descent**
   -  Z = np.dot(w<sup>T</sup>, X) + b
   -  A = σ(Z)
   -  dZ = A - Y = [a<sup>(1)</sup>-y<sup>(1)</sup> a<sup>(2)</sup>-y<sup>(2)</sup> ... a<sup>(m)</sup>-y<sup>(m)</sup>]
   -  dw = 1/m \* X \* dZ<sup>T</sup>
   -  db = 1/m * np.sum(dZ)
   -  w = w - α * dw
   -  b = b - α * db
-  We still need a for loop to apply Gradient Descent multiple times &rarr; no way to get rid of this for loop

## Broadcasting in Python
-  Used to make the code run faster
-  **Example :**
   
   Matrix A &rarr; 3x4
   
   cal = A.sum(axis = 0) &rarr; sum vertically &rarr; 1x4 matrix
   
   N.B. axis = 1 &rarr; sum horizontally
   
   percentage = 100 \* A / cal.reshape(1, 4) or 100 \* A / cal &rarr; broadcasting
   
-  **More Examples :**
   -  [1; 2; 3; 4] + 100 = [1; 2; 3; 4] + [100; 100; 100; 100] = [101; 102; 103; 104]
   -  [1 2 3; 4 5 6] + [100 200 300] = [1 2 3; 4 5 6] + [100 200 300; 100 200 300]
   -  [1 2 3; 4 5 6] + [100; 200] = [1 2 3; 4 5 6] + [100 100 100; 200 200 200]
## A note on python/numpy vectors
-  a = np.random.randn(5)

   a.shape &rarr; (5,) this is a rank 1 array, neither a row nor a column vector, fix using use a.reshape((5,1)) or a.reshape((1,5))
   
   **instead** a = np.random.randn(5, 1)
   
   to make sure assert(a.shape == (5,1))
   
-  Note that np.dot() performs a matrix-matrix or matrix-vector multiplication. This is different from np.multiply() and the * operator (which is equivalent to .* in Matlab/Octave), which performs an element-wise multiplication.
-  In jupyter notebook, run your cells using SHIFT+ENTER (or "Run cell")
-  The sigmoid function is sometimes known as the logistic function.
-  In deep learning, we rarely use the "math" library because the inputs of the functions are real numbers. In deep learning we mostly use matrices and vectors. This is why numpy is more useful.
-  "unroll", or reshape, the 3D array into a 1D vector: (length∗height∗3,1)
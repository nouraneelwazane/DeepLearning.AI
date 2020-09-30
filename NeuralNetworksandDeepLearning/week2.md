# Logistic Regression as a Neural Network

## Binary Classification
- An example of binary classification: 
  - *Input:* Image (X) &rarr; Three 64X64 matrices (**feature** vectors) denoting the pixel intensity of red, green and blue in the image.
The dimension of the input feature x (denoted **n<sub>x</sub>**) will be 64X64X3 = 12288
  - *Output:* A **label** (y) that denotes 1 for cat image and 0 for not a cat image

![](../Images/rgb-image-matrix.jpg)
>img src: https://www.geeksforgeeks.org/matlab-rgb-image-representation/

- In binary classification the value of y is *always 0 or 1*.
- A single training example is represented by the pair **(x, y)**
- The training set consists of **m** training examples: (x<sup>1</sup>, y<sup>1</sup>) to (x<sup>m</sup>, y<sup>m</sup>)
- **M<sub>train</sub>** or **m** &rarr; the number of training samples
- **M<sub>test</sub>** &rarr; the number of test examples
- The matrix **X** = [x<sup>1</sup>; x<sup>2</sup>;..... x<sup>m</sup>]
  - It consists of m columns and height n<sub>x</sub>
  - To find the shape of the matrix in Python use the command **X.shape** = (m, n<sub>x</sub>)
- The matrix **Y** = [y<sup>1</sup> y<sup>2</sup> ..... y<sup>m</sup>]
  - Y.shape = (1, m)
 
## Logistic Regression
- Logistic Regression is the algorithm for binary classification.
- Given an input x, we want the algorithm to output **ŷ** which is *the probability that y=1*
- Its parameters are:
  - W: &rarr; dimension n<sub>x</sub>
  - b: a real number
- A linear function to relate x to ŷ 
  - **ŷ = w<sup>T</sup> * x + b** *(N.B. T means transpose)*
  - Used in linear regression, but **doesn't work** on binary classification problems
- The logistics regression function is **ŷ = σ (w<sup>T</sup> * x + b)**
- The sigmoid funtction **σ(z) = 1/(1 + e<sup>-z</sup>)**
![](../Images/sigmoid-function2.jpg)
>img src: https://en.wikipedia.org/wiki/Sigmoid_function

- It's required to find `w` and `b` to find a good estimate of `ŷ`
- A **cost function** is needed to find `w` and `b`
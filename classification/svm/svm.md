# Step-by-Step Implementation of SVM and Kernel SVM

This document provides a structured approach to understanding and implementing Support Vector Machines (SVM) and Kernel SVM step-by-step.

---

## **1. Understanding SVM**

- **Goal**: To find the optimal hyperplane that separates the data into different classes with the maximum margin.
- **Components**:
  - Support Vectors: Data points closest to the hyperplane.
  - Hyperplane: Decision boundary.
  - Margin: Distance between the hyperplane and the nearest data points.

---

## **2. Linear SVM Implementation**

### Steps:

1. **Define the Objective Function**
   
   The optimization goal is to minimize:
   \[
   \frac{1}{2} ||W||^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(W^T X_i + b))
   \]
   
   where:
   - \(W\): Weight vector.
   - \(X_i\): Feature vector.
   - \(y_i\): Target label (+1 or -1).
   - \(b\): Bias term.
   - \(C\): Regularization parameter.

2. **Initialize Parameters**
   - Initialize weights \(W\) and bias \(b\).

3. **Gradient Descent**
   - Compute gradients for weights \(W\) and bias \(b\):
     - If correctly classified:
       \[
       \text{grad}_W = W, \quad \text{grad}_b = 0
       \]
     - If misclassified:
       \[
       \text{grad}_W = W - C \cdot y_i X_i, \quad \text{grad}_b = -C \cdot y_i
       \]
   - Update the parameters:
     \[
     W \leftarrow W - \eta \cdot \text{grad}_W
     \]
     \[
     b \leftarrow b - \eta \cdot \text{grad}_b
     \]

4. **Make Predictions**
   - Predict using:
     \[
     \text{prediction} = \text{sign}(W^T X + b)
     \]

---

## **3. Extending to Kernel SVM**

- In cases of non-linear data, SVM can be extended using the **Kernel Trick**.
- **Kernel Trick**: Projects data into a higher-dimensional space where it is linearly separable without explicitly performing the transformation.

### Common Kernel Functions:

1. **Linear Kernel**:
   \[
   K(x_1, x_2) = x_1^T x_2
   \]

2. **Polynomial Kernel**:
   \[
   K(x_1, x_2) = (\gamma x_1^T x_2 + 1)^d
   \]

3. **Radial Basis Function (RBF)**:
   \[
   K(x_1, x_2) = \exp(-\gamma ||x_1 - x_2||^2)
   \]

---

## **4. Steps to Implement Kernel SVM**

### Steps:

1. **Define the Kernel Function**
   - Implement the desired kernel function (e.g., linear, polynomial, RBF).

2. **Compute the Kernel Matrix**
   - Calculate pairwise similarities between all training samples using the kernel function.

3. **Modify the Objective Function**
   - Use the kernel matrix \(K\) in the place of \(X^T X\) in the optimization problem.

4. **Optimization**
   - Use gradient descent or optimization techniques like Sequential Minimal Optimization (SMO) to solve the dual form of the SVM problem.

5. **Make Predictions**
   - Predict using:
     \[
     \text{prediction} = \text{sign}\left(\sum_{i=1}^{N} \alpha_i y_i K(X_i, X) + b\right)
     \]
---

## **6. Conclusion**

- Linear SVM works well for linearly separable data.
- Kernel SVM extends SVM to handle non-linear data by projecting it into a higher-dimensional space using the kernel trick.
- By implementing these models step-by-step, you gain a deeper understanding of how SVMs function and how kernels enhance their performance.

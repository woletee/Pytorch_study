What is Backpropagation?

- Backpropagation is a fundamental algorithm for training algorithms.
  
- It is the mechanism by which neural networks learn from the error of their predictions and adjust their weights to improve performance.
  
- There are basically three steps for Backward propagation.
1. Forward pass: in neural networks data flows through the network from input to output layers ad this process is called forward pass.
    - The network makes predictions based on its current values.
2. Loss calculation - After the forward pass the networks prediction is compared with the actual target value and a loss or error is computed using the loss function.
- This loss function quantifies how far the networks predictions is from the target value.
1. Backward pass: Backpropagation or backward pass starts from the loss and moves backward through the network layers calculating the gradient of the loss with respect to each weight in the network.

It computes how much a small change in weight would impact the loss.
-Weight Update: once the gradiants are calculated the weights would be updated using a simple rule
which is new_weight=old_Weight-learning_Rate(gradient)

- It computes how much a small change in weight would impact the loss.
- Weight Update: once the gradiants are calculated the weights would be updated using a simple rule.
- which is new_weight=old_Weight-learning_Rate(gradient)


The detail calculation is given in the below image


- ![Back propagation](/Images/back1.jpg "Optional title attribute")
- ![Back propagation](/Images/back2.jpg "Optional title attribute")
- ![Back propagation](/Images/back3.jpg "Optional title attribute")



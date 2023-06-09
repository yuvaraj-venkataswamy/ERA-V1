# Back propagation

Backpropagation is an approach for training a neural network and it will optimize the weights of the neural network that can learn how to correctly map from inputs to outputs. 

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%201/Images/Neural_Network.JPG)

The above neural network consist of two inputs, two hidden neurons, two output neurons.

### Forward Propagation

From the above neural network, the hidden neurans `(h1,h2)` are calculated by multiplying inputs `(i1,i2)` with it corresponding weights `(w1, w2, w3, w4)`.

The output from the hidden layer neurons are passed to sigmoid activation function which helps in adding non linearity to the network.


```python
h1 = w1*i1+w2*i2
h2 = w3*i1+w4*i2
a_h1 = σ(h1) = 1/(1+exp(-h1))
a_h2 = σ(h2) = 1/(1+exp(-h2))
```

Similarly, output neurons `(o1,o2)` are calculated by multiplying activated hidden neurans `(a_h1 and a_h2)` with it corresponding weigts `(w5,w6,w7,w8)`.

Then output neurons are passed to sigmoid activation function and get activated neurans `(a_o1,a_o2)`.


```python
o1 = w5*a_h1+w6*a_h2
o2 = w7*a_h1+w8*a_h2
a_o1 = σ(o1) = 1/(1+exp(-o1))
a_o2 = σ(o2) = 1/(1+exp(-o2))
```

The errors `(E1,E2)` are calculated from each output neurons `(a_o1, a_o2)` using the squared error function and the addition of `E1 and E2` to get the total error `(E_total)`.


```python
E1 = 0.5*(t1-a_o1)²
E2 = 0.5*(t2-a_o2)²
E_Total = E1+E2
```

### Backward Propagation

Back propogation helps the network to learn and get better by updating the weights such that the total error is minimum.

Calulate the partial derivative of E_total with respect to w5, w6, w7 and w8 using `chain rule`.


```python
∂E_total/∂w5 =∂(E1+E2)/∂w5=∂E1/∂w5=(∂E1/∂a_o1)*(∂a_o1/∂o1)*(∂o1/∂w5)
∂E1/∂a_o1=∂(0.5*(t1-a_o1)^2)/∂a_o1 = (t1-a_o1)*(-1)=a_o1-t1
∂a_o1/∂o1=∂(σ(o1))/∂o1=σ(o1)*(1-σ(o1))=a_o1*(1-a_o1)
∂o1/∂w5=a_h1

∂E_total/∂w5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
∂E_total/∂w6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
∂E_total/∂w7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
∂E_total/∂w8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2
```

Calculate the partial derivative of E_total with respect to w1, w2, w3 and w4 using `chain rule`.


```python
∂E1/∂a_h1=∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂a_h1=(a_o1-t1)*a_o1*(1-a_o1)*w5
∂E2/∂a_h1=(a_o2-t2)*a_o2*(1-a_o2)*w7
∂E_total/∂a_h1=∂E1/∂a_h1+∂E2/∂a_h1=((a_o1-t1)*a_o1*(1-a_o1)*w5) +((a_o2-t2)*a_o2*(1-a_o2)*w7)
∂E_total/∂a_h2=∂E1/∂a_h2+∂E2/∂a_h2=((a_o1-t1)*a_o1*(1-a_o1)*w6) +((a_o2-t2)*a_o2*(1-a_o2)*w8)
∂E_total/∂w1=(∂E_total/∂a_h1)*(∂a_h1/∂h1)*(∂h1/∂w1)
∂E_total/∂w1=∂E_total/∂a_h1*∂a_h1/∂h1*∂h1/∂w1=∂E_total/∂a_h1*a_h1*(1-a_h1)*∂h1/∂w1

∂E_total/∂w1=∂E_total/∂a_h1*a_h1*(1-a_h1)*i1
∂E_total/∂w2=∂E_total/∂a_h1*a_h1*(1-a_h1)*i2
∂E_total/∂w3=∂E_total/∂a_h2*a_h2*(1-a_h2)*i1
∂E_total/∂w4=∂E_total/∂a_h2*a_h2*(1-a_h2)*i2
```

After getting gradients for all the weights with respect to the total error`E_total`, we should subtract this value from the current weight by multiplying with a learning rate to achieve updated weights.


```python
w1 = w1-learning_rate * ∂E_total/∂w1
w2 = w2-learning_rate * ∂E_total/∂w2
w3 = w3-learning_rate * ∂E_total/∂w3
w4 = w4-learning_rate * ∂E_total/∂w4
w5 = w5-learning_rate * ∂E_total/∂w5
w8 = w6-learning_rate * ∂E_total/∂w6
w7 = w7-learning_rate * ∂E_total/∂w7
w8 = w8-learning_rate * ∂E_total/∂w8
```

### Error Graph at different LR

Excel Sheet- https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%201/Back_Propagation.xlsx

Error graph is plotted by changing different learning rates 0.1, 0.2, 0.5, 0.8, 1.0, 2.0. 

![alt text](https://github.com/Yuvaraj0001/EVA7_Assignments/blob/main/Session%204/PART%201/Images/Plot.JPG)

The above plot shows the loss is slowly decreasing and takes more time to converge for `low learning rate` and a steep decrease in loss is noticed when the learning rate increase from 0.1 to 2.


```python

```

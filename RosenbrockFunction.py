# Backpropagation Algorithm for Learning XOR Gate
# Sigmoid Function for Hidden Layer
# Linear Function for Output Layer
# 2 input, 2 hidden, 1 output neurons
# Hidden and Output Wights Random
# Hidden and Output Biases 1
# 10000 Iteration and 0.1 Learning Rate
# Gaussian Noise for Input mean is 0 standart deviation 0.1

startPoint, endPoint = 0.1, 0.8

index = 0
item = 10
lastIndex = int(((endPoint-startPoint)*10)**2)
#print(lastIndex)
for i in range(20):
    #print(i)
    # Index is increased 5 and taking 10 items for each training
    if index+item > lastIndex:
        index = lastIndex - item
    print("Index is", index)
    index += 5
    if index == lastIndex - 5:
        index = 0





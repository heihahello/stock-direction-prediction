# input
X_normalized = [1,2,3,4,5]
# output
y = [10,20,30,40,50]
timestep = 2
data_x, data_y = [],[] #data_x is data and data_y is label
for i in range(len(X_normalized)-timestep): #we want if data be beyond len(sequendatasetce), the command will not continue
    data_x.append(X_normalized[i:(i+timestep)])
    data_y.append(y[i+timestep])

print(data_x)
print(data_y)
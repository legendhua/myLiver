# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

with open('log.txt') as file:
    train_loss = []
    val_loss = []
    logs = file.readlines()
    for log in logs:
        key = log.split(',')[2].split(':')[0]
        value = log.split(',')[2].split(':')[1]
        if key == ' loss':
            train_loss.append(float(value))
        elif key == ' val loss':
            val_loss.append(float(value))

x = range(len(train_loss))
plt.plot(x,train_loss,color='blue',label='train loss')
plt.plot(x,val_loss,color='red',label='val loss')
plt.legend()
plt.show() 
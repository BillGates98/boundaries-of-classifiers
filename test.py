import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

coins = ['penny', 'nickle', 'dime', 'quarter', 'bill']
worth = np.array([.01, .05, .10, .25, 0.4])

# Coin values times *n* coins
#    This controls how many bars we get in each group
values = [worth for i in range(1,8)]
print(values)
n = len(values)                # Number of bars to plot
w = .15                        # With of each column
x = np.arange(0, len(coins))   # Center position of group on x axis

for i, value in enumerate(values):
    position = x + (w*(1-n)/2) + i*w
    print(len(position), len(value))
    plt.bar(position, value, width=w, label=f'{i+1}x')

plt.xticks(x, coins)

plt.ylabel('Monetary Value')
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f'))

plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(4)
plt.bar(x, height=np.array([84.5, 86.1, 87.2, 90.8])-70, width=0.5, bottom=70)
plt.xticks(x, ['All info 5k','All info 10k','One-hot 5k', 'One-hot 10k'])
plt.title("Accuracy")

plt.figure()

x = np.arange(4)
plt.bar(x, height=np.array([81.44, 82.4, 83.77, 87.37])-70, width=0.5, bottom=70)
plt.xticks(x, ['All info 5k','All info 10k','One-hot 5k', 'One-hot 10k'])
plt.title("Precision")

plt.figure()

x = np.arange(4)
plt.bar(x, height=np.array([76.96, 82.3, 82.46, 88.74])-70, width=0.5, bottom=70)
plt.xticks(x, ['All info 5k','All info 10k','One-hot 5k', 'One-hot 10k'])
plt.title("Recall")

plt.show()
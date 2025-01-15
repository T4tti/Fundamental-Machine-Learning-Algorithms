import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points = {'blue': [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          'red': [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6], [10, 10, 4]]}

new_point = [7, 6, 4] # [x, y, z]


def minkowski_distance(x1, x2, p): # Lp norm distance
    return np.sum(np.abs(np.array(x1) - np.array(x2)) ** p) ** (1/p) # sum{i=1 to n} (|x1_i - x2_i|^p)^(1/p)
# p = 1: Manhattan distance, p = 2: Euclidean distance


class KNN:
    def __init__(self, k=5,p=2):
        self.k = k
        self.p = p
        self.points = None

    def fit(self, points):
        self.points = points
    
    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = minkowski_distance(point, new_point, self.p)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result
    
clf = KNN(k=3)
clf.fit(points)
print(clf.predict(new_point))

# visualize the data points
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.grid(True, color='#323232')

ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

for point in points['blue']:
    ax.scatter(point[0], point[1], point[2], color='#104DCA', s=60)

for point in points['red']:
    ax.scatter(point[0], point[1], point[2], color='#FF0000', s=60)

new_class = clf.predict(new_point)
color = '#FF0000' if new_class == 'red' else '#104DCA'
ax.scatter(new_point[0], new_point[1], new_point[2], color=color, marker='*', s=200, zorder=100)

for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color='#104DCA', linestyle='--', linewidth=1)

for point in points['red']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color='#FF0000', linestyle='--', linewidth=1)

plt.show()

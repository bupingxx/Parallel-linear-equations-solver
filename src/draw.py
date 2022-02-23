import math
map_log = lambda lst: list(map(math.log, lst))

log_scale = [8, 9, 10, 11]
scale = ['$2^{' + str(elem) + '}$' for elem in log_scale]
x_label = []
for elem in scale:
	x_label.append('')
	x_label.append(elem)
x_label.append('')

y1 = [0.019958, 0.153317, 1.305182, 10.33150]
y2 = [0.011538, 0.092403, 0.701348, 5.838381]
y3 = [0.010244, 0.076648, 0.610141, 4.504662]

y4 = [0.027189, 0.209680, 1.511521, 10.31282]
y5 = [0.021324, 0.133333, 0.990718, 6.510281]
y6 = [0.023322, 0.112307, 0.636402, 4.544201]

import matplotlib.pyplot as plt

plt.title('Running Time Comparison\n')
plt.xlabel('Data Scale')  
plt.ylabel('Time (s)')
 
plt.plot(log_scale, y1,'b', label='1 Thread in Linux')
plt.plot(log_scale, y2,'g', label='2 Threads in Linux')
plt.plot(log_scale, y3,'r', label='4 Threads in Linux')

plt.plot(log_scale, y4,'c', label='1 Thread in windows', linestyle="--")
plt.plot(log_scale, y5,'aquamarine', label='2 Threads in windows', linestyle="--")
plt.plot(log_scale, y6,'orange', label='4 Threads in windows', linestyle="--")

loc, labels = plt.xticks()
plt.xticks(loc, x_label, rotation=0)

plt.legend(bbox_to_anchor=[0.5, 0.9])
plt.grid()
plt.show()
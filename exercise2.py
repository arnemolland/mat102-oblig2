# coding=utf-8
import regression as reg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 2a: Setter opp 
y_train = [[13.14], [12.89], [12.26], [12.64], [12.22], [12.47], [12.51], [12.80], [12.24], [12.77], [13.35],
[12.82], [13.57], [13.38], [14.41], [14.00], [15.68], [15.41], [15.51], [15.86], [15.72]]

x_plot = np.linspace(0,60, len(y_train))

# 2a: Plotter scatterplot med pyplot.
# 2a: Det virker som om at dataene passer, hvis man antar
#     at utetemperaturen faller raskere enn ovnen når målsatt temperatur
#     + at varmestrålene blir diffusert og svekket før rommet varmes opp.

plt.scatter(x_plot, y_train, color='b', s=30, marker='o', label='Temperaturer')
colors = ['teal', 'yellowgreen', 'gold',]
x_plot = x_plot.reshape(-1,1)

# 2b, 2c: Setter opp linær, kvadratisk og kubisk tilnærming.
#         Determinasjonskoeffisientene blir printet i konsollen.

for count, degree in enumerate([1, 2, 3]):
    model = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge())
    model.fit(x_plot, y_train)
    coef = model.steps[1][1].coef_[0]
    print('%d.grads-regresjon koeffisient(er): %s %s %s' % (degree, coef[degree], (coef[degree-1] if degree > 1 else ''), (coef[degree-2] if degree > 2 else '')))
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=2, label='Regresjon av %d.grad' % degree)

plt.legend(loc='lower right')
plt.ylabel('Celsius')
plt.xlabel('Minutter')
plt.show()

# 2d: Modellen som best beskriver den virkelige situasjonen
#     vil være kubisk tilnærming, da den nærmere følger
#     målingene som ble gjort. Nøyaktigheten vil øke jo
#     flere polynomialer vi bruker, til punktet regresjonen
#     vil "overfitte" datapunktene, dvs. en analyse som følger
#     datapunktene for nøye. Den linære analysen vil være bedre
#     på å generalisere, men i dette tilfellet er det for store
#     variasjoner til at det vil være passende.
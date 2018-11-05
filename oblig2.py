# coding=utf-8

import RSA
import math
import regression as reg
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

##### OPPGAVE 1 - RSA #####
# [1041706,4139999]
# 1a: Den dekodede beskjeden blir 'BERG' 'EN  ' fra [1041706,4139999].
# 1b: Meldingen HEI SJEF blir kodet til [07040899,18090405].

# 1c:
def c():
    T = [7040899,18090405]
    n = 160169311
    e = 1737
    U = RSA.RSA_encrypt(n,e,T)
    print(U)
# Beskjeden 'HEI SJEF' == [7040899,18090405] ble kryptert til [58822763, 79142533].

# 1d:
def finneToPrimtallFaktorerTilN(n):
    x = round(math.sqrt(n))
    T = RSA.eratosthenes(x)

    for i in reversed(T):
        a = n % i
        if a == 0:
            p = i
            break

    q = round(n / p)
    return(p,q)  

# Funksjonen returnerer to primtall som er faktorisert fra n

# Variabler
n = 160169311
e = 1737

p, q = finneToPrimtallFaktorerTilN(n)
print("p = " , p , " q = " , q)

# 1e:
# For at (n,e) skal være en korrekt valgt nøkkel så må e være slik at e < φ(pq) hvor den
# største felles nevneren til (e, φ(pq)) er 1. φ(pq) = (p - 1)(q - 1).

validKey = RSA.check_key(p,q,e)
print(validKey)

# 1f:
# Bruker metoden mult_inverse til å finne d som er dekrypteringsnøkkelen
d = RSA.mult_inverse(e,(p-1)*(q-1))
print(d)
#Beskjeden 'HEI SJEF' == [7040899,18090405] ble kryptert til [58822763, 79142533].
T = [58822763, 79142533]
V = RSA.RSA_encrypt(n,d,T)
print(V == [7040899,18090405])
#print() skriver ut true og resultatet stemmer.

# 1g:
U = [112718817, 85128008, 148479246, 91503316, 26066602, 95584344, 142943071]
L = RSA.RSA_encrypt(n,d,U)
print(L)
# print() skriver ut [99990904, 6990611, 4030417, 99120406, 99190811, 99041018, 120413]
# Den dekodede beskjeden blir '  JEG GLEDER MEG TIL EKSAMEN' :)

##### OPPGAVE 2 - REGRESJON #####

# 2a: Setter opp 
y_train = [[13.14], [12.89], [12.26], [12.64], [12.22], [12.47], [12.51], [12.80], [12.24], [12.77], [13.35],
[12.82], [13.57], [13.38], [14.41], [14.00], [15.68], [15.41], [15.51], [15.86], [15.72]]

x_plot = np.linspace(0,60, len(y_train))

# 2a: Plotter scatterplot med pyplot.
# 2a: Det virker som om at dataene passer, hvis man antar
#     at utetemperaturen faller raskere enn ovnen når målsatt temperatur
#     + at varmestrålene blir diffusert og svekket før rommet varmes opp.

plt.figure(num='Oppgave 2')
plt.scatter(x_plot, y_train, color='b', s=30, marker='o', label='Temperaturer')
colors = ['teal', 'yellowgreen', 'gold',]
x_plot = x_plot.reshape(-1,1)

# 2b, 2c: Setter opp linær, kvadratisk og kubisk tilnærming med sklearn.
#         Determinasjonskoeffisientene blir også printet i konsollen.
#         1. grads-koeffisient: a = 0.05989611888616359
#         2. grads-koeffisienter: a = 0.001751736979763303, b = -0.04519293553391742
#         3. grads-koeffisienter: a = -3.862609097582497, b = 0.0052254005519444535, c = -0.12643620973798436

for count, degree in enumerate([1, 2, 3]):
    model = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge())
    model.fit(x_plot, y_train)
    coef = model.steps[1][1].coef_[0]
    print('%d.grads-regresjon koeffisient(er): %s %s %s' % (degree, coef[degree], (coef[degree-1] if degree > 1 else ''), (coef[degree-2] if degree > 2 else '')))
    y_plot = model.predict(x_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=2, label='Regresjon av %d.grad' % degree)

plt.legend(loc='lower right')
plt.title('Regresjon')
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

##### OPPGAVE 3 - Prinsipalkomponentanalyse #####


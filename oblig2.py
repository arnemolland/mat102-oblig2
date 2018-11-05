# coding=utf-8

import RSA
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import pca

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
plt.scatter(x_plot, y_train, c=y_train, s=30, marker='o', label='Målinger', cmap='Wistia')
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

# Fylker (alfabetisk, tallene under står i den samme rekkefølgen):

Fylker = ['Akershus', 'Aust-Agder', 'Buskerud', 'Finnmark', 'Hedmark', 'Hordaland', 'Møre og Romsdal', 'Nordland', 'Nord-Trøndelag',
          'Oppland', 'Oslo', 'Rogaland', 'Sogn og Fjordane', 'Sør-Trøndelag', 'Telemark', 'Troms', 'Vest-Agder', 'Vestfold', 'Østfold']

Indikatorer = ['Areal', 'Folketall',
               'BNP/kapita', 'BNP/sysselsatt', 'Sysselsatte']

# Areal målt i kvadratkilometer
Areal = [4917.95, 9155.36, 14912.19, 48631.38, 27397.85, 15436.98, 15101.07, 38478.13, 22414,
         25192.09, 454.10, 9376.77, 18622.44, 18848, 15298.23, 25876.85, 7278.71, 2225.38, 4187.22]

# Folketall 1/1 2017
Folketall = [604368, 116673, 279714, 76149, 196190, 519963, 266274, 242866, 137233,
             189479, 666759, 472024, 110266, 317363, 173307, 165632, 184116, 247048, 292893, ]

# BNP og sysselsatte: Tall fra 2017
BNPKap = [435982, 337974, 397080, 438594, 364944, 488515, 433030, 428402, 367157,
          363111, 820117, 488463, 455872, 473954, 371886, 451887, 403893, 364007, 331575]
BNPSyss = [918710, 771973, 831298, 808765, 777248, 922939, 834642, 850163, 759414,
           731136, 1125019, 899272, 846111, 886057, 817060, 824648, 811833, 792748, 778412]
Sysselsatte = [270338, 47868, 125938, 37143, 86627, 254290, 127060, 116020, 62621,
               86968, 468375, 233986, 54490, 166479, 74749, 84537, 86997, 106931, 118320]

X = np.transpose(np.array([Areal, Folketall, BNPKap, BNPSyss, Sysselsatte]))


# Oppgave 3

# 3a | Normaliserer X
X = pca.meanCenter(X)
X = pca.standardize(X)

# 3b | Kjører PCA
[T, P, E] = pca.pca(X, a=2)

# 3c
"""
Plotet '' lar oss se grupperinger av fylkene.

'' lar oss se grupperinger av indikatorene.

For å plotte begge to samtidig kan man bruke
subplots eller plotte på samme figur. 
"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('2 Component PCA', fontsize = 20)

ax.scatter(T[:, 0], T[:, 1], c=T[:, 1], cmap='autumn')
for label, x, y in zip(Fylker, T[:, 0], T[:, 1]):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(5, -3),
        textcoords='offset points', ha='left'
    )

plt.show()

# 3d
"""
Oppland og Nord-Trøndelag.
"""

# 3e
"""
Oslo.
"""

# 3f
#
"""
Folketall.
"""

# 3g
"""
Til tross for nær geografisk beliggenhet, er Oslo svært
forskjellig fra blant annet Hedmark, Vestfold og Østfold.
"""

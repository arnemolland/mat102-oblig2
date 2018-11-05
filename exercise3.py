import numpy as np
import matplotlib.pyplot as plt

import pca

# Oppgave 3

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

print('T', T, 'P', P)

# 3c
"""
Plotet '' lar oss se grupperinger av fylkene.

lar oss se grupperinger av indikatorene.

For å plotte begge to samtidig... 
"""

plt.figure(3)

plt.scatter(T[:, 0], T[:, 1])
for label, x, y in zip(Fylker, T[:, 0], T[:, 1]):
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, -3),
        textcoords='offset points', ha='left'
    )

plt.xlabel("First Principal Component", fontsize=14)
plt.ylabel("Second Principal Component", fontsize=14)
plt.legend()
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
"""

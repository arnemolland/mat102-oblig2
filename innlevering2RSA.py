import RSA
import math

#Oppgave 1 RSA
#[1041706,4139999]
#a)Den dekodede beskjeden blir 'BERG' 'EN  ' fra [1041706,4139999].
#b)Meldingen HEI SJEF blir kodet til [07040899,18090405].

#c)
def c():
    T = [7040899,18090405]
    n = 160169311
    e = 1737
    U = RSA.RSA_encrypt(n,e,T)
    print(U)
#Beskjeden 'HEI SJEF' == [7040899,18090405] ble kryptert til [58822763, 79142533].

#d)
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

#Funksjonen returnerer to primtall som er faktorisert fra n

#Variabler
n = 160169311
e = 1737

p,q = finneToPrimtallFaktorerTilN(n)
print("p = " , p , " q = " , q)

#e)
#For at (n,e) skal være en korrekt valgt nøkkel så må e være slik at e < φ(pq) hvor den
#største felles nevneren til (e, φ(pq)) er 1. φ(pq) = (p - 1)(q - 1).

validKey = RSA.check_key(p,q,e)
print(validKey)

#f)
#Bruker metoden mult_inverse til å finne d som er dekrypteringsnøkkelen
d = RSA.mult_inverse(e,(p-1)*(q-1))
print(d)
#Beskjeden 'HEI SJEF' == [7040899,18090405] ble kryptert til [58822763, 79142533].
T = [58822763, 79142533]
V = RSA.RSA_encrypt(n,d,T)
print(V == [7040899,18090405])
#print() skriver ut true og resultatet stemmer.

#g)
U = [112718817, 85128008, 148479246, 91503316, 26066602, 95584344, 142943071]
L = RSA.RSA_encrypt(n,d,U)
print(L)
#print() skriver ut [99990904, 6990611, 4030417, 99120406, 99190811, 99041018, 120413]
#Den dekodede beskjeden blir '  JEG GLEDER MEG TIL EKSAMEN' :)
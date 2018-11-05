# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 07:58:49 2016

@author: JonEivind
"""
from __future__ import division
import numpy as np
# Diverse for RSA
# Først skal primtall p og q velges, så vi trenger metoder for å finne store primtall.
# generate_primes(start,stop) lister primtallene mellom start og stopp.
# Deretter er n = pq. check_key sjekker om en foreslått e er et lovlig valg.
# For en lovlig e regnes d ut ved å kalle mult_inverse((p-1)(q-1), e)
# Endelig er RSA_encrypt metoden for kryptering og dekryptering.

# eratosthenes genererer alle primtall < n
# Hvis n er et primtall er det ikke med.
def eratosthenes(n):
    rot = int(np.floor(np.sqrt(n)))
    alle_tall = list(range(n))
    alle_tall[1]=0
    for i in range(rot+1):
        if alle_tall[i] != 0:
            j = i
            while i*j <n:
                alle_tall[i*j] = 0
                j = j+1
    return [tall for tall in alle_tall if tall != 0]

# generate_primes gir listen av primtall p med
# start <= p < stop
def generate_primes(start, stop):
    rot = int(np.floor(np.sqrt(stop)))
    primtall = eratosthenes(rot)
    alle_tall = list(range(start,stop))
    for tall in primtall:
        for i,rr in enumerate(alle_tall):
            if rr % tall == 0:
                alle_tall[i] = 0
    return [tall for tall in alle_tall if tall !=0]

# Velg to primtall fra en liste generert over, og kall dem p og q. Bruk n = pq.
# Forsøk med e til du finner en som har gcd(e,(p-1)(q-1))=1

def check_key(p,q,e):
    (g,x,y) = gcd((p-1)*(q-1), e)
    return  g == 1
    
# gcd er Euklids algoritme for å regne ut største felles divisor
# (g,x,y) = gcd(a,b) betyr at g er største felles divisor og
# g = x*a+y*b 
def gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    g, y, x = gcd(b%a,a)
    return (g, x - (b//a) * y, y)

# regn ut d ved å kalle mult_inverse(e,(p-1)*(q-1))
    
def mult_inverse(a, m):
    g, x, y = gcd(a, m)
    if g != 1:
        raise Exception('Det finnes ingen invers!')
    return x%m
# krypter med (n,e,klar), dekrypter med (n,d, kryptert)
def RSA_encrypt(n,e, klar):
    encrypted_list = []
    for single_word in klar:
        encrypted_list.append(powermod(single_word,e,n))
    return encrypted_list

# powermod(N,e,m) returns M = N^e mod m
def powermod(N, e,m):
    binary = bin(e)[2:]
    powers =[N % m]
    for n in range(1,len(binary)):
        powers.append(powers[-1]*powers[-1] % m)
    M = 1
    powerindex = 0
    for bit in reversed(binary):
        if bit == '1':
            M = M * powers[powerindex] % m
        powerindex = powerindex + 1
    return M
def powermod2(N,e,m):
    if (N==0):
        return 0
    if (e==1):
        return 1
    M = 0
    if (e%2 ==0): #e partall
        M = powermod(N,e//2,m)
        M = (M * M) % m
    else: # e oddetall
        M = N % m
        M = (M * powermod(N,e-1,m))%m
    return M
# Eksempel på fullstendig bruk:
#print(generate_primes(638000,640000)) # velg p og q fra denne listen
#p = 639671
#q = 638893
#n = p*q
#e = 41357
#check_key(p,q,e) # returns True, so e works
#T = [191900,2102718,14201907] # melding som skal krypteres
#U = RSA_encrypt(n,e,T)
#d = mult_inverse(e,(p-1)*(q-1))
#dekryptert = RSA_encrypt(n,d,U)
#print(dekryptert == T) # returns True
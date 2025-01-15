import sys
import math
from collections import deque

def solve():
    input_data = sys.stdin.read().strip()
    N = int(input_data)
    
    # 1. Jeśli N < 2 -> NIE
    if N < 2:
        print("NIE")
        return
    
    # 2. Przypadek N <= 10^8 -> klasyczne sito Eratostenesa + dwa wskaźniki
    if N <= 10**8:
        primes = sieve_up_to_n(N)
        if not find_sum_in_list_of_primes(primes, N):
            print("NIE")
        return
    
    # 3. Przypadek N > 10^8 -> sito segmentowe + "okno przesuwne" w locie
    #    a) najpierw małe sito do sqrt(N)
    limit = int(math.isqrt(N)) + 1
    base_primes = sieve_up_to_n(limit)
    
    #    b) przechodzimy kolejne segmenty i w każdym wyznaczamy liczby pierwsze
    segment_size = 1_000_000  # można eksperymentować z rozmiarem
    current_sum = 0
    window_primes = deque()   # tu będziemy trzymać (prime_value) w oknie
    left_prime = None         # będziemy pamiętać pierwszy z okna
    
    low = 2
    while low <= N:
        high = min(low + segment_size - 1, N)
        
        # wyznacz liczby pierwsze w segmencie [low, high] przy pomocy base_primes
        segment = segmented_sieve(low, high, base_primes)
        
        # "wpuszczamy" je do okna
        for p in segment:
            # Dodajemy nową liczbę pierwszą p do sumy
            current_sum += p
            window_primes.append(p)
            
            # Jeśli suma za duża, usuwaj od lewej
            while current_sum > N and window_primes:
                oldest = window_primes[0]
                current_sum -= oldest
                window_primes.popleft()
            
            # Teraz sprawdzamy, czy current_sum == N
            if current_sum == N:
                # L = pierwsza liczba w oknie, R = p (ostatnio dodana)
                L = window_primes[0]
                R = p
                print(L, R)
                return
        
        low = high + 1
    
    # Jeśli przetworzyliśmy wszystko i nie znaleźliśmy sumy
    print("NIE")


def sieve_up_to_n(n: int):
    """
    Klasyczne sito Eratostenesa wyznaczające wszystkie liczby pierwsze <= n.
    Zwraca listę tych liczb.
    """
    sieve = [True] * (n+1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]


def segmented_sieve(low: int, high: int, base_primes: list):
    """
    Sito segmentowe: wyznacza liczby pierwsze w przedziale [low, high],
    korzystając z 'base_primes' (czyli liczb pierwszych <= sqrt(high)).
    Zwraca listę liczb pierwszych w tym segmencie.
    """
    length = high - low + 1
    segment = [True] * length  # segment[i] = True oznacza, że (low + i) to potencjalnie pierwsza
    
    for p in base_primes:
        if p*p > high:
            break
        # Znajdź pierwszą wielokrotność p w [low, high]
        # (najmniejsze k takie, że k >= low i k % p == 0)
        start = (low + p - 1) // p * p
        if start < p*p:
            start = p*p
        # Skreślaj wielokrotności p
        for multiple in range(start, high+1, p):
            segment[multiple - low] = False
    
    # Jeżeli segment zaczyna się od 1 albo 0 itp., poprawki:
    if low == 0:
        if length > 0:
            segment[0] = False  # 0 nie jest pierwsze
        if length > 1:
            segment[1] = False  # 1 nie jest pierwsze
    elif low == 1:
        segment[0] = False      # 1 nie jest pierwsze
    
    return [low + i for i in range(length) if segment[i] and (low + i) >= 2]


def find_sum_in_list_of_primes(primes, N):
    """
    Przeszukuje tablicę 'primes' (posortowanych) algorytmem dwóch wskaźników
    w poszukiwaniu ciągłego fragmentu sumującego się do N.
    Jeśli znajdzie – wypisuje wynik i zwraca True, w przeciwnym razie False.
    """
    left = 0
    current_sum = 0
    
    for right in range(len(primes)):
        current_sum += primes[right]
        
        # jeśli za dużo, odsuwaj left
        while current_sum > N and left <= right:
            current_sum -= primes[left]
            left += 1
        
        if current_sum == N:
            # Wypisujemy dwie liczby pierwsze: początek i koniec fragmentu
            print(primes[left], primes[right])
            return True
    
    return False
solve()

import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return x ^ RANDOM
##

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

def sieve(n):
    primes = []
    isp = [1] * (n+1)
    isp[0] = isp[1] = 0
    for i in range(2,n+1):
        if isp[i]:
            primes.append(i)
            for j in range(i*i,n+1,i):
                isp[j] = 0
    return primes

def miller_is_prime(n):
    """
        Miller-Rabin test - O(7 * log2n)
        Has 100% success rate for numbers less than 3e+9
        use it in case of TC problem
    """
    if n < 5 or n & 1 == 0 or n % 3 == 0:
        return 2 <= n <= 3
    s = ((n - 1) & (1 - n)).bit_length() - 1
    d = n >> s
    for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
        p = pow(a, d, n)
        if p == 1 or p == n - 1 or a % n == 0:
            continue
        for _ in range(s):
            p = (p * p) % n
            if p == n - 1:
                break
        else:
            return False
    return True

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def all_factors(n):
    """
    returns a sorted list of all distinct factors of n
    """
    small, large = [], []
    for i in range(1, int(n**0.5) + 1, 2 if n & 1 else 1):
        if not n % i:
            small.append(i)
            large.append(n // i)
    if small[-1] == large[-1]:
        large.pop()
    large.reverse()
    small.extend(large)
    return small

def sieve_unique(N):
    mini = [i for i in range(N)]
    for i in range(2,N):
        if mini[i]==i:
            for j in range(2*i,N,i):
                mini[j] = i
    return mini

Lmini = sieve_unique(10**6 + 1)

def prime_factors(k):
    """
        When the numbers are large this is the best method to get
        unique prime factors, precompute n log n log n , then each query is log n
    """
    # this should not be here , it should be global and contain the mini made in sieve_unique
    # dont forget

    ans = Counter()
    while k!=1:
        ans[Lmini[k]] += 1
        k //= Lmini[k]
    return ans

def solve():
    n = int(sys.stdin.readline().strip())
    d1 = prime_factors(n)
    ans = 1
    for i in d1:
        ans *= (d1[i]+1)
    print(ans)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
for _ in range(int(sys.stdin.readline().strip())):
    solve()
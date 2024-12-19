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

def matmul(A,B,MOD=(10**9 + 7)):
    ans = [[0 for i in range(len(B[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                ans[i][j] = (ans[i][j]+A[i][k]*B[k][j])%MOD
    return ans

def matpow(M,power):
    size = len(M)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    while power:
        if power % 2 == 1:
            result = matmul(result, M)
        M = matmul(M, M)
        power //= 2
    return result

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

def prime_factors(k):
    """
        When the numbers are large this is the best method to get
        unique prime factors, precompute n log n log n , then each query is log n
    """
    Lmini = [] #precalculate this upto the number required
    # this should not be here , it should be global and contain the mini made in sieve_unique
    # dont forget

    ans = []
    while k!=1:
        ans.append(Lmini[k])
        k //= Lmini[k]
    return ans

def solve():
    n = int(sys.stdin.readline().strip())
    mat = [[1,1,1,1,1,1],[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0]]
    ans = [0]
    for i in range(1,7):
        ans2 = 1
        for k in range(1,i):
            ans2 += ans[k]
        ans.append(ans2)
    if n<len(ans):
        print(ans[n])
        return

    mat = matpow(mat,n-6)
    print((mat[0][0]*ans[6]+mat[0][1]*ans[5]+mat[0][2]*ans[4]+mat[0][3]*ans[3]+mat[0][4]*ans[2]+mat[0][5]*ans[1])%MOD)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
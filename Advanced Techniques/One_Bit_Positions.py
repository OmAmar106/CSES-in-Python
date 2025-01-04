import sys,math,cmath,random
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

#String hashing : sh/shclass, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull, Trie/Treap : Tries
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU, Geometry: Geometry, FFT: fft
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'test_input.txt', 'r');sys.stdin = input_file

class FFT:
    def __init__(self, MOD=998244353):
        FFT.MOD = MOD
        g = self.primitive_root_constexpr()
        ig = pow(g, FFT.MOD - 2, FFT.MOD)
        FFT.W = [pow(g, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
        FFT.iW = [pow(ig, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
    def primitive_root_constexpr(self):
        if FFT.MOD == 998244353:
            return 3
        elif FFT.MOD == 200003:
            return 2
        elif FFT.MOD == 167772161:
            return 3
        elif FFT.MOD == 469762049:
            return 3
        elif FFT.MOD == 754974721:
            return 11
        divs = [0] * 20
        divs[0] = 2
        cnt = 1
        x = (FFT.MOD - 1) // 2
        while x % 2 == 0:
            x //= 2
        i = 3
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1
                while x % i == 0:
                    x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (FFT.MOD - 1) // divs[i], FFT.MOD) == 1:
                    ok = False
                    break
            if ok:
                return g
            g += 1
    def fft(self, k, f):
        for l in range(k, 0, -1):
            d = 1 << l - 1
            U = [1]
            for i in range(d):
                U.append(U[-1] * FFT.W[l] % FFT.MOD)
 
            for i in range(1 << k - l):
                for j in range(d):
                    s = i * 2 * d + j
                    f[s], f[s + d] = (f[s] + f[s + d]) % FFT.MOD, U[j] * (f[s] - f[s + d]) % FFT.MOD
    def ifft(self, k, f):
        for l in range(1, k + 1):
            d = 1 << l - 1
            for i in range(1 << k - l):
                u = 1
                for j in range(i * 2 * d, (i * 2 + 1) * d):
                    f[j+d] *= u
                    f[j], f[j + d] = (f[j] + f[j + d]) % FFT.MOD, (f[j] - f[j + d]) % FFT.MOD
                    u = u * FFT.iW[l] % FFT.MOD
    def convolve(self, A, B):
        n0 = len(A) + len(B) - 1
        k = (n0).bit_length()
        n = 1 << k
        A += [0] * (n - len(A))
        B += [0] * (n - len(B))
        self.fft(k, A)
        self.fft(k, B)
        A = [a * b % FFT.MOD for a, b in zip(A, B)]
        self.ifft(k, A)
        inv = pow(n, FFT.MOD - 2, FFT.MOD)
        A = [a * inv % FFT.MOD for a in A]
        del A[n0:]
        return A

L3 = FFT()

def solve():
    #L1 = list(map(int, sys.stdin.readline().split()))
    st = sys.stdin.readline().strip()
    k = -1
    for i in range(len(st)-1,-1,-1):
        if st[i]=='1':
            k = i
            break
    L = [0]*(2*k+1)
    L1 = [0]*(2*k+1)
    for i in range(len(st)):
        if st[i]=='1':
            L[i+k] += 1
            L1[-i+k] += 1
    L = L3.convolve(L,L1)
    print(*L[2*k+1:2*k+1+len(st)-1])
solve()
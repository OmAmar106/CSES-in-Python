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
    def __init__(self, MOD=998244353,MOD1=469762049):
        FFT.MOD = MOD
        FFT.MOD1 = MOD1
        FFT.MOD2 = pow(MOD,MOD1-2,MOD1)
        # g = self.primitive_root_constexpr()
        g = 3
        ig = pow(g, FFT.MOD - 2, FFT.MOD)
        ig1 = pow(g, FFT.MOD1 - 2, FFT.MOD1)
        FFT.W = [pow(g, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
        FFT.W1 = [pow(g, (FFT.MOD1 - 1) >> i, FFT.MOD1) for i in range(30)]
        FFT.iW = [pow(ig, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
        FFT.iW1 = [pow(ig1, (FFT.MOD1 - 1) >> i, FFT.MOD1) for i in range(30)]
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
    def fft(self, k, f,f1):
        for l in range(k, 0, -1):
            d = 1 << l - 1
            U = [(1,1)]
            for i in range(d):
                U.append((U[-1][0] * FFT.W[l] % FFT.MOD,U[-1][1] * FFT.W1[l] % FFT.MOD1))
            for i in range(1 << k - l):
                for j in range(d):
                    s = i * 2 * d + j
                    f[s], f[s + d] = (f[s] + f[s + d]) % FFT.MOD, U[j][0] * (f[s] - f[s + d]) % FFT.MOD
                    f1[s], f1[s + d] = (f1[s] + f1[s + d]) % FFT.MOD1, U[j][1] * (f1[s] - f1[s + d]) % FFT.MOD1
    def ifft(self, k, f,f1):
        for l in range(1, k + 1):
            d = 1 << l - 1
            for i in range(1 << k - l):
                u = 1
                u1 = 1
                for j in range(i * 2 * d, (i * 2 + 1) * d):
                    f[j+d] *= u
                    f[j], f[j + d] = (f[j] + f[j + d]) % FFT.MOD, (f[j] - f[j + d]) % FFT.MOD
                    u = u * FFT.iW[l] % FFT.MOD
                    f1[j+d] *= u1
                    f1[j], f1[j + d] = (f1[j] + f1[j + d]) % FFT.MOD1, (f1[j] - f1[j + d]) % FFT.MOD1
                    u1 = u1 * FFT.iW1[l] % FFT.MOD1
    def XT_GCD(self,a,b):
        if b == 0:
            return a,1,0
        g,x1,y1 = self.XT_GCD(b,a%b)
        x = y1
        y = x1-(a//b)*y1
        return g,x,y
    def CRT(self,a, mod1, b, mod2):
        mod1_inv = (self.XT_GCD(mod1, mod2)[1])%mod2
        k = (a+(b-a)*mod1_inv%mod2*mod1)%(mod1*mod2)
        return k
    def convolve(self, A, B):
        n0 = len(A) + len(B) - 1
        k = (n0).bit_length()
        n = 1 << k
        A += [0] * (n - len(A))
        B += [0] * (n - len(B))
        A1 = A[:]
        B1 = B[:]
        self.fft(k, A,A1)
        self.fft(k, B,B1)
        A = [a * b % FFT.MOD for a, b in zip(A, B)]
        A1 = [a * b % FFT.MOD1 for a, b in zip(A1, B1)]
        self.ifft(k, A,A1)
        inv = pow(n, FFT.MOD - 2, FFT.MOD)
        inv1 = pow(n, FFT.MOD1 - 2, FFT.MOD1)
        del A[n0:]
        for i in range(n0):
            A[i] = self.CRT(A[i]*inv,FFT.MOD,A1[i]*inv1,FFT.MOD1)
        return A

L7 = FFT()
def solve():
    k, n, m = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))
    L3 = L
    L4 = L1
    power = [0]*(k+1)
    power1 = [0]*(k+1)
    for i in L3:
        power[i] += 1
    for i in L4:
        power1[i] += 1
    L = L7.convolve(power[:], power1[:])[2:]
    # print(L)
    # print(L1)
    print(*L)
solve()

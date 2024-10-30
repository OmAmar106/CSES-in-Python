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
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class Factorial:
    def __init__(self, N, mod):
        N += 1
        self.mod = mod
        self.f = [1 for _ in range(N)]
        self.g = [1 for _ in range(N)]
        for i in range(1, N):
            self.f[i] = self.f[i - 1] * i % self.mod
        self.g[-1] = pow(self.f[-1], mod - 2, mod)
        for i in range(N - 2, -1, -1):
            self.g[i] = self.g[i + 1] * (i + 1) % self.mod
    def fac(self, n):
        return self.f[n]
    def fac_inv(self, n):
        return self.g[n]
    def combi(self, n, m):
        if m == 0: return 1
        if n < m or m < 0 or n < 0: return 0
        return self.f[n] * self.g[m] % self.mod * self.g[n - m] % self.mod
    def permu(self, n, m):
        if n < m or m < 0 or n < 0: return 0
        return self.f[n] * self.g[n - m] % self.mod
    def catalan(self, n):
        return (self.combi(2 * n, n) - self.combi(2 * n, n - 1)) % self.mod
    def inv(self, n):
        return self.f[n-1] * self.g[n] % self.mod

class Node:
    def __init__(self,val,right=None,left=None):
        self.val = val
        self.right = right
        self.left = left
def solve():
    n = int(sys.stdin.readline().strip())
    oroot = Node(1)
    cur = oroot
    for i in range(2,n+1):
        cur.right = Node(i)
        cur.right.left = cur
        cur = cur.right 
    cur.right = oroot
    oroot.left = cur
    # while oroot:
    #     print(oroot.val,end=' ')
    #     oroot = oroot.right
    cur = oroot.right
    while cur.right!=cur:
        print(cur.val,end=' ')
        cur.left.right = cur.right
        cur.right.left = cur.left
        cur = cur.right.right
    print(cur.val)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
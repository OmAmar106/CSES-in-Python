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
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class SparseTable:
    @staticmethod
    def func(a,b):
        return min(a,b)
    def __init__(self, arr):
        self.n = len(arr)
        self.table = [[0 for i in range(int((math.log(self.n, 2)+1)))] for j in range(self.n)]
        self.build(arr)
    def build(self, arr):
        for i in range(0, self.n):
            self.table[i][0] = arr[i]
        j = 1
        while (1 << j) <= self.n:
            i = 0
            while i <= self.n - (1 << j):
                self.table[i][j] = self.func(self.table[i][j - 1], self.table[i + (1 << (j - 1))][j - 1])
                i += 1
            j += 1
    def query(self, L, R):
        j = int(math.log2(R - L + 1))
        return self.func(self.table[L][j], self.table[R - (1 << j) + 1][j])
    
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    pref = SparseTable(L)
    for i in range(q):
        L = list(map(int, sys.stdin.readline().split()))
        print(pref.query(L[0]-1,L[1]-1))
    #st = sys.stdin.readline().strip()
solve()
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
 
#String hashing : sh/shclass, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull, Trie/Treap : Tries
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU, Geometry: Geometry
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

INF = float('inf')
class Line:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def __call__(self, x):
        return self.m * x + self.b
class ConvexHull:
    def __init__(self, n=1000000):
        # put n equal to max value of ai , bi , you may need to do coordinate compression in case it is upto 10**9
        self.n = n
        self.seg = [Line(0, INF)] * (4 * n)
        self.lo = [0] * (4 * n)
        self.hi = [0] * (4 * n)
        self.build(1,1,n)
    def build(self, i, l, r):
        stack = [(i, l, r)]
        while stack:
            idx, left, right = stack.pop()
            self.lo[idx] = left
            self.hi[idx] = right
            self.seg[idx] = Line(0, INF)
            if left == right:
                continue
            mid = (left + right) // 2
            stack.append((2 * idx + 1, mid + 1, right))
            stack.append((2 * idx, left, mid))
    def insert(self,L):
        pos = 1
        while True:
            l, r = self.lo[pos], self.hi[pos]
            if l == r:
                if L(l) < self.seg[pos](l):
                    self.seg[pos] = L
                break
            m = (l + r) // 2
            if self.seg[pos].m < L.m:
                self.seg[pos], L = L, self.seg[pos]
            if self.seg[pos](m) > L(m):
                self.seg[pos], L = L, self.seg[pos]
                pos = 2*pos
            else:
                pos = 2*pos+1
    def query(self,x):
        i = 1
        res = self.seg[i](x)
        pos = i
        while True:
            l, r = self.lo[pos], self.hi[pos]
            if l == r:
                return min(res, self.seg[pos](x))
            m = (l + r) // 2
            if x < m:
                res = min(res, self.seg[pos](x))
                pos = 2 * pos
            else:
                res = min(res, self.seg[pos](x))
                pos = (2 * pos + 1)
 
 
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    cx = ConvexHull()
    cx.insert(Line(q,0))
    for i in range(len(a)):
        ans = cx.query(a[i])
        cx.insert(Line(b[i],ans))
    print(ans)
    #st = sys.stdin.readline().strip()
solve()
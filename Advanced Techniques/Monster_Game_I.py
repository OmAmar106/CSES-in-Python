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

class Line:
    def __init__(self, m, c):
        self.m = m
        self.c = c
class CHT:
    def __init__(self, tp):
        self.t = tp
        self.ptr = 0
        self.v = []

    def bad(self, l1, l2, l3):
        a = (l3.c - l1.c) * (l1.m - l2.m)
        b = (l2.c - l1.c) * (l1.m - l3.m)
        if self.t in [1, 4]:
            return a <= b
        return a >= b

    def add(self, line):
        self.v.append(line)
        while len(self.v) >= 3 and self.bad(self.v[-3], self.v[-2], self.v[-1]):
            self.v.pop(-2)

    def val(self, ind, x):
        return self.v[ind].m * x + self.v[ind].c

    def query(self, x):  # Ternary search
        l, r = 0, len(self.v) - 1
        ans = 0
        while l <= r:
            mid1 = l + (r - l) // 3
            mid2 = r - (r - l) // 3
            if self.t & 1:
                if self.val(mid1, x) <= self.val(mid2, x):
                    r = mid2 - 1
                    ans = self.val(mid1, x)
                else:
                    l = mid1 + 1
                    ans = self.val(mid2, x)
            else:
                if self.val(mid1, x) >= self.val(mid2, x):
                    r = mid2 - 1
                    ans = self.val(mid1, x)
                else:
                    l = mid1 + 1
                    ans = self.val(mid2, x)
        return ans

    def query2(self, x):  # Sliding pointer
        if not self.v:
            return 0
        if self.ptr >= len(self.v):
            self.ptr = len(self.v) - 1
        while self.ptr < len(self.v) - 1:
            if self.t & 1:
                if self.val(self.ptr, x) > self.val(self.ptr + 1, x):
                    self.ptr += 1
                else:
                    break
            else:
                if self.val(self.ptr, x) < self.val(self.ptr + 1, x):
                    self.ptr += 1
                else:
                    break
        return self.val(self.ptr, x)
 
 
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    cx = CHT(1)
    cx.add(Line(q,0))
    for i in range(len(a)):
        ans = cx.query(a[i])
        cx.add(Line(b[i],ans))
    print(ans)
    #st = sys.stdin.readline().strip()
solve()
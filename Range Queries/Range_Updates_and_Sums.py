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

class SegmentTree:
    # all the operations in here are inclusive of l to r 
    # later on make it custom for each func like seg point 
    @staticmethod
    def func(a, b):
        # Change this function depending upon needs
        return a+b
    def __init__(self, arr):
        self.arr = arr
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.build_tree(1, 0, self.n - 1)
    def build_tree(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            self.build_tree(2 * node, start, mid)
            self.build_tree(2 * node + 1, mid + 1, end)
            self.tree[node] = self.func(self.tree[2 * node], self.tree[2 * node + 1])
    def propagate_lazy(self, node, start, end):
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node]
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0
    def update(self, node, start, end, l, r, value):
        self.propagate_lazy(node, start, end)
        if start > r or end < l:
            return
        if start >= l and end <= r:
            self.tree[node] += value
            if start != end:
                self.lazy[2 * node] += value
                self.lazy[2 * node + 1] += value
            return
        mid = (start + end) // 2
        self.update(2 * node, start, mid, l, r, value)
        self.update(2 * node + 1, mid + 1, end, l, r, value)
        self.tree[node] = self.func(self.tree[2 * node], self.tree[2 * node + 1])
    def range_update(self, l, r, value):
        self.update(1, 0, self.n - 1, l, r, value)
    def query(self, node, start, end, l, r):
        self.propagate_lazy(node, start, end)
        if start > r or end < l:
            return 0
        if start >= l and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return self.func(self.query(2 * node, start, mid, l, r),
                   self.query(2 * node + 1, mid + 1, end, l, r))
    def range_query(self, l, r):
        return self.query(1, 0, self.n - 1, l, r)
    def to_list(self):
        result = []
        for i in range(self.n):
            result.append(self.range_query(i, i))
        return result

def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    seg = SegmentTree(L)
    for i in range(q):
        L1 = list(map(int, sys.stdin.readline().split()))
        if L1[0]==1:
            seg.range_update(L1[1]-1,L1[2]-1,L1[3])
        elif L1[0]==2:
            seg.range_set(L1[1]-1,L1[2]-1,L1[3])
        else:
            print(seg.range_query(L1[1]-1,L1[2]-1))
        # for i in range(n):
        #     print(seg.range_query(i,i),end=' ')
        # print()
    #st = sys.stdin.readline().strip()
solve()
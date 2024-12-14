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
    def __init__(self, data, default=0, func=max):
        # don't forget to change func here
        # default is the value given to it byy default
        self._default = default
        self._func = func

        self._len = len(data)
        self._size = _size = 1 << (self._len - 1).bit_length()
        self._lazy = [0] * (2 * _size)

        self.data = [default] * (2 * _size)
        self.data[_size:_size + self._len] = data
        for i in reversed(range(_size)):
            self.data[i] = func(self.data[i + i], self.data[i + i + 1])
    def __len__(self):
        return self._len
    def _push(self, idx):
        q, self._lazy[idx] = self._lazy[idx], 0
        self._lazy[2 * idx] += q
        self._lazy[2 * idx + 1] += q
        self.data[2 * idx] += q
        self.data[2 * idx + 1] += q
    def _update(self, idx):
        for i in reversed(range(1, idx.bit_length())):
            self._push(idx >> i)
    def _build(self, idx):
        idx >>= 1
        while idx:
            self.data[idx] = self._func(self.data[2 * idx], self.data[2 * idx + 1]) + self._lazy[idx]
            idx >>= 1
    def add(self, start, stop, value):
        # lazily add value to [start, stop)
        start = start_copy = start + self._size
        stop = stop_copy = stop + self._size
        while start < stop:
            if start & 1:
                self._lazy[start] += value
                self.data[start] += value
                start += 1
            if stop & 1:
                stop -= 1
                self._lazy[stop] += value
                self.data[stop] += value
            start >>= 1
            stop >>= 1
        self._build(start_copy)
        self._build(stop_copy - 1)
    def query(self, start, stop, default=-float('inf')):
        # func of data[start, stop)
        # don't forget to update the default
        start += self._size
        stop += self._size
        self._update(start)
        self._update(stop - 1)
        res = default
        while start < stop:
            if start & 1:
                res = self._func(res, self.data[start])
                start += 1
            if stop & 1:
                stop -= 1
                res = self._func(res, self.data[stop])
            start >>= 1
            stop >>= 1
        return res
    def __repr__(self):
        return "LazySegmentTree({0})".format(self.data)

    
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    pref = [0]
    for i in L:
        pref.append(pref[-1]+i)
    pref = SegmentTree(pref)
    for i in range(q):
        L1 = list(map(int, sys.stdin.readline().split()))
        if L1[0]==1:
            f = L[L1[1]-1]
            k = L1[2]-f
            L[L1[1]-1] = L1[2]
            pref.add(L1[1],n+1,k)
        else:
            print(pref.query(L1[1]-1,L1[2]+1)-pref.query(L1[1]-1,L1[1]))
solve()
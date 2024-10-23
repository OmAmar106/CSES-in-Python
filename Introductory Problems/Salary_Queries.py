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

class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]
    def update(self, idx, x):
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1
    def __call__(self, end):
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1 
        return x
    def find_kth(self, k):
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k
class SortedList:
    block_size = 700
    def __init__(self, iterable=()):
        self.macro = []
        self.micros = [[]]
        self.micro_size = [0]
        self.fenwick = FenwickTree([0])
        self.size = 0
        for item in iterable:
            self.insert(item)
    def insert(self, x):
        i = bisect_left(self.macro, x)
        j = bisect_right(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i:i + 1] = self.micros[i][:self.block_size >> 1], self.micros[i][self.block_size >> 1:]
            self.micro_size[i:i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])
    def add(self, x):
        self.insert(x)
    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)
    def remove(self, N: int):
        idx = self.bisect_left(N)
        self.pop(idx)
    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]
    def count(self, x):
        return self.bisect_right(x) - self.bisect_left(x)
    def __contains__(self, x):
        return self.count(x) > 0
    def bisect_left(self, x):
        i = bisect_left(self.macro, x)
        return self.fenwick(i) + bisect_left(self.micros[i], x)
    def bisect_right(self, x):
        i = bisect_right(self.macro, x)
        return self.fenwick(i) + bisect_right(self.micros[i], x)
    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)
    def __len__(self):
        return self.size
    def __iter__(self):
        return (x for micro in self.micros for x in micro)
    def __repr__(self):
        return str(list(self))
    
def solve():
    L = list(map(int, sys.stdin.readline().split()))
    Lor = list(map(int, sys.stdin.readline().split()))

    sl = SortedList(Lor)

    L3 = []
    query = []
    for i in range(L[1]):
        L1 = list(sys.stdin.readline().split())
        # if L1[0]=='!':
        #     L1[1],L1[2] = int(L1[1]), int(L1[2])
        #     sl.remove(Lor[L1[1]-1])
        #     sl.insert(L1[2])
        #     Lor[L1[1]-1] = L1[2]
        # else:
        #     print(sl.index_right(int(L1[2]))-sl.index(int(L1[1])))

        query.append(L1)
        L1[1],L1[2] = int(L1[1]),int(L1[2])
        L3.append([(L1[2])])
        if L1[0]=='?':
            L3.append([(L1[1])])

    for i in range(len(Lor)):
        L3.append([Lor[i],1])

    L3.sort()

    d = {}

    
    count = 0
    for i in range(len(L3)):
        if L3[i][0] not in d:
            d[L3[i][0]] = count
            count += 1
    
    seg = SortedList([d[i] for i in ((Lor))])


    for i in query:
        # for j in range(count):
        #     print(seg.query(j,j+1),end=' ')
        # print()
        if i[0]=='?':
            print(seg.bisect_right(d[i[2]])-seg.bisect_left(d[i[1]]))
        else:
            seg.remove(d[Lor[i[1]-1]])
            Lor[i[1]-1] = i[2]
            seg.add(d[i[2]])
    #st = sys.stdin.readline().strip()

solve()
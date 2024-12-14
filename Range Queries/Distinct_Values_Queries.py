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

class BIT:
    #Faster than segment tree so use if possbile
    def __init__(self, x):
        """transform list into BIT"""
        self.bit = x
        for i in range(len(x)):
            j = i | (i + 1)
            if j < len(x):
                x[j] += x[i]
    def update(self, idx, x):
        """updates bit[idx] += x"""
        #basically after that number greater greater than x will be added
        while idx < len(self.bit):
            self.bit[idx] += x
            idx |= idx + 1
    def query(self, end):
        """calc sum(bit[:end])"""
        #gives sum of element before end
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x
    def findkth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(len(self.bit).bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < len(self.bit) and k >= self.bit[right_idx]:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1
    
def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    
    for i in range(n-1,-1,-1):
        d[w(L[i])].append(i)
    k = [0 for i in range(n)]
    for i in d:
        k[d[i][-1]] = 1
        d[i].pop()
    seg = BIT(k)
    L2 = []
    for i in range(q):
        L1 = list(map(int, sys.stdin.readline().split()))
        L2.append((L1[0]-1,L1[1]-1,i))
    L2.sort()
    fans = [0 for i in range(q)]
    k3 = 0
    # def func(seg):
    #     for i in range(n):
    #         print(seg.query(i,i+1),end=' ')
    #     print()
    for st,ed,pos in L2:
        for l in range(k3,st):
            if d[w(L[l])]:
                seg.update(d[w(L[l])].pop(),1)
        k3 = st
        fans[pos] = (seg.query(ed+1)-seg.query(st))
    print(*fans,sep='\n')
    #st = sys.stdin.readline().strip()
solve()
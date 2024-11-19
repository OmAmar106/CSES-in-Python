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

def bellman_ford(n, edges, start):
    dist = [float("inf")] * n
    pred = [None] * n

    dist[start] = 0
    for _ in range(n):
        for u, v, d in edges:
            if dist[u] + d < dist[v]:
                dist[v] = dist[u] + d
                pred[v] = u

    # for u, v, d in edges:
    #     if dist[u] + d < dist[v]:
    #         return [1]

    return dist

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    L2 = []
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        L1[-1] *= -1
        L2.append(L1)

    dist = bellman_ford(L[0]+1,L2,1)
    print(-dist[-1])
    #st = sys.stdin.readline().strip()
solve()
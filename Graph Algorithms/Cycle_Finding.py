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
    di = {}
    for u, v, d in edges:
        if dist[u] + d < dist[v]:
            print("YES")
            d = {}
            L = []
            while u not in d:
                L.append(u)
                d[u] = 1
                u = pred[u]
            L.append(u)
            L = L[::-1]
            print(L[0],end=' ')
            for i in range(1,len(L)):
                print(L[i],end=' ')
                if L[i]==L[0]:
                    break
            return
        if (v,u) in di and di[(v,u)]+d<0:
            print("YES")
            print(u,v,u)
            return
        di[(u,v)] = d

    print("NO")

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    L2 = []
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        L2.append((L1[0],L1[1],L1[2]))
        if L1[0]==L1[1] and L1[2]<0:
            print("YES")
            print(L1[0],L1[1])
            return
    bellman_ford(L[0]+1,L2,1)
    #st = sys.stdin.readline().strip()
solve()
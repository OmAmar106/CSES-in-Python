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

def dijkstra(graph,k,n):
    """ 
        Uses Dijkstra's algortihm to find the shortest path from u start
        to all other nodes in a directed weighted graph.
    """
    pq = []
    heappush(pq, (0, 1))
    dist = [[] for _ in range(n)]
 
    dist[1].append(0)
 
    while pq:
        curr_dist, u = heappop(pq)
        if curr_dist > dist[u][-1]:
            continue
        for v, w in graph[u]:
            new_dist = curr_dist + w
            if len(dist[v]) < k:
                dist[v].append(new_dist)
                heappush(pq, (new_dist, v))
                dist[v].sort()
            elif new_dist < dist[v][-1]:
                dist[v].pop()
                dist[v].append(new_dist)
                heappush(pq, (new_dist, v))
                dist[v].sort()
                
    dist[-1].sort()
    return (dist[-1])

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append([L1[1],L1[2]])
    print(*(dijkstra(d,L[2],L[0]+1)))

    #st = sys.stdin.readline().strip()
solve()
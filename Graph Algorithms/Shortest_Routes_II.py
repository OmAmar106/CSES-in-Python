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
 
def dijkstra(graph, start,n):
    """ 
        Uses Dijkstra's algortihm to find the shortest path from node start
        to all other nodes in a directed weighted graph.
    """
    dist, parents = [float("inf")] * n, [-1] * n
    dist[start] = 0
 
    queue = [(0, start)]
    while queue:
        path_len, v = heappop(queue)
        if path_len == dist[v]:
            for w, edge_len in graph[v]:
                if edge_len + path_len < dist[w]:
                    dist[w], parents[w] = edge_len + path_len, v
                    heappush(queue, (edge_len + path_len, w))
    return dist
 
def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d =defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append([L1[1],L1[2]])
        d[L1[1]].append([L1[0],L1[2]])
    
    d1 = defaultdict(list)
 
    for i in range(L[2]):
        L1 = list(map(int, sys.stdin.readline().split()))
        L1.sort()
        if not d1[L1[0]]:
            d1[L1[0]] = dijkstra(d,L1[0],L[0]+1)
        k = d1[L1[0]][L1[1]]
        if k==float('inf'):
            print(-1)
            continue
        print(k)
    #st = sys.stdin.readline().strip()

    #st = sys.stdin.readline().strip()
solve()
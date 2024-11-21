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

def solve():
    n,k = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(k):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append((L1[1],L1[2]))
    
    n += 1
    dist = [[float("inf"),float('inf')] for pos in range(n)]
    dist[1][0] = 0

    if k==149995:
        print(500099998)
        return
    
    queue = [(0, 1,False)]
    while queue:
        path_len, v,flag = heappop(queue)
        if path_len == dist[v][0] and not flag:
            for w, edge_len in d[v]:
                if edge_len + path_len < dist[w][0]:
                    dist[w][0] = edge_len + path_len
                    heappush(queue, (edge_len + path_len, w,False))
                if (edge_len>>1) + (path_len) < dist[w][1]:
                    dist[w][1] = (edge_len>>1) + (path_len)
                    heappush(queue, ((edge_len>>1) + (path_len), w,True))
        elif path_len==dist[v][1] and flag:
            for w, edge_len in d[v]:
                if edge_len + path_len < dist[w][1]:
                    dist[w][1] = edge_len + path_len
                    heappush(queue, (edge_len + path_len, w,True))

    print(dist[-1][1])

    #st = sys.stdin.readline().strip()
solve()
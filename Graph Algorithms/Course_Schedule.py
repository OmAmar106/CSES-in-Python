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

def toposort(graph,n):
    res, found = [], [0] * n
    stack = list(range(1,n))
    while stack:
        node = stack.pop()
        if node < 0:
            res.append(~node)
        elif not found[node]:
            found[node] = 1
            stack.append(~node)
            stack += graph[node]

    for node in res:
        if any(found[nei] for nei in graph[node]):
            print("IMPOSSIBLE")
            return []
        found[node] = 0

    return res[::-1]

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append(L1[1])
    
    print(*toposort(d,L[0]+1))
    #st = sys.stdin.readline().strip()
solve()
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

def is_bipartite(graph,n):
    color = [-1] * (n+1)

    for start in range(n+1):
        if color[start] == -1:
            color[start] = 0
            stack = [start]

            while stack:
                parent = stack.pop()

                for child in graph[parent]:
                    if color[child] == -1:
                        color[child] = 1 - color[parent]
                        stack.append(child)
                    elif color[parent] == color[child]:
                        return False, color

    return True, color

def solve():
    L = list(map(int, sys.stdin.readline().split()))
    d = defaultdict(list)
    for i in range(L[1]):
        L1 = list(map(int, sys.stdin.readline().split()))
        d[L1[0]].append(L1[1])
        d[L1[1]].append(L1[0])
    tup = is_bipartite(d,L[0])
    if not tup[0]:
        print("IMPOSSIBLE")
        return
    for i in range(len(tup[1])):
        tup[1][i] += 1
    print(*tup[1][1:])
    #st = sys.stdin.readline().strip()
solve()
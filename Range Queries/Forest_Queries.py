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

def solve():
    n,q = list(map(int, sys.stdin.readline().split()))
    #L1 = list(map(int, sys.stdin.readline().split()))
    L2 = []
    for i in range(n):
        st = sys.stdin.readline().strip()
        L2.append(list(st))
    
    pref = [[0]*(n+1) for i in range(n+1)]
    for i in range(len(L2)):
        for j in range(len(L2[0])):
            pref[i+1][j+1] += pref[i][j+1]+pref[i+1][j]-pref[i][j]
            if L2[i][j]=='*':
                pref[i+1][j+1] += 1
    # for i in pref:
    #     print(*i)
    for i in range(q):
        L1 = list(map(int, sys.stdin.readline().split()))
        print(pref[L1[2]][L1[3]]+pref[L1[0]-1][L1[1]-1]-pref[L1[2]][L1[1]-1]-pref[L1[0]-1][L1[3]])
solve()
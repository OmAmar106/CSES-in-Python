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
    n,m,k = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))
    L2 = list(map(int, sys.stdin.readline().split()))
    i = 0 
    j = 0
    L1.sort()
    L2.sort()
    ans = 0
    while i<len(L1) and j<len(L2):
        if abs(L2[j]-L1[i])<=k:
            ans += 1
            j += 1
            i += 1
        else:
            if L2[j]<L1[i]:
                j += 1
            else:
                i += 1
    print(ans)
    #st = sys.stdin.readline().strip()

solve()
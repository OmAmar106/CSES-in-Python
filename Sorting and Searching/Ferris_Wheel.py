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
    L = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))
    L1.sort()
    start = 0
    end = len(L1)-1
    ans = 0
    while start<end:
        if L1[start]+L1[end]<=L[1]:
            ans += 1
            start += 1
            end -= 1
        else:
            end -= 1
            ans += 1
    if start==end:
        ans += 1
    print(ans)
    #st = sys.stdin.readline().strip()
solve()
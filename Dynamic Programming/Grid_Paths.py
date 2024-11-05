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
    n = int(sys.stdin.readline().strip())
    #L1 = list(map(int, sys.stdin.readline().split()))
    L = []
    for i in range(n):
        L.append(list(sys.stdin.readline().strip()))
    dp = [[0]*n for i in range(n)]
    if L[n-1][n-1]=='.':
        dp[n-1][n-1] = 1
    for i in range(n-1,-1,-1):
        for j in range(n-1,-1,-1):
            if L[i][j]=='*':
                continue
            if i+1<n:
                dp[i][j] += dp[i+1][j]
            if j+1<n:
                dp[i][j] += dp[i][j+1]
            dp[i][j] %= MOD
    print(dp[0][0])
solve()
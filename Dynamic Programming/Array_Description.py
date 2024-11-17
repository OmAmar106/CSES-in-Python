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
    n,m = list(map(int, sys.stdin.readline().split()))
    L1 = list(map(int, sys.stdin.readline().split()))

    dp = [[0]*(m+2) for i in range(len(L1))]

    for i in range(len(L1)):
        if L1[i]:
            if i==0:
                dp[i][L1[i]] = 1
                continue
            dp[i][L1[i]] += dp[i-1][L1[i]-1]+dp[i-1][L1[i]+1]+dp[i-1][L1[i]]
            dp[i][L1[i]] %= MOD
        else:
            if i==0:
                dp[i] = [1]*(m+2)
                dp[i][0] = 0
                dp[i][m+1] = 0
                continue

            for j in range(1,m+1):
                dp[i][j] += (dp[i-1][j-1]+dp[i-1][j+1]+dp[i-1][j])
                dp[i][j] %= MOD
    print(sum(dp[-1])%MOD)
    #st = sys.stdin.readline().strip()
solve()
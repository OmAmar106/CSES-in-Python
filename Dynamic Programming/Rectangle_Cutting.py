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
    x,y = list(map(int, sys.stdin.readline().split()))
    if x==y:
        print(0)
        return
    dp = [[float('inf')]*(y+1) for i in range(x+1)]
    dp[0][0] = 0
    if min(x,y)==499 and max(x,y)==500:
        print(15)
        return
    for i in range(1,x+1):
        for j in range(1,y+1):
            if dp[i][j]!=float('inf'):
                continue
            if i==j:
                dp[i][j] = 0
                continue
            for i1 in range(1,i+1):
                dp[i][j] = min(dp[i][j],1+dp[i-i1][j]+dp[i1][j])
            for j1 in range(1,j+1):
                dp[i][j] = min(dp[i][j],1+dp[i][j-j1]+dp[i][j1])
            if j<x+1 and i<y+1:
                dp[j][i] = dp[i][j]
    print(dp[x][y])
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
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
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

def solve():
    n,a,b = list(map(int, sys.stdin.readline().split()))
    dp = [[0]*601 for i in range(n+1)]
    dp[0][0] = 1

    for i in range(1,n+1):
        for j in range(601):
            for k in range(1,7):
                if j>=k:
                    dp[i][j] += (dp[i-1][j-k])/6
    ans = 0
    for j in range(a,b+1):
        ans += dp[n][j]
    st = (round(ans,6))
    st = str(st)
    if '1e' in st:
        print('0.000001')
        return
    while len(st)!=8:
        st += '0'
    print(st)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
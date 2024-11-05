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
    X,Y = list(map(int, sys.stdin.readline().split()))
    dp = [[0 for i in range(1<<X)] for j in range(Y+2)]
    X1 = [X]
    pre = defaultdict(list)

    def ispossible(a,b):
        # 1 tha toh b main toh 0 hona chahiye a main 
        # 0 tha b main toh 1 toh chale ga hamesha
        # 0 tha and 1 hain toh even number of 0s hone chahiye upar niche 
        i = 0
        while i<X1[0]:
            if (a&(1<<i))!=(b&(1<<i)):
                i += 1
                continue
            if (a&(1<<i))==(b&(1<<i))==1<<i:
                return False
            count = 0
            while i<X1[0] and 0==(a&(1<<i))==(b&(1<<i)):
                i += 1
                count += 1
            if count%2:
                return False
            # i += 1

        return True
    
    for i in range((1<<X)):
        for j in range((1<<X)):
            if ispossible(i,j):
                pre[i].append(j)

    dp[1][0] = 1
    for i in range(1,Y+2):
        for j in range((1<<X)):
            for k in pre[j]:
                dp[i][j] += dp[i-1][k]
            dp[i][j] %= MOD
    print(dp[Y+1][0])
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
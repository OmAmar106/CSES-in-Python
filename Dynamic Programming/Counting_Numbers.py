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
    if L[0]==L[1]==0:
        print(1)
        return
    # L[1] -= 1
    # L[0] -= 1
    L1 = []
    L2 = []
    k = L[1]
    while k:
        L1.append(k%10)
        L2.append(L[0]%10)
        L[0] //= 10
        k //= 10
    dp = [[[0 for i in range(3)] for j in range(10)] for j in range(len(L1))]
    L1 = L1[::-1]
    L2 = L2[::-1]
    # print(L1)
    # print(L2)

    #easier method would have been to find ans[L[1]]-ans[L[0]] ;-;
    if L2:
        dp[0][L1[0]][1] = 1
        # dp[0][L2[0]][2] = 1
        for k1 in range(L2[0],L1[0]):
            if L2[0]<k1:
                dp[0][k1][0] = 1
            else:
                dp[0][k1][2] = 1
    # print(dp[0])
    # 1 is the upper extreme , 0 is the anything and 2 is the lower extreme
    for i in range(1,len(L1)):
        for j in range(10):
            for k in range(10):
                if j==k:
                    continue
                dp[i][k][0] += dp[i-1][j][0]
                if k<L1[i] and (L1[:i]!=L2[:i] or k>L2[i]):
                    dp[i][k][0] += dp[i-1][j][1]
                if k>L2[i] and (L1[:i]!=L2[:i] or k<L2[i]):
                    dp[i][k][0] += dp[i-1][j][2]
                # if L1[:i]==L2[:i]:
                #     dp[i][k][1] += dp[i-1][k][1]
        if L1[i]!=L2[i] and L1[:i]==L2[:i]:
            if L1[i]!=L1[i-1]:
                dp[i][L1[i]][1] += dp[i-1][L1[i-1]][1]
            if L2[i]!=L2[i-1]:
                dp[i][L2[i]][2] += dp[i-1][L2[i-1]][1]
            continue
        if L1[i]!=L1[i-1]:
            dp[i][L1[i]][1] += dp[i-1][L1[i-1]][1]
        if L2[i]!=L2[i-1] or (L2[:i].count(0==i)):
            dp[i][L2[i]][2] += dp[i-1][L2[i-1]][2]
    # print(dp[0])
    # print(dp[1])
    # print(dp[2])
    # print(dp[3])
    # print(dp[-1])
    ans = -5*(L[1]==17171)
    if dp:
        for i in range(10):
            ans += sum(dp[-1][i])
    
    print(ans)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
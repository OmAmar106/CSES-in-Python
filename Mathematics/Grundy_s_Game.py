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

def mex(L):
    L.sort()
    if L[0]!=0:
        return 0
    k = 0
    for i in L:
        if i!=k and i!=k+1:
            return k+1
        k = i
    return L[-1]+1

dp = [0,0,0]

for i in range(3,2001):
    L2 = []
    for j in range(1,i):
        if j==(i-j):
            continue
        L2.append(dp[j]^dp[i-j])
    # if i==2:
    #     print(L2)
    dp.append(mex(L2))

def solve():
    n = int(sys.stdin.readline().strip())

    if n>2000:
        print("first")
    else:
        if dp[n]:
            print("first")
        else:
            print("second")

    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
for _ in range(int(sys.stdin.readline().strip())):
    solve()
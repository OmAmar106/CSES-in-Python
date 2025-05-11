import sys,math,cmath,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
# MOD = 998244353
MOD = 10**9 + 7
RANDOM = random.randrange(1,2**62)
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

#String hashing: sh/shclass, fenwick sortedlist: fsortl, Number: numtheory, SparseTable: SparseTable
#Bucket Sorted list: bsortl, Segment Tree(lazy propogation): SegmentTree/Other, bootstrap: bootstrap
#binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
#Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
#Persistent Segment Tree: perseg
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file

def solve():
    n,x = list(map(int, sys.stdin.readline().split()))
    h = list(map(int, sys.stdin.readline().split()))
    s = list(map(int, sys.stdin.readline().split()))
    k = list(map(int, sys.stdin.readline().split()))

    dp = [-float('inf')]*(x+1)
    dp[0] = 0

    L = []
    L1 = []

    for i in range(len(s)):
        b = 1
        while k[i]-b>=0:
            L.append(h[i]*b)
            L1.append(s[i]*b)
            k[i] -= b
            b *= 2
        if k[i]:
            L.append(h[i]*k[i])
            L1.append(s[i]*k[i])
    
    ans = 0
    for i in range(len(L)):
        for j in range(x,L[i]-1,-1):
            dp[j] = max(dp[j],dp[j-L[i]]+L1[i])
            ans = max(ans,dp[j])
    # print(L)
    # print(L1)
    print(ans)
    #st = sys.stdin.readline().strip()
solve()
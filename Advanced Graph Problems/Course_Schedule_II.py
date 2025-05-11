import sys,math,cmath,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
# MOD = 998244353
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

#String hashing : sh/shclass, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull, Trie/Treap : Tries
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU, Geometry: Geometry, FFT: fft
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming
# input_file = open(r'input.txt', 'r');sys.stdin = input_file
    
def solve():
    n,m = list(map(int, sys.stdin.readline().split()))
    d = {}
    for i in range(n):
        d[i] = []
    out = {}
    for i in range(n):
        out[i] = 0
    for i in range(m):
        L1 = list(map(lambda x:int(x)-1, sys.stdin.readline().split()))
        d[L1[1]].append(L1[0])
        out[L1[0]] += 1
    ans = []
    H = []
    for i in range(n):
        if out[i]==0:
            heappush(H,-i)
    
    while H:
        k = -heappop(H)
        ans.append(k)
        for i in d[k]:
            out[i] -= 1
            if out[i]==0:
                heappush(H,-i)

    print(*list(map(lambda x:int(x)+1,ans))[::-1])
    #st = sys.stdin.readline().strip()
solve()
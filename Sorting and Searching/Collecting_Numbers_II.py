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
    d = [0 for i in range(len(L1)+1)]
    ans = 1
    isprob = [False for i in range(L[0]+1)]
    for i in range(len(L1)):
        d[L1[i]] = i
    for i in range(1,L[0]):
        if d[i]>d[i+1]:
            ans += 1
            isprob[i] = True
    # print(LDS)
    for i in range(L[1]):
        L2 = list(map(int, sys.stdin.readline().split()))
        L2[0] -= 1
        L2[1] -= 1
        if isprob[L1[L2[1]]]:
            isprob[L1[L2[1]]] = False
            ans -= 1
        if isprob[L1[L2[0]]]:
            isprob[L1[L2[0]]] = False
            ans -= 1
        if isprob[L1[L2[1]]-1]:
            isprob[L1[L2[1]]-1] = False
            ans -= 1
        if isprob[L1[L2[0]]-1]:
            isprob[L1[L2[0]]-1] = False
            ans -= 1
 
        L1[L2[0]],L1[L2[1]] = L1[L2[1]],L1[L2[0]]
        d[L1[L2[0]]] = L2[0]
        d[L1[L2[1]]] = L2[1]
 
        if L1[L2[0]]+1<=L[0] and d[L1[L2[0]]]>d[L1[L2[0]]+1] and not isprob[L1[L2[0]]]:
            ans += 1
            isprob[L1[L2[0]]] = True
        if L1[L2[1]]+1<=L[0] and d[L1[L2[1]]]>d[L1[L2[1]]+1] and not isprob[L1[L2[1]]]:
            ans += 1
            isprob[L1[L2[1]]] = True
        if L1[L2[1]]!=1 and d[L1[L2[1]]-1]>d[L1[L2[1]]] and not isprob[L1[L2[1]]-1]:
            ans += 1
            isprob[L1[L2[1]]-1] = True
        if L1[L2[0]]!=1 and d[L1[L2[0]]-1]>d[L1[L2[0]]] and not isprob[L1[L2[0]]-1]:
            ans += 1
            isprob[L1[L2[0]]-1] = True
        # if isprob[L1[L2[0]-1]]:
        #     ans += 1
        # if isprob[L1[L2[1]-1]]:
        #     ans += 1
        print(ans)
    #st = sys.stdin.readline().strip()
solve()
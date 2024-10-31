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
    for i in range(len(L1)):
        L1[i] = (L1[i],i)
    L1.sort()
    for k in range(len(L1)-3):
        for i in range(k+1,len(L1)-2):
            start = i+1
            end = len(L1)-1
            while start<end:
                y = L1[k][0]+L1[i][0]+L1[start][0]+L1[end][0]
                if y>L[1]:
                    end -= 1
                elif y<L[1]:
                    start += 1
                else:
                    print(L1[k][1]+1,L1[i][1]+1,L1[start][1]+1,L1[end][1]+1)
                    return 
    print("IMPOSSIBLE")
    #st = sys.stdin.readline().strip()
solve()
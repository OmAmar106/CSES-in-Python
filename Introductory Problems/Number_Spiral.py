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
    if (x%2==0 and x>y) or (y%2==1 and y>x):
        print(max(x,y)**2-min(x,y)+1)
    else:
        print((max(x,y)-1)**2+min(x,y))
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
for _ in range(int(sys.stdin.readline().strip())):
    solve()

# def solve1(x,y):
#     if (x%2==0 and x>y) or (y%2==1 and y>x):
#         print(max(x,y)**2-min(x,y)+1,end=' ')
#     else:
#         print((max(x,y)-1)**2+min(x,y),end=' ')

# for i in range(1,6):
#     for j in range(1,6):
#         solve1(i,j)
#     print()
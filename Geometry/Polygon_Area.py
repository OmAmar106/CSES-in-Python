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

#String hashing : sh/shclass, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

def solve():

    # this is called as shoelace formula , it says that area of any polygon is equal to 1/2 * ( (x1y2+x2y3...) - (y1x2+y2x3+....) )
    # pretty weird that it works ;-;, also known as Gauss's area formula
    n = int(sys.stdin.readline().strip())
    L2 = []

    area = lambda *p: abs(sum(i[0] * j[1] - j[0] * i[1] for i, j in zip(p, p[1:] + p[:1]))) / 2

    for i in range(n):
        L = list(map(int, sys.stdin.readline().split()))
        L2.append(L)
    ans = 0

    for i in range(len(L2)):
        ans += (L2[i][0]*(L2[(i+1)%len(L2)][1]))

    for i in range(len(L2)):
        ans -= (L2[i][1]*(L2[(i+1)%len(L2)][0]))
    ans = abs(ans)
    try:
        assert(ans==2*area(*L2))
        print(int(2*area(*L2)))
    except:
        print(ans)
    #L1 = list(map(int, sys.stdin.readline().split()))
    #st = sys.stdin.readline().strip()
solve()
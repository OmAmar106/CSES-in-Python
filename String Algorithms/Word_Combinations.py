import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 10**9+7
MOD1 = 10**9 + 7
RANDOM = random.randrange(2**62)
def gcd(a,b):
    if a%b==0:
        return b
    else:
        return gcd(b,a%b)
def lcm(a,b):
    return a//gcd(a,b)*b
def w(x):
    return (x ^ RANDOM)
##

#String hashing : sh, fenwick sortedlist : fsortl, Number : numtheory, SparseTable : SparseTable
#bucket sorted list : bsortl, segment tree(lazy propogation) : SegmentTree,Other, bootstrap : bootstrap
#binary indexed tree : BIT, segment tree(point updates) : SegmentPoint, Convex Hull : hull
#Combinatorics : pnc, Diophantine Equations : dpheq, Graphs : graphs, DSU : DSU
#Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class Trie:
    def __init__(self):
        self.root = {}

    def add(self, word):
        current_dict = self.root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict[0] = 0

    def traverse(self,index,dp,st):
        current_dict = self.root
        for i in range(index,len(st)):
            letter = st[i]
            if letter not in current_dict:
                return
            current_dict = current_dict[letter]
            if 0 in current_dict:
                dp[i+1] += dp[index]
                if dp[i+1]>MOD:
                    dp[i+1] -= MOD
    
def solve():
    # got AC in pypy2 only , not in pypy3 :/ 
    st = list(sys.stdin.readline().strip())
    T = Trie()
    for i in range(int(sys.stdin.readline().strip())):
        T.add(sys.stdin.readline().strip())
    dp = [0]*(len(st)+1)
    dp[0] = 1
    for i in range(1,len(dp)):
        T.traverse(i-1,dp,st)
    print(dp[-1])
solve()
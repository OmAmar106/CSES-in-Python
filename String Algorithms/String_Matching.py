import sys,math,random
from heapq import heappush,heappop
from bisect import bisect_right,bisect_left
from collections import Counter,deque,defaultdict
from itertools import permutations

# functions #
MOD = 998244353
MOD = 10**11 + 7
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

def solve():
    #L1 = list(map(int, sys.stdin.readline().split()))
    st = sys.stdin.readline().strip()
    def getSubHash(left,right,hash,power):
        #returns hash value of [left,right]
        #check for two diff hash,power to prevent collision
        #change MOD according to how many values you want ,
        #the more the value , lesser the collisiosn
        left += 1
        right += 1
        h1 = hash[right]
        h2 = (hash[left - 1] * power[right - left + 1]) % MOD
        return (h1 + MOD - h2) % MOD
    n = len(st)
    hash1 = [0] * (n + 1)
    # hash1 = [0] * (n + 1)
    power = [0] * (n + 1)
    # power1 = [0] * (n + 1)
    P = 137
    # P1 = 132
    power[0] = 1
    # power1[0] = 1
    for i in range(n):
        power[i + 1] = (power[i] * P) % MOD
        # power1[i+1] = (power1[i] * P1) % MOD
        hash1[i + 1] = (hash1[i] * P + ord(st[i])) % MOD
        # hash1[i + 1] = (hash1[i] * P1 + ord(st[i])) % MOD
    hasho = hash1
    def z_function(S):
        # return: the Z array, where Z[i] = length of the longest common prefix of S[i:] and S
        n = len(S)
        Z = [0] * n
        l = r = 0
        for i in range(1, n):
            z = Z[i - l]
            if i + z >= r:
                z = max(r - i, 0)
                while i + z < n and S[z] == S[i + z]:
                    z += 1
                l, r = i, i + z
            Z[i] = z
        Z[0] = n
        return Z
    
    st = sys.stdin.readline().strip()
    n = len(st)
    hash = [0] * (n + 1)
    # hash1 = [0] * (n + 1)
    # power1 = [0] * (n + 1)
    P = 137
    # P1 = 132
    if n+1>len(power):
        power = [0] * (n + 1)
    # power1[0] = 1
    for i in range(n):
        power[i + 1] = (power[i] * P) % MOD
        # power1[i+1] = (power1[i] * P1) % MOD
        hash[i + 1] = (hash[i] * P + ord(st[i])) % MOD

    f = getSubHash(0,len(st)-1,hash,power)
    ans = 0
    for i in range(len(hasho)-len(st)):
        if getSubHash(i,i+len(st)-1,hasho,power)==f:
            ans += 1
    print(ans)

solve()
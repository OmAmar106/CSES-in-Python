// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define print(arr) for (auto it : arr){cout<<it<<" ";}cout<<endl;
#define len(arr) arr.size()
#define printf(x) cout << x << endl;
#define printm(map) cout<<"{";for(auto it: map){cout<<it.first<<":"<<it.second<<",";};cout<<"}"<<endl;

// const int MOD = 998244353;
const int MOD = 1e9 + 7;

int gcd(int a, int b) {
    if (b == 0) return a;
    return gcd(b, a % b);
}

int lcm(int a, int b) {
    return a / gcd(a, b) * b;
}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable
// Segment Tree(lazy propogation): SegmentTree
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull, Trie/Treap: Tries
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, DSU: DSU, Geometry: Geometry, FFT: fft
// Persistent Segment Tree: perseg, FreqGraphs: bgraph
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void solve() {
    int n,x; cin >> n>>x;
    vector<int> L(n);
    for (int &x : L) cin >> x;
    vector<int> L1(n);
    for (int &x : L1) cin >> x;
    int dp[x+1] = {-INF};
    dp[0] = 0;
    int ans = 0;
    for(int i=0;i<n;i++){
        for(int j=x-L[i];j>=0;j--){
            dp[j+L[i]] = max(dp[j+L[i]],dp[j]+L1[i]); 
        }
    }
    for(int i=0;i<x+1;i++){
        ans = max(ans,dp[i]);
    }
    printf(ans);
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}
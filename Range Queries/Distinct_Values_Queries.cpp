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

class BIT {
public:
    int n;
    vector<int> bit;

    BIT(vector<int>& arr) {
        n = arr.size();
        bit = arr;
        for (int i = 0; i < n; ++i) {
            int j = i | (i + 1);
            if (j < n) bit[j] += bit[i];
        }
    }

    void update(int idx, int x) {
        while (idx < n) {
            bit[idx] += x;
            idx |= idx + 1;
        }
    }

    int query(int end) {
        int x = 0;
        while (end > 0) {
            x += bit[end - 1];
            end &= end - 1;
        }
        return x;
    }

    int findkth(int k) {
        int idx = -1;
        for (int d = 63 - __builtin_clzll(n); d >= 0; --d) {
            int right_idx = idx + (1LL << d);
            if (right_idx < n && k >= bit[right_idx]) {
                idx = right_idx;
                k -= bit[idx];
            }
        }
        return idx + 1;
    }
};

void solve() {
    int n,q; cin >> n>>q;
    vector<int> L(n);
    map<int,vector<int>> d;
    for(int i=0;i<n;i++){
        cin>>L[i];
    }
    for(int i=n-1;i>=0;i--){
        d[L[i]].push_back(i);
    }

    vector<int> k(n,0);
    for(auto &it:d){
        k[it.second[len(it.second)-1]] = 1;
        it.second.pop_back();
    }
    BIT seg = BIT(k);
    vector<tuple<int,int,int>> L2;
    for(int i=0;i<q;i++){
        int a,b;
        cin>>a>>b;
        L2.push_back(make_tuple(--a,--b,i));
    }
    sort(L2.begin(),L2.end());
    vector<int> fans(q,0);
    int k3 = 0;
    for(auto tup:L2){
        int st = get<0>(tup);
        int ed = get<1>(tup);
        int pos = get<2>(tup);
        for(int l=k3;l<st;l++){
            if(len(d[L[l]])){
                seg.update(d[L[l]][len(d[L[l]])-1],1);
                d[L[l]].pop_back();
            }
        }
        k3 = st;
        fans[pos] = seg.query(ed+1)-seg.query(st);
    }
    for(auto it:fans){
        cout<<it<<endl;
    }
}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}
// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define int long long
#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(0);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vector<int>>;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<typename T>
class is_printable {
    template<typename U> static auto test(int) -> decltype(cout << declval<U>(), true_type{});
    template<typename> static auto test(...) -> false_type;
public: static constexpr bool value = decltype(test<T>(0))::value;
};
template<typename T> enable_if_t<is_printable<T>::value> __print(const T& val) { cout << val; }
template<typename T1, typename T2> void __print(const pair<T1, T2>& p) {
    cout << "("; __print(p.first); cout << ":"; __print(p.second); cout << ")";
}
template<typename T> enable_if_t<!is_printable<T>::value> __print(const T& container) {
    cout << "{"; bool first = true;
    for (const auto& x : container) { if (!first) cout << ", "; __print(x); first = false; }
    cout << "}";
}
void print() { cout << "\n"; }
template<typename T, typename... Args>
void print(const T& t, const Args&... rest) {
    __print(t); if constexpr (sizeof...(rest)) cout << " "; print(rest...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

void DnC(vi &prev, vi &dp, vi &pref,int l, int r,int opt_l, int opt_r){
    if(l>r){
        return;
    }
    int mid = (l+r)/2;
    int best_k = -1;
    int best_val = INF;
    int end = max(0ll,min(mid-1,opt_r));
    // print(opt_l,opt_r);
    for(int k=opt_l;k<end+1;k++){
        if(prev[k]==INF){
            continue;
        }
        // print(k);
        // print(prev[k],(pref[mid]-pref[k]));
        int val = prev[k]+(pref[mid]-pref[k])*(pref[mid]-pref[k]);
        if(val<best_val){
            best_val = val;
            best_k = k;
        }
    }
    dp[mid] = best_val;
    DnC(prev,dp,pref,l,mid-1,opt_l,best_k);
    DnC(prev,dp,pref,mid+1,r,best_k,opt_r);
}

void solve() {
    int n,k;
    cin>>n>>k;

    vi L = LII(n);

    vi pref = {0};
    for(auto it:L){
        pref.pb(pref.back()+it);
    }
    // print(pref);
    vi prev(n+1,INF);
    prev[0] = 0;

    vi dp(n+1,0);

    while(k--){
        // print(prev);
        // print(dp);
        DnC(prev,dp,pref,0,n,0,n);
        swap(prev,dp);
    }

    print(prev.back());

}

int32_t main() {
    fastio
    int t=1;
    while (t--) solve();
    return 0;
}
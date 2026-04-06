// Choose cpp when the time
// limit will for sure give tle in pypy
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;

#define INF LLONG_MAX
#define fastio ios::sync_with_stdio(false); cin.tie(nullptr); cout.tie(nullptr);
#define len(arr) arr.size()
#define f first
#define s second
#define pb push_back
#define all(x) x.begin(), x.end()
#define range(i, n) for (int i = 0; i < (n); ++i)
#define rangea(i, a, b) for (int i = (a); i < (b); ++i)
#define int long long
#define MOD1 998244353
#define MOD 1000000007
using vi = vector<int>;
using vii = vector<vi>;
using viii = vector<vii>;
using ll = long long;
using pi = pair<int,int>;
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

#ifdef LOCAL
#define dbg(x...) cout<<"["<<#x<<"] = "; print(x);
#else
#define dbg(x...)
#endif

template <typename T>
using ordered_multiset = tree<
    std::pair<T, int>,
    null_type,
    std::less<std::pair<T, int>>,
    rb_tree_tag,
    tree_order_statistics_node_update
>;
template<class T> bool chmin(T& a,const T& b){ if(b<a){ a=b; return 1;} return 0; }
template<class T> bool chmax(T& a,const T& b){ if(a<b){ a=b; return 1;} return 0;}
template<class T> using minpq = priority_queue<T,vector<T>,greater<T>>;
template<class T> using maxpq = priority_queue<T>;
template<typename T> istream& operator>>(istream& in, vector<T>& v) {for (auto& x : v) in >> x;return in;}
template<typename T1, typename T2> istream& operator>>(istream& in, pair<T1, T2>& p) {in >> p.first >> p.second;return in;}
template<class T> void _print(const T &x){ cout << x; }
template<class T,class U> void _print(const pair<T,U> &p){cout<<"(";_print(p.first);cout<<",";_print(p.second);cout<<")";}
template<class T>
void _print(const vector<T> &v){
    cout<<"[";bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"]";
}
template<class T>
void _print(const set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T>
void _print(const unordered_set<T> &v){
    cout<<"{";
    bool f=0;
    for(const auto &x: v){
        if(f) cout<<", ";
        _print(x);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const unordered_map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
template<class T,class U>
void _print(const map<T,U> &m){
    cout<<"{";
    bool f=0;
    for(const auto &p: m){
        if(f) cout<<", ";
        _print(p);
        f=1;
    }
    cout<<"}";
}
void print(){ cout<<"\n"; }
template<class T,class... Args>
void print(const T& a,const Args&... args){
    _print(a);
    if constexpr(sizeof...(args)) cout<<" ";
    print(args...);
}

int gcd(int a, int b) {if (b == 0) return a;return gcd(b, a % b);}
int lcm(int a, int b) {return a / gcd(a, b) * b;}
int II(){int a;cin>>a;return a;}
string SI(){string s;cin>>s;return s;}
vi LII(int n){vi a(n);cin>>a;return a;}
int rnd(int l,int r){return uniform_int_distribution<int>(l,r)(rng);}

// String hashing: sh/shclass, Number: numtheory, SparseTable: SparseTable, SortedList: sortl
// Segment Tree(lazy propogation): SegmentTree, Merge Sort Tree: sorttree, Trie/Treap: Tries
// binary indexed tree: BIT, Segment Tree(point updates): SegmentPoint, Convex Hull: hull
// Combinatorics: pnc, Diophantine Equations: dpheq, Graphs: graphs, Centroid Decomp.: graph_decom
// Persistent Segment Tree: perseg, FreqGraphs: bgraph, GrapthOth: graphoth, DSU: DSU, FFT:fft
// Rollback/Par DSU: rbdsu, treap: treap, graphflow(mat_match): graphflow, Persistent Seg Tree: perseg
// Segment Tree(Nodes): SegmentNode, HLD: hld, fwht: fwht
// Template : https://github.com/OmAmar106/Template-for-Competetive-Programming

class SegmentTree {
public:
    int n;
    vii tree;

    static vi func(vi a, vi b) {
        vi ans(31,INF);
        for(int i=0;i<31;i++){
            ans[i] = min(a[i],b[i]);
        }
        // dbg(a);
        // dbg(b);
        // dbg(ans);
        return ans;
    }

    SegmentTree(const vector<int>& data) {
        n = data.size();
        tree.assign(2*n, vi(31,INF));
        build(data);
    }

    void build(const vector<int>& data) {
        for (int i = 0; i < n; ++i) {
            tree[n + i][31-__builtin_clz(data[i])] = data[i];
        }
        for (int i = n - 1; i > 0; --i) {
            tree[i] = func(tree[i * 2], tree[i * 2 + 1]);
        }
    }

    // void update(int pos, int value) {
    //     pos += n;
    //     tree[pos] = value;
    //     while (pos > 1) {
    //         pos /= 2;
    //         tree[pos] = func(tree[2 * pos], tree[2 * pos + 1]);
    //     }
    // }

    vi query(int left, int right) {
        left += n;
        right += n;
        vi max_val(31,INF);
        while (left < right) {
            if (left % 2 == 1) {
                max_val = func(max_val, tree[left]);
                left++;
            }
            if (right % 2 == 1) {
                right--;
                max_val = func(max_val, tree[right]);
            }
            left /= 2;
            right /= 2;
        }
        return max_val;
    }
};

void solve() {
    int n,q;
    cin>>n>>q;
    vi L = LII(n);
    vii pref(n+1,vi (31,0));

    for(int i=1;i<=len(L);i++){
        for(int j=0;j<31;j++){
            pref[i][j] = pref[i-1][j]+L[i-1]*(L[i-1]<=(1<<j));
        }
    }

    SegmentTree seg = SegmentTree(L);

    while(q--){
        int l,r;
        cin>>l>>r;
        l--;r--;
        int cur = 0;
        vi ans = seg.query(l,r+1);
        for(int j=0;j<31;j++){
            if((cur+1)>=(1<<j)){
                cur = pref[r+1][j]-pref[l][j];
            }
            else{
                if(ans[j-1]<=cur+1){
                    cur = pref[r+1][j]-pref[l][j];
                }
                else{
                    break;
                }
            }
        }
        print(cur+1);  
    }
}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}
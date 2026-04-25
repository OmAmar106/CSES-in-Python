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

struct Node {
    public:
    int val,count;

    Node(int v = 0,int cnt = 1) { val = v; count=cnt;}

    // merge
    static Node func(const Node* left, const Node* right) {
        if (!right) {
            if (!left) return Node(0);
            return *left;
        }
        if (!left) return *right;
        if(left->val<right->val){return Node(left->val,left->count);}
        if(left->val>right->val){return Node(right->val,right->count);}
        return Node(left->val,right->count+left->count);
    }

    void add(int v) { val += v; }
    void set(int v) { val = v; }
};

class SegmentTree {
public:
    int n;
    vector<Node> tree;
    vector<int> lazy_add;
    vector<int> lazy_set;
    vector<bool> has_set;

    SegmentTree(vector<int>& data) {
        n = data.size();
        tree.resize(4*n);
        lazy_add.assign(4*n, 0);
        lazy_set.assign(4*n, 0);
        has_set.assign(4*n, false);
        build(1, 0, n-1, data);
    }

    void build(int idx, int l, int r, vector<int>& data) {
        if (l == r) {
            tree[idx] = Node(data[l]);
            return;
        }
        int mid = (l + r) / 2;
        build(idx*2, l, mid, data);
        build(idx*2+1, mid+1, r, data);
        tree[idx] = Node::func(&tree[idx*2], &tree[idx*2+1]);
    }


    void apply_add(int idx, int l, int r, int val) {
        tree[idx].add(val);
        if (has_set[idx]) lazy_set[idx] += val;
        else lazy_add[idx] += val;
    }

    void push(int idx, int l, int r) {
        if (l == r) return;
        int mid = (l + r) / 2;

        if (lazy_add[idx]) {
            apply_add(idx*2, l, mid, lazy_add[idx]);
            apply_add(idx*2+1, mid+1, r, lazy_add[idx]);
            lazy_add[idx] = 0;
        }
    }

    // Range Update in [L,R] if flag, then add
    void update(int idx, int l, int r, int ql, int qr, int val) {
        if (qr < l || r < ql) return;

        if (ql <= l && r <= qr) {
            apply_add(idx, l, r, val);
            return;
        }

        push(idx, l, r);

        int mid = (l + r) / 2;
        update(idx*2, l, mid, ql, qr, val);
        update(idx*2+1, mid+1, r, ql, qr, val);

        tree[idx] = Node::func(&tree[idx*2], &tree[idx*2+1]);
    }

    // Range Query in [L,R]
    Node query(int idx, int l, int r, int ql, int qr) {
        if (qr < l || r < ql) return Node(0);

        if (ql <= l && r <= qr) return tree[idx];

        push(idx, l, r);

        int mid = (l + r) / 2;
        Node left = query(idx*2, l, mid, ql, qr);
        Node right = query(idx*2+1, mid+1, r, ql, qr);

        return Node::func(&left, &right);
    }
};

void solve() {
    int n;
    cin>>n;
    vector<tuple<int,int,int,int>> L;
    int t = -INF;
    int t1 = INF;

    for(int i=0;i<n;i++){
        int x1,y1,x2,y2;
        cin>>x1>>y1>>x2>>y2;
        t = max(t,y2);
        t1 = min(t1,y1);
        L.emplace_back(x1,y1,y2,1);
        L.emplace_back(x2,y1,y2,-1);
    }

    sort(L.begin(),L.end());
    
    const int MAXI = t-t1+1;
    
    vi L3(MAXI,0);
    SegmentTree seg = SegmentTree(L3);
    
    int prev = get<0>(L[0]);
    int i = 0;
    int ans = 0;
    while(i<len(L)){
        int f = get<0>(L[i]);
        // return;
        ans += (f-prev)*(MAXI-seg.query(1,0,MAXI-1,0,MAXI-1).count);
        // dbg(prev,f,(MAXI-seg.query(1,1,n,0,MAXI-1).count));
        while(i<len(L) && get<0>(L[i])==f){
            seg.update(1,0,MAXI-1,get<1>(L[i])-t1,get<2>(L[i])-t1-1,get<3>(L[i]));
            i += 1;
        }
        prev = f;
    }

    print(ans);

}

int32_t main() {
    fastio
    int t=1;
    // cin >> t;
    while (t--) solve();
    return 0;
}
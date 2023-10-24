

// /////////// adjacency list/////////////////

#include<bits/stdc++.h>
#define int long long int
using namespace std;
int32_t main(){
// 6 5
// 1 5
// 1 2
// 2 3
// 3 6
// 3 4
    int v,e;
    cout<<"enter the vertex and edges"<<endl;
    cin>>v>>e;
    vector<int> g[v+1]; // row are fix but col is variable
    for(int i=0;i<e;i++){   // col is variable
        int x,y;
        cin>>x>>y;
        g[x].push_back(y);
        g[y].push_back(x);
    }
 
// 1--> 5 2 
// 2--> 1 3 
// 3--> 2 6 4 
// 4--> 3 
// 5--> 1 
// 6--> 3 
    for(int i=1;i<=v;i++){
        cout<<i<<"--> "; // row 
        for(int j=0;j<g[i].size();j++){  // variable col
            cout<<g[i][j]<<" ";
        }
        cout<<endl;
    }
    return 0; 
}

//////////// matrix///////////////////////


#include<bits/stdc++.h>
using namespace std;
int main(){
// 6 5 
// 1 5
// 1 2
// 2 3
// 3 6
// 3 4
    int v,e;
    cout<<"enter the vertex and edges"<<endl;
    cin>>v>>e;
    int g[v+1][v+1];
    for(int i =1;i<=v;i++){
         for(int j =1;j<=v;j++){
             g[i][j]=0;
         }
    }
    for(int i =0;i<e;i++){
         int x, y;
         cin>>x>>y;
         g[x][y]=1;
         g[y][x]=1;
    }

// 1-->0 1 0 0 1 0 
// 2-->1 0 1 0 0 0 
// 3-->0 1 0 1 0 1 
// 4-->0 0 1 0 0 0 
// 5-->1 0 0 0 0 0 
// 6-->0 0 1 0 0 0 

 for(int i =1;i<=v;i++){
      cout<<i<<"-->";
      for(int j =1;j<=v;j++){
          cout<<g[i][j]<<" ";
      }cout<<endl;
 }
}

//////////////////////////////// from adjmatrix to adjacency list///////////////////

#include <iostream>
#include <vector>

int main() {
    // Given list of edges
    std::vector<std::vector<int>> edges = {
        {0, 1},
        {0, 2},
        {3, 5},
        {5, 4},
        {4, 3}
    };

    int n = 6;  // Total number of vertices (assuming vertex IDs are 0 to 5)

    // Convert edges to adjacency list
    std::vector<std::vector<int>> adjacencyList(n);
    for (const auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        adjacencyList[u].push_back(v);
        adjacencyList[v].push_back(u);  // Assuming an undirected graph
    }

    // Print the adjacency list
    std::cout << "Adjacency List:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "Vertex " << i << " -> ";
        for (int j : adjacencyList[i]) {
            std::cout << j << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

//////////////////// from list to matrix ////////////////////////////////

#include <iostream>
#include <vector>

using namespace std;

const int MAX_VERTICES = 6; // Update this with the actual number of vertices

void adjacencyListToMatrix(const vector<vector<int>>& adjList, vector<vector<int>>& adjMatrix) {
    for (int i = 0; i < adjList.size(); ++i) {
        for (int j : adjList[i]) {
            adjMatrix[i][j] = 1;
        }
    }
}

int main() {
    vector<vector<int>> adjacencyList = {
        {1, 2},
        {0},
        {0},
        {5, 4},
        {5, 3},
        {3, 4}
    };

    vector<vector<int>> adjacencyMatrix(MAX_VERTICES, vector<int>(MAX_VERTICES, 0));

    adjacencyListToMatrix(adjacencyList, adjacencyMatrix);

    // Display the adjacency matrix with 1s and 0s
    for (int i = 0; i < MAX_VERTICES; ++i) {
        for (int j = 0; j < MAX_VERTICES; ++j) {
            cout << adjacencyMatrix[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}

///////////////////////// in adjacency list outdegree///////////////////////

// Now, the outdegree of a vertex is simply the number of outgoing edges from that vertex.
//  In terms of the adjacency list, it's the size of the inner vector for that vertex. So, 
//  when we iterate through the vertices and print the size of the corresponding inner vector, we get the outdegree for each vertex.

#include <bits/stdc++.h>
#define int long long int
using namespace std;

int main() {
    int v, e;
    cout << "Enter the number of vertices and edges: ";
    cin >> v >> e;

    vector<vector<int>> adjList(v + 1);

    for (int i = 0; i < e; i++) {
        int x, y;
        cin >> x >> y;
        adjList[x].push_back(y);
    }

    // Calculate and print outdegree for each vertex
    for (int i = 1; i <= v; i++) {
        cout << "Outdegree of vertex " << i << " is: " << adjList[i].size() << endl;
    }

    return 0;
}


///////////////////// calculation if indegree and outdergree////////////


#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

void calculateDegrees(int n, vector<vector<int>>& trust) {
    unordered_map<int, int> indegree;
    unordered_map<int, int> outdegree;

    // Initialize dictionaries with zeros
    for (int i = 1; i <= n; ++i) {
        indegree[i] = 0;
        outdegree[i] = 0;
    }

    // Update dictionaries based on trust relationships
    for (auto t : trust) {
        int src = t[0];
        int dest = t[1];
        outdegree[src]++; // first element of vector
        indegree[dest]++; // second element of vector
    }

    // Calculate and display degrees
    cout<<"indegree"<<endl;
    for (int node = 1; node <= n; ++node) {
        cout<<indegree[node]<<endl;;
    }

        cout<<"outdegree"<<endl;
    for (int node = 1; node <= n; ++node) {
        cout<<outdegree[node]<<endl;;
    }
}

int main() {
    int n = 3;
    vector<vector<int>> trust = {{1, 3}, {2, 3}, {3, 1}};
    calculateDegrees(n, trust);
    return 0;
}


// ////////////////bfs gfg////////////////

class Solution {
  public:
    // Function to return Breadth First Traversal of given graph.
    vector<int> bfsOfGraph(int V, vector<int> adj[]) {
        vector<int> ans;
        vector<int>visited(V,false);
        queue<int>q;
        q.push(0);
        visited[0] =true;
        while(!q.empty()){
              int top = q.front();
              q.pop();
              ans.push_back(top);
              for(auto x:adj[top]){
                   if(!visited[x]){
                       visited[x]=true;
                       q.push(x);
                   }
              }
        }
        return ans;
    }
};


////////////////////////dfs/////////////

class Solution {
  public:
    // Function to return a list containing the DFS traversal of the graph.
    
    void dfs(int src ,vector<int>&visited,vector<int> &ans,vector<int> adj[]){
             ans.push_back(src);
             visited[src] =true;
            for(auto nbr:adj[src]){
                if(!visited[nbr]) dfs(nbr,visited,ans,adj);
            }
    }
    vector<int> dfsOfGraph(int V, vector<int> adj[]) {
        vector<int> ans;
        vector<int> visited(V,false);
        dfs(0,visited,ans,adj);
        return ans;
        
    }
};


//////////////////////// cycle detection in directed graph gfg////////////////////

// step1: make visited vector to mark the visibility
// step2: make dfs visited vector to ensure the function called 
// step3: iterate over graph using dfs if anywhere we we get the cycle means (visited == true ans dfsvisited is == true cycle is present)]
// step4: after comlication calles if we are returning then we have to again mark false;


class Solution {
  public:
    // Function to detect cycle in a directed graph.
    bool cycleDfs(int src,  vector<int>&visited,  vector<int> &dfsvisited,vector<int> adj[]){
         visited[src]=1;
         dfsvisited[src]=1;
         for(auto nbr:adj[src]){
              if(!visited[nbr]){  // not visited
                   bool conf = cycleDfs(nbr,visited,dfsvisited,adj);
                   if(conf==true) return true;
              }
              else if(dfsvisited[nbr]) return true;
         }
          dfsvisited[src]=0;// returning call
          return false;
         
    }
    bool isCyclic(int V, vector<int> adj[]) {
        // code here
        vector<int>visited(V,0);
        
        vector<int>dfsVisited(V,0);
        
        for(int i =0;i<V;i++){
            if(!visited[i]){
                bool flag = cycleDfs(i,visited,dfsVisited,adj);
                if(flag) return true;
            }
        }
        
        return false;
    }
};


//////////////////////// cycle detection in undirected graph gfg////////////////////

// step1: make visited vector to mark the visibility
// step2: make parent vector for ensuring the parent of evert node
// step3: if(node is visited and  is not parent[src] !=nbr) the cycle exits

class Solution {
  public:
    // Function to detect cycle in an undirected graph.
    bool cycleInUndir(int src,vector<int> &visited,vector<int> & parent,vector<int> adj[]){
          visited[src]=1;
          for(auto nbr:adj[src]){
               if(!visited[nbr]){
               parent[nbr] = src;
               bool flag = cycleInUndir(nbr,visited,parent,adj);
               if(flag) return true;
               } else{
                    if(parent[src] !=nbr) return true;
               }
          }
          
          return false;
    }
    bool isCycle(int V, vector<int> adj[]) {
        vector<int>visited(V,0);
        vector<int>parent(V,-1);
        
        // for disconnected graph
        for(int i =0;i<V;i++){
            if(!visited[i]){
                bool flag = cycleInUndir(i,visited,parent,adj);
                if(flag) return true;
            }
        }
        return false;
    }
};

/////////////////// topological sort(Kanhe's algo) gfg/////////////////////

class Solution
{
	public:
	//Function to return list containing vertices in Topological order. 
	vector<int> topoSort(int V, vector<int> adj[]) {
	    unordered_map<int,int>mp;
	    // calculating indegree , id the freq of adjList
	    
	    for(int i =0;i<V;i++){
	         for(auto x:adj[i]){
	              mp[x]++;
	         }
	    }
	    
	    queue<int>q;
	    vector<int> ans;
	    for(int i =0;i<V;i++){
	         if(mp.find(i) == mp.end()){ // freq is zero
	           q.push(i);
	         }
	    }
	    
	        while (!q.empty()) {
            int frontNode = q.front();
            q.pop();
            ans.push_back(frontNode);
            
            for (auto nbr : adj[frontNode]) {
                mp[nbr]--;
                if (mp[nbr] == 0) {
                    q.push(nbr);
                }
            }
            
	  }        
        return ans;
	}
};


//////////////// topological sort using dfs gfg///////////////

class Solution
{
	public:
	//Function to return list containing vertices in Topological order. 
	void dfs(vector<int> & visited,vector<int> adj[] , int src,stack<int>&st){
	     visited[src]=1;
	     for(auto x:adj[src]){
	        if(!visited[x]){
	              dfs(visited,adj,x,st);
	        }
	     }
	     
	     st.push(src);
	     
	}
	vector<int> topoSort(int V, vector<int> adj[]){
	    stack<int>st;
	    vector<int> visited(V,0);
	    
	    for(int i =0;i<V;i++){
	        if(!visited[i]){
	             dfs(visited,adj,i,st);
	        }
	         
	    }
	    
	    vector<int> ans;
	    while(st.size()>0){
	         ans.push_back(st.top());
	         st.pop();
	    }
	    
	    return ans;
	}
};

////////////////// Minimum Spanning Tree//////////////////////////


class Solution
{
	public:
	//Function to find sum of weights of edges of the Minimum Spanning Tree.
    int spanningTree(int V, vector<vector<int>> adj[]){
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq; // minheap
        int sum = 0;
        pq.push({0,0}); // wt,node 
        vector<bool> visited(V,false);
        while(!pq.empty()){
             auto top = pq.top();
             pq.pop();
             int wt = top.first;
             int node = top.second;
             if(visited[node]) continue;
             else visited[node]=true;
             sum +=wt;
             for(auto nbr:adj[node]){
                  int adjNode = nbr[0];
                  int endW  =   nbr[1];
                  if(!visited[adjNode]) pq.push({endW,adjNode});
             }
             
        }
        return sum;
        
    }
};



////////////////// Minimum Spanning Tree//////////////////////////

class Solution
{
	public:
	//Function to find sum of weights of edges of the Minimum Spanning Tree.
	
	int findMini(vector<int> &key,vector<int> &mst){
	     int temp = INT_MAX;
	     int indx =-1;
	     for(int i =0;i<key.size();i++){
	          if(key[i]<temp && mst[i]==false) {
	               temp=key[i];
	               indx =i;
	          }
	     }
	      return indx;
	}
    int spanningTree(int V, vector<vector<int>> adj[])
    {
        vector<int> key(V,INT_MAX);
        vector<int> mst(V,false);
        vector<int> parent(V,-1);
        key[0] =0;
        while(true){
             // step1; // find mini value node from the key vector u
             int u = findMini(key,mst);
             if(u==-1) break;
             // step2: // mark mst of u as true
             mst[u]=true;
             // step3: find all adjacent v
             for(auto edge:adj[u]){
                 int v = edge[0];
                 int w = edge[1];
                 if(mst[v]==false && w<key[v]){
                      key[v]=w;
                      parent[v]=u;
                 }
             }
        }
        int sum =0;
        for(int i =0;i<key.size();i++) sum+=key[i];
        return sum;
        
    }
};

////////////////////////////Eventual Safe States////////////////////////////////

class Solution {
  public:
    bool f(int src , vector<int>&visited,vector<int>&dfsvisited,vector<int>adj[],vector<int>&ans1){
         visited[src]=true;
         dfsvisited[src]=true;
         ans1[src]=0;
         for(auto nbr:adj[src]){
             if(!visited[nbr]){  // not visited
                   bool conf = f(nbr,visited,dfsvisited,adj,ans1);
                   if(conf==true) return true;
              }
              else if (dfsvisited[nbr]) return true;
         }
         dfsvisited[src]=false;
         ans1[src]=1;
         return false;
         
    }
    vector<int> eventualSafeNodes(int V, vector<int> adj[]) {
        vector<int> visited(V,false);
        vector<int> dfsvisit(V,false);
        vector<int> ans1(V);
        for(int i =0;i<V;i++){
             if(!visited[i]){
                 f(i,visited,dfsvisit,adj,ans1);
             }
        }
        
        vector<int> ans;
        for(int i =0;i<ans1.size();i++){
             if(ans1[i]) ans.push_back(i);
        }
        
        return ans;
        
    }
};

//##############################(krushkal's algo) Minimum Spanning Tree//////////////////////////

// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************
// *************************************************************************************************

/////////////////********************* decode batch*************************************//
////////////////////////////////// flood fill///////////////////////////////////////////

/////////////////////////// DFS algo///////////////////////////////////////////////////////////

class Solution {
public:
    bool isvalid(int sr,int sc , int oldcolor,vector<vector<int>>& image,map<pair<int,int>,bool> &visited){
    if(sr>=0 && sr<image.size() && sc>=0 && sc<image[0].size() && image[sr][sc] ==oldcolor && !visited[{sr,sc}]) {
        return true;
    }
        return false;
    }
   void dfs(int sr , int sc , int color , map<pair<int,int>,bool> &visited,vector<vector<int>>& image){
       visited[{sr, sc}] = true; 
        int oldcolor = image[sr][sc];
        image[sr][sc] = color; 
        if(isvalid(sr+1,sc,oldcolor,image,visited)){
           dfs(sr+1,sc,color,visited,image);
        }
         if(isvalid(sr-1,sc,oldcolor,image,visited)){
           dfs(sr-1,sc,color,visited,image);
        }

         if(isvalid(sr,sc+1,oldcolor,image,visited)){
           dfs(sr,sc+1,color,visited,image);
        }
         if(isvalid(sr,sc-1,oldcolor,image,visited)){
           dfs(sr,sc-1,color,visited,image);
        }
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
        map<pair<int,int>,bool>visited;
        dfs(sr,sc,color,visited,image);
         return image;
    }
};




///////////////////////////////////////////////////////////////////////////

class Solution {
public:
    void f(vector<vector<int>>& image, int sr, int sc, int incolor,int newcolor){
       if(sr<0 || sc<0 || sr>=image.size() || sc>=image[0].size()) return;
       if(image[sr][sc] !=incolor) return;
       image[sr][sc]=newcolor;
       f(image,sr+1,sc,incolor,newcolor);
       f(image,sr,sc-1,incolor,newcolor);
       f(image,sr-1,sc,incolor,newcolor);
       f(image,sr,sc+1,incolor,newcolor);
        
    }
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
      if(image[sr][sc]==color) return image;
      f(image,sr,sc,image[sr][sc],color);
       return image;
    }
};

////////////////////  1791. Find Center of Star Graph///////////////////////////

class Solution {
public:
    int findCenter(vector<vector<int>>& edges) {
        int a = edges[0][0];  //1   [1,2],[2,3]
        int b = edges[0][1];  //2
        int c = edges[1][0];  //2
        int d = edges[1][1];  //3
       return (c==a || c==b) ? c:d;
    }
};


////////////////////////////////////841. Keys and Rooms/////////////////////

class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        vector<int>visited(rooms.size(),false);
        queue<int>q;
        visited[0]=true;
        q.push(0);
        while(!q.empty()){
           int curr = q.front();
           q.pop();
           for(auto nbr:rooms[curr]){
               if(!visited[nbr]){
                   q.push(nbr);
                   visited[nbr]=true;
               }
           }
        }

        for(auto x:visited) if(x==false ) return false;
        return true;
    }
};


/////////////////////// connected components/////////////////////////////

#include<bits/stdc++.h>
#include<iostream>
#include<vector>
#include<list>
#include<unordered_set>
using namespace std;
vector<list<int>> graph;
int v;
void add_edge(int src, int dest, bool bi_dir = true) {
    graph[src].push_back(dest);
    if (bi_dir) graph[dest].push_back(src);
}
void dfs(int node, unordered_set<int>&visited){
    visited.insert(node);
    for(auto nbr:graph[node]){
        if(!visited.count(nbr)){
             dfs(nbr,visited);
        }
    }
}
int count_Of_connected_Components(){
    unordered_set<int>visited;
    int ans =0;
    for(int i =0;i<v;i++){
         if(visited.count(i)==0){
               ans++;
               dfs(i,visited);
         }
    } 
    return ans;
}
int main() {
    cin >> v;
    graph.resize(v, list<int>());
    int e;
    cin >> e;
    while (e--) {
        int s, d;
        cin >> s >> d;
        add_edge(s, d);
    }
    cout<<count_Of_connected_Components()<<endl;
}


/////////////////////// islands/////////////////////////////
/////////////////////////DFS////////////////////////////////
class Solution {
public:
    void dfs(vector<vector<char>>& grid,int i , int j, int r ,int c){
    if (i < 0 || j < 0 || i == r || j == c || grid[i][j] == '0') {
        return;
    }
      grid[i][j] = '0'; // Fix the typo here
      dfs(grid,i+1,j,r,c);
      dfs(grid,i-1,j,r,c);
      dfs(grid,i,j+1,r,c);
      dfs(grid,i,j-1,r,c);
    }
    int numIslands(vector<vector<char>>& grid) {
        int r=grid.size();
        int c=grid[0].size();
        int ans =0;
        unordered_set<int>visited;
        for(int i =0;i<grid.size();i++){
          for(int j =0;j<grid[0].size();j++){
              if(grid[i][j]=='1'){
                 ans++;
                 dfs(grid,i,j,r,c);
              }
          }
        }
        return ans;
    }
};

///////////////////////////// BFS/////////////////////////////////////

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int cc =0;
        for(int i =0;i<rows;i++){
          for(int j =0;j<cols;j++){
            if(grid[i][j]=='0') continue;
            cc++;
            grid[i][j]='0';
            queue<pair<int,int>>q;
            q.push({i,j});
            while(!q.empty()){
               auto curr = q.front();
               q.pop();
               int currow = curr.first;
               int curcols = curr.second;
               // up check
               if(currow-1>=0 && grid[currow-1][curcols]=='1'){
                  q.push({currow-1,curcols});
                  grid[currow-1][curcols]='0';
               }
                    // donw check
                  if(currow+1<rows && grid[currow+1][curcols]=='1'){
                  q.push({currow+1,curcols});
                  grid[currow+1][curcols]='0';
               }

               ////////////// right check
                if(curcols+1<cols && grid[currow][curcols+1]=='1'){
                  q.push({currow,curcols+1});
                  grid[currow][curcols+1]='0';
               }

                if(curcols-1>=0 && grid[currow][curcols-1]=='1'){
                  q.push({currow,curcols-1});
                  grid[currow][curcols-1]='0';
               }
            }
          }
        }
        return cc;
    }
};

//////////////////////// 417. pacific and atlantic water flow//////////////

///////////////// multisource bfs/dfs

class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}}; // {{i+1,j},{i-1,j},{i,j+1},{i,j-1}}
     int rows;
     int cols;
     vector<vector<int>>h;

    vector<vector<bool>> bfs(queue<pair<int,int>> &qu){
     vector<vector<bool>> visited(rows,vector<bool>(cols,false)); 
     while(!qu.empty()){
        auto cell = qu.front();
        qu.pop();
        int i = cell.first;
        int j = cell.second;
        visited[i][j]=true;
        for(int d =0;d<4;d++){
           int newRow = i+dir[d][0];
           int newCol=  j+dir[d][1];
           if(newRow<0 || newCol<0 || newRow>=rows || newCol>=cols) continue; // outofboundary
           if(visited[newRow][newCol]) continue;     // already visited
           if(h[i][j]>h[newRow][newCol]) continue;  // if(curr cell's value is greater than next cell
           qu.push({newRow,newCol});   // fairable case
        }
     }
     return visited;
    }    
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        rows = heights.size();
        cols = heights[0].size();
        h = heights;
        queue<pair<int,int>> q1; //pacific ocean
        queue<pair<int,int>> q2; // atlantic ocean
        // multisource bfs inilazing
        for(int i =0;i<rows;i++){
           q1.push({i,0});     //pacific first col
           q2.push({i,cols-1}); // altanic last col
        }

         for(int j =1;j<cols;j++){
           q1.push({0,j}); // pacific first row
         }

             for(int j = 0;j<cols-1;j++){
               q2.push({rows-1,j}); // altanic last row
           }
           vector<vector<bool>> pacific = bfs(q1);
           vector<vector<bool>> atlantic =bfs(q2);
           vector<vector<int>>ans;
           for(int i =0;i<rows;i++){
              for(int j =0;j<cols;j++){
                 if(pacific[i][j] && atlantic[i][j]) ans.push_back({i,j});
              }
           }

        return ans;

    }
};

///////////////////994. Rotting Oranges///////////////////////


// 0 representing an empty cell,
// 1 representing a fresh orange, or
// 2 representing a rotten orange.

class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
    int miniTime(vector<vector<int>>&grid,queue<pair<int,int>>&q){
        int ans =0;
        while(!q.empty()){
           auto cell = q.front();
           q.pop();
           if(cell.first==-1 && cell.second==-1) {
              ans++;
              if(q.size()>0) q.push({-1,-1});
           }
           
           int i = cell.first;
           int j = cell.second;
           for(int d =0;d<4;d++){
              int newRow = i+dir[d][0];
              int newCol = j+dir[d][1];
              if(newRow<0 || newCol<0||newRow>=grid.size() || newCol>=grid[0].size()) continue;
              if(grid[newRow][newCol]==2 || grid[newRow][newCol]==0) continue;
              grid[newRow][newCol]=2;
              q.push({newRow,newCol});
           }
        }
        
          for(int i =0;i<grid.size();i++){
          for(int j =0;j<grid[0].size();j++){
          if(grid[i][j]==1) return -1;
        }
      }

        return ans-1;
    }
   
    int orangesRotting(vector<vector<int>>& grid) {
      int rows = grid.size();
      int cols=grid[0].size();
      queue<pair<int,int>>q;
      for(int i =0;i<rows;i++){
         for(int j =0;j<cols;j++){
            if(grid[i][j]==2) q.push({i,j});
         }
      }

    q.push({-1,-1});
    return miniTime(grid,q);
        
    }
};

///////////////// 542. 01 Matrix ////////////////////////////

class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int rows = mat.size();
        int cols = mat[0].size();
        queue<pair<int,int>> q;
        vector<vector<int>> ans(rows,vector<int>(cols,-1));
        for(int i =0;i<rows;i++){
           for(int j =0;j<cols;j++){
              if(mat[i][j]==0) {
                 ans[i][j]=0;
                 q.push({i,j});
              }
           }
        }
        while(!q.empty()){
           auto cell = q.front();
           q.pop();
           int i = cell.first;
           int j =cell.second;
           for(int d =0;d<4;d++){
              int newi = i+dir[d][0];
              int newj = j+dir[d][1];
              if(newi>=0 && newj>=0 && newi<rows && newj<cols && ans[newi][newj]==-1){
                 ans[newi][newj] = 1+ans[i][j];
                 q.push({newi,newj});
              }
           }
        }
         return ans;
    }
};


//////////////// 1197. Minimum Knight Moves////////////////


class Solution {
public:
    vector<vector<int>> dir = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
    int minStepToReachTarget(vector<int>& KnightPos, vector<int>& TargetPos, int N) {
        /// NxN chess board
        vector<vector<bool>> vis(N, vector<bool>(N, false));
        queue<pair<int, int>> q;
        q.push({KnightPos[0] - 1, KnightPos[1] - 1});
        vis[KnightPos[0] - 1][KnightPos[1] - 1] = true;
        int moves = 0;

        while (!q.empty()) {
            int n = q.size();

            for (int k = 0; k < n; k++) {
                auto cell = q.front();
                int i = cell.first;
                int j = cell.second;
                q.pop();

                if (TargetPos[0] - 1 == i && TargetPos[1] - 1 == j) return moves;

                for (int d = 0; d < 8; d++) {
                    int newi = i + dir[d][0];
                    int newj = j + dir[d][1];

                    if (newi >= 0 && newj >= 0 && newi < N && newj < N && !vis[newi][newj]) {
                        q.push({newi, newj});
                        vis[newi][newj] = true;
                    }
                }
            }
            moves++;
        }

        return -1; // If the target position is not reachable, return -1.
    }
};


///////////////////130. Surrounded Regions/////////////////////


class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
       int m;
       int n;
    void BFS(vector<vector<char>>&board,queue<pair<int,int>> &q){
          while(!q.empty()){
             auto cell = q.front();
             int i = cell.first;
             int j = cell.second;
             board[i][j] ='B';
             q.pop();
             for(int d =0;d<4;d++){
                int newi  = i+dir[d][0];
                int newj  = j+dir[d][1];
                if(newi>=0 && newj>=0 && newi<m && newj<n && board[newi][newj]=='O'){
                    q.push({newi,newj});
                }
             }
          }
          for(int i =0;i<m;i++){
             for(int j =0;j<n;j++){
                if(board[i][j]=='B') board[i][j]='O';
                else board[i][j]='X'; 
             }
          }

    }
    void solve(vector<vector<char>>& board) {
            m = board.size();
            n = board[0].size();
       queue<pair<int,int>>q;
       for(int i =0;i<m;i++){ // first col
          if(board[i][0]=='O') q.push({i,0});
       }
        for(int i =0;i<m;i++){ // last col
          if(board[i][n-1]=='O') q.push({i,n-1});
       }
         for(int j =0;j<n;j++){ // first row
          if(board[0][j]=='O') q.push({0,j});
       }
          for(int j =0;j<n;j++){ // last row
          if(board[m-1][j]=='O') q.push({m-1,j});
       }
       BFS(board,q);
    }
};

////////////////////// GFG numberOfEnclaves but this same question giving TLE leetcode /////////////

class Solution {
  public:
  
      vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
    int m;
    int n;
    void multiBFS(vector<vector<int>>& grid, queue<pair<int,int>>&q){
       while(!q.empty()){
          auto cell = q.front();
          int i =cell.first;
          int j =cell.second;
          q.pop();
          grid[i][j]=0;
          for(int d =0;d<4;d++){
             int newi = i+dir[d][0];
             int newj = j+dir[d][1];
             if(newi>=0 && newj>=0 && newi<m && newj<n && grid[newi][newj]==1){
                q.push({newi,newj});
             }
          }
       }
    }
    
    int numberOfEnclaves(vector<vector<int>> &grid) {
          m = grid.size();
          n = grid[0].size();
        queue<pair<int,int>>q;
        for(int i =0;i<m;i++){ // first col
           if(grid[i][0]==1) q.push({i,0});
        }
         for(int i =0;i<m;i++){ // last col
           if(grid[i][n-1]==1) q.push({i,n-1});
        }
        for(int j =0;j<n;j++){ // first row
           if(grid[0][j]==1) q.push({0,j});
        }

          for(int j =0;j<n;j++){ // last row
           if(grid[m-1][j]==1) q.push({m-1,j});
        }
        
        int ans =0;
        multiBFS(grid,q);
        for(int i =0;i<m;i++){
           for(int j =0;j<n;j++){
              if(grid[i][j]==1) ans++;
           }
        }
       return ans;
       
    }
};




///////////////////// multisource DFS // boundry DFS ///////////////////

class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
    int m;
    int n;
    void dfs(vector<vector<int>>& grid,int i , int j){
        if(i<0 || j<0 || i>=m || j>=n || grid[i][j]==0) return ;
        grid[i][j]=0;
        for(int d =0;d<4;d++){
           int newi = i+dir[d][0];
           int newj =j+dir[d][1];
           dfs(grid,newi,newj);
        }
    }
    int numEnclaves(vector<vector<int>>& grid) {
       m = grid.size();
       n = grid[0].size();
       for(int i =0;i<m;i++){ // first col
           if(grid[i][0]==1){
              dfs(grid,i,0);
           }
       }

        for(int i =0;i<m;i++){
           if(grid[i][n-1]==1){ // last col
              dfs(grid,i,n-1);
           }
       }
        for(int j =0;j<n;j++){ // first row
           if(grid[0][j]==1){
              dfs(grid,0,j);
           }
       }

         for(int j =0;j<n;j++){ // last row
           if(grid[m-1][j]==1){
              dfs(grid,m-1,j);
           }
       }

    int count =0;
    for(int i =0;i<m;i++){
       for(int j =0;j<n;j++){
          if(grid[i][j]==1) count++;
       }
    }

   return count;

    }
};

//////////////////// 547. Number of Provinces ////////////////////
//A province is a group of directly or indirectly connected cities and no other cities outside of the group.

       //step1: Provinces group of cities connected each others
       //step2: total no of disconnected graph


class Solution {
public:
    void dfs(int src , unordered_set<int> &visited,  unordered_map<int, list<int>> &adj){
       visited.insert(src);
       for(auto nbr:adj[src]){
          if(!visited.count(nbr)){
             dfs(nbr,visited,adj);
          }
       }
    }
    int findCircleNum(vector<vector<int>>& isConnected) {
        // adjacency matrix given
         int n = isConnected.size();
         unordered_set<int> visited;
         unordered_map<int, list<int>> adjList;
        for (int i =0;i<n;i++ ) {
          for(int j =0;j<n;j++){
             if(isConnected[i][j]){ // 0 not allow according to question
                adjList[i].push_back(j);
                adjList[j].push_back(i); //undirected graph
             }
          }
        }


        int count = 0;
        for(int i =0;i<n;i++){
            if(!visited.count(i)){
               count++;
               dfs(i,visited, adjList);
            }
        }
        return count;
    }
};



//////////// 207. Course Schedule ////////////////////

class Solution {
public:
    bool topo(int v,unordered_map<int,list<int>>&graph){
       vector<int>indegree(v,0);
       vector<int>check;
       unordered_set<int>visited;
       queue<int>q;
       for(int i =0;i<v;i++){
          for(auto nbr:graph[i]){
             indegree[nbr]++;
          }
       }

       for(int i =0;i<v;i++){
          if(indegree[i]==0){
             q.push(i);
             visited.insert(i);
          }
       }

       while(!q.empty()){
          int node = q.front();
          check.push_back(node);
          q.pop();
          for(auto nbr:graph[node]){
             if(!visited.count(nbr)){
                indegree[nbr]--;
                if(indegree[nbr]==0){
                   q.push(nbr);
                   visited.insert(nbr);
                }
             }
          }
       }
       if(check.size()==v) return true;
       else return false;
    }
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int,list<int>>graph;
        for(auto x:prerequisites){
           int src  = x[0];
           int dest =x[1];
           graph[src].push_back(dest); // directed 
        }
        return topo(numCourses,graph);
    }
};


/////////////////////////// 207. Course Schedule ///////////////////


class Solution {
public:
   void topoLogicalSort(int numCourses, unordered_map<int, list<int>>& adjList, vector<int> &ans) {
    unordered_map<int, int> indegree; // node, indegree;
    for (auto x : adjList) {
        for (auto nbr : x.second) {
            indegree[nbr]++;
        }
    }
    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) q.push(i);
    }
    while (!q.empty()) {
        int frontNode = q.front();
        q.pop();
        ans.push_back(frontNode);
        for (auto nbr : adjList[frontNode]) {
            indegree[nbr]--;
            if (indegree[nbr] == 0) q.push(nbr);
        }
    }
  
}

    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int,list<int>>adjList;
        for(auto x:prerequisites){
           int u = x[0];
           int v = x[1];
           adjList[v].push_back(u);
        }
        vector<int> ans;
        topoLogicalSort(numCourses,adjList,ans);
          if(ans.size()== numCourses) return ans;
          else return {};
    }
};



/// //////////////////// 210 course schedule II////////////////

class Solution {
public:
    void topoBFS(  unordered_map<int,list<int>>&graph,vector<int>& ans,int v){
       unordered_set<int>visited;
       vector<int>indegree(v,0);
       queue<int>q;
       for(int i =0;i<v;i++){
          for(auto nbr:graph[i]){
             indegree[nbr]++;
          }
       }
       for(int i =0;i<v;i++){
          if(indegree[i]==0) {
             q.push(i);
             visited.insert(i);
          }
       }

       while(!q.empty()){
          int node = q.front();
          ans.push_back(node);
          q.pop();
          for(auto nbr:graph[node]){
              indegree[nbr]--;
                 if(!visited.count(nbr)){
                     if(indegree[nbr]==0){
                       q.push(nbr);
                       visited.insert(nbr);
                     }
                 }
           }
       }
    }
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        unordered_map<int,list<int>>graph;
        for(auto x: prerequisites){
           int src  = x[0];
           int dest = x[1];
           graph[src].push_back(dest);
        }
        vector<int> ans;
        topoBFS(graph,ans,numCourses);
        reverse(ans.begin(),ans.end());
        if(ans.size()==numCourses) return ans;
        else return {};
    }
};


/////////////////////////// 695. Max Area of Island ////////////////

class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
    int m;
    int n;
    void bfs(vector<vector<int>>& grid,  queue<pair<int,int>>&q,int &area){
          while(!q.empty()){
             auto cell = q.front();
             int i =cell.first;
             int j =cell.second;
             grid[i][j]=0;
             q.pop();
             area++;
             for(int d =0;d<4;d++){
                int newi=i+dir[d][0];
                int newj =j+dir[d][1];
                if(newi>=0 && newj>=0 && newi<m && newj<n && grid[newi][newj]==1){
                    q.push({newi,newj});
                    grid[newi][newj]=0;
                }
             }
          }
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
         m = grid.size();
         n = grid[0].size();
         int maxArea=0;
         queue<pair<int,int>>q;
         for(int i=0;i<m;i++){
            for(int j =0;j<n;j++){
               if(grid[i][j]==1){
                  int area = 0;
                  q.push({i,j});
                  bfs(grid,q,area);
                  maxArea = max(maxArea,area); //1 
               }
            }
         }
         return maxArea;
    }
};



///////////////////////////////////////// DFS////////////////////////////////

class Solution {
public:
    vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
    int m;
    int n;
    void dfs(vector<vector<int>>& grid,int &area,int i , int j){
      grid[i][j]=0;
      area++;
      for(int d =0;d<4;d++){
         int newi = i+dir[d][0];
         int newj = j+dir[d][1];
         if(newi>=0 && newj>=0 && newi<m && newj <n && grid[newi][newj]==1){
            dfs(grid,area,newi,newj);
         } 
      }
          
    }
    int maxAreaOfIsland(vector<vector<int>>& grid) {
         m = grid.size();
         n = grid[0].size();
         int maxArea=0;
         queue<pair<int,int>>q;
         for(int i=0;i<m;i++){
            for(int j =0;j<n;j++){
               if(grid[i][j]==1){
                  int area = 0;
                  q.push({i,j});
                  dfs(grid,area,i,j);
                  maxArea = max(maxArea,area); //1 
               }
            }
         }
         return maxArea;
    }
};

/////////////////////2658. Maximum Number of Fish in a Grid//////////////


///////////// multisource dfs/bfs////////////

class Solution {
public:
    vector<vector<int>> dir ={{1,0},{-1,0},{0,1},{0,-1}};
    int m;
    int n;
    void bfs(vector<vector<int>>& grid,queue<pair<int,int>>&q,int &ans){
       while(!q.empty()){
          auto cell = q.front();
          q.pop();
          int i = cell.first;
          int j = cell.second;
          ans +=grid[i][j];
          grid[i][j]=0;
          for(int d=0;d<4;d++){
              int newi = i+dir[d][0];
              int newj  =j+dir[d][1];
              if(newi>=0 && newj>=0 && newi<m && newj<n && grid[newi][newj]>0){
                 q.push({newi,newj});
              }
          }
       }
    }
    int findMaxFish(vector<vector<int>>& grid) {
         m = grid.size();
         n =grid[0].size();
         int maxans =0;
         queue<pair<int,int>>q;
         for(int i =0;i<m;i++){
            for(int j =0;j<n;j++){
               if(grid[i][j]>0){
                  int ans = 0;
                  q.push({i,j});
                  bfs(grid,q,ans);
                  maxans=max(maxans,ans);
               }
            }
         }
         return maxans;
    }
};


class Solution {
public:
    vector<vector<int>> dir ={{1,0},{-1,0},{0,1},{0,-1}};
    int m;
    int n;
    void dfs(vector<vector<int>>& grid,int i , int j , int &ans){
         ans +=grid[i][j];
         grid[i][j]=0;
         for(int d =0;d<4;d++){
            int newi = i+dir[d][0];
            int newj  =j+dir[d][1];
            if(newi>=0 && newj >=0 && newi<m && newj<n && grid[newi][newj]>0){
               dfs(grid,newi,newj,ans);
            }
         }
    }
    int findMaxFish(vector<vector<int>>& grid) {
         m = grid.size();
         n =grid[0].size();
         int maxans =0;
         for(int i =0;i<m;i++){
            for(int j =0;j<n;j++){
               if(grid[i][j]>0){
                  int ans = 0;
                  dfs(grid,i,j,ans);
                  maxans=max(maxans,ans);
               }
            }
         }
         return maxans;
    }
};



/////////////////// 286 walls and gates//////////////////////

class Solution {
public:
void wallsAndGates(vector<vector<int>>& rooms) {
    const int row = rooms.size();
    if (0 == row) return;
    const int col = rooms[0].size();
    queue<pair<int, int>> canReach;  // save all element reachable
    vector<pair<int, int>> dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}; // four directions for each reachable
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(0 == rooms[i][j])
                canReach.emplace(i, j);
        }
    }
    while(!canReach.empty()){
        int r = canReach.front().first, c = canReach.front().second;
        canReach.pop();
        for (auto dir : dirs) {
            int x = r + dir.first,  y = c + dir.second;
            // if x y out of range or it is obstasle, or has small distance aready
            if (x < 0 || y < 0 || x >= row || y >= col || rooms[x][y] <= rooms[r][c]+1) continue;
            rooms[x][y] = rooms[r][c] + 1;
            canReach.emplace(x, y);
        }
    }
 }

};

///////////////////// my method to write the code//////////////

class Solution {
public:
vector<vector<int>> dir = {{1,0},{-1,0},{0,1},{0,-1}};
int m;
int n;
void bfs(vector<vector<int>>& rooms,queue<pair<int,int>>&q){
     while(!q.empty()){
         auto cell = q.front();
         int i = cell.first;
         int j = cell.second;
         q.pop();
         for(int d =0;d<4;d++){
             int newi = i+dir[d][0];
             int newj = j+dir[d][1];
             if(newi>=0 && newj>=0 && newi<m && newj<n && rooms[newi][newj]==INT_MAX){
                  rooms[newi][newj] = rooms[i][j]+1;
                  q.push({newi,newj});
             }
         }
     }
}
void wallsAndGates(vector<vector<int>>& rooms) {
    m = rooms.size();
    n = rooms[0].size();
    queue<pair<int,int>>q;
    for(int i =0;i<m;i++){
         for(int j =0;j<n;j++){
             if(rooms[i][j]==0){
                q.push({i,j});
                bfs(rooms,q);
             }
         }
    }

 }

};



////////////////815. Bus Routes////////////
/////////////// shortage bridge//////////////////////
////////////////// 1559. Detect Cycles in 2D Grid ///////////
//////////2316. Count Unreachable Pairs of Nodes in an Undirected Graph
/////////////947. Most Stones Removed with Same Row or Column
//////////////////////////////// 1034. Coloring A Border////////////////////////////
/////////////////////////////////clone the graph///////////////////////////////
////////////269. Alien Dictionary///////////////////
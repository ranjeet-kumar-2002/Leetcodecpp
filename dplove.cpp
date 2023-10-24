       /////////////////////////////fibbnachi////////////////////////////////////////

#include<bits/stdc++.h>
using namespace std;
int c =0; // 15 if the fib(6);
vector<int> memo;
int fib(int n){
     c++;
    if(n<=2) return 1;
    return fib(n-1)+fib(n-2);
}
int main(){
   cout<<fib(6)<<endl;
   cout<<c<<endl;
}

// // /////////////////memoaziation tc O(n) and sc O(n)1D DP////////////////////////////////

// step1: create dp array ans inialise and pass into the function
// step2: store the ans in dp array
// step3:check the dp array has already ans the return from dp array

 #include<bits/stdc++.h>
using namespace std;
int fib(int n,vector<int> &dp ){
    if(n<=1) return n; // base case
    if(dp[n] !=-1) return dp[n]; // if ans is stored  then return from dp array 
  return  dp[n] =fib(n-1,dp)+fib(n-2,dp); // store the ans and return
}
int main(){
   int n;
   cin>>n;
   vector<int> dp(n+1,-1);
   cout<<fib(n,dp);
   
}


// ////////////////////tabulation tc O(n) and sc O(n) // from buttom 0 to top n//////////////////////

// step1: crete dp array
//step2:change base case into inialization
// step3: change recursive code into iterative

#include<bits/stdc++.h>
using namespace std;
vector<int> memo;
int fib(int n){
   vector<int> dp(n+1,0);
   // base case
   dp[0]=1;
   dp[1]=1;
   for(int i =2;i<=n;i++){
       dp[i] =dp[i-1]+dp[i-2];
   }
   return dp[n];
}
int main(){
   int n;
   cin>>n;
   cout<<fib(n);
}


///////////////////// coin change ///////////////////////////////////
///////////////////////////// recursinve///////////////////////////////////

class Solution {
public:
    int solve(vector<int> & coins , int amount){
         if(amount==0) return 0;
         if(amount<0) return INT_MAX;
         int mini =INT_MAX;
         for(int i =0;i<coins.size();i++){
            int ans = solve(coins , amount-coins[i]);
             if(ans !=INT_MAX){
                mini = min(ans+1,mini); // because when we are return thenwe have to add one coin
             }
         }
         return mini;
    }
    int coinChange(vector<int>& coins, int amount) {
       int ans = solve(coins , amount);
       if(ans ==INT_MAX) return -1;
       else return ans;
    }
};

///////////////////////////////////memo////////////////////////////////////

class Solution {
public:
   int  solveMemo(vector<int> &coins, int amount,vector<int>&dp){
        if(amount==0) return 0;
        if(amount < 0) return INT_MAX;
        if(dp[amount] !=-1) return dp[amount];
        int mini =INT_MAX;
        for(int i =0;i<coins.size();i++){
            int ans =  solveMemo(coins, amount-coins[i],dp);
            if(ans !=INT_MAX){
               mini = min(ans+1, mini);
            }
        }
        return dp[amount]=mini;
   }
    int coinChange(vector<int>& coins, int amount) {
         vector<int> dp(10005,-1);
         int ans = solveMemo(coins, amount,dp);
         if(ans !=INT_MAX) return ans;
         else return -1;
        
    }
};
//////////////////////////////// tabulation////////////////////////////////////////////



class Solution {
public:
    int solveTab(vector<int> & coins,int amount){
        //step1: create vector and inilize
        vector<int> dp(10005,INT_MAX);
        // step2: change the base cse into initialization
        dp[0]=0;
        // step3: recusive code into iterative
        for(int target =1;target<=amount;target++){ // here i == target for each time or amount
          int mini =INT_MAX;
           for(int i =0;i<coins.size();i++){
            if(target-coins[i]>=0){
                 int ans =  dp[target-coins[i]];
                 if(ans !=INT_MAX) mini = min(ans+1, mini);
          }
            
        }
            dp[target]=mini;
        }
        return dp[amount];
    }
    int coinChange(vector<int>& coins, int amount) {
        int ans = solveTab(coins,amount);
         if(ans !=INT_MAX) return ans;
          else return -1;
    }
};


////////////////Max Sum Of Non-Adjacent Elements/ house robber (leetcode)////////1D DP

//////////////////// recursive code////////////////////

class Solution {
public:
    int f(vector<int> & nums,int n){ // n-> last index 
    if(n<0) return 0;
    if(n==0) return nums[0];
    int enclude = nums[n]+f(nums,n-2);
    int exclude = 0+f(nums,n-1);
    return max(enclude,exclude);
         
    }
    int rob(vector<int>& nums) {
       int n = nums.size()-1;
       return f(nums,n);
    }
};

////////////////////// memoization/////////////////////////////////////

class Solution {
public:
    int f(vector<int> & nums,int n, vector<int> &dp){ // n-> last index 
    if(n<0) return 0;
    if(n==0) return nums[0];
    if(dp[n] !=-1) return dp[n];
    int enclude = nums[n]+f(nums,n-2,dp);
    int exclude = 0+f(nums,n-1,dp);
    dp[n]= max(enclude,exclude);
    return dp[n];
         
    }
    int rob(vector<int>& nums) {
       int n = nums.size()-1;
       vector<int> dp(105,-1);
       return f(nums,n,dp);
    }
};

//////////////////////// tabulation//////////////////////////////


class Solution {
public:
    int f(vector<int> & nums,int n){ // n-> last index 
    vector<int> dp(105,0); // s
    dp[0] =nums[0];
    for(int i =1;i<=n;i++){ // in memoization  going from end to 0 in this vice virsa
         int temp =0;
         if(i-2>=0)  temp = dp[i-2];
         int enclude = nums[i]+temp;
         int exclude = dp[i-1]+0;
         dp[i]= max(enclude,exclude);  
 }
      return dp[n];
}
    int rob(vector<int>& nums) {
        int n = nums.size()-1;
       return f(nums,n);
    }
};



// ////////////////////space optimation///////////////////////////////

#include<bits/stdc++.h>
using namespace std;
int maxsum(vector<int>& v){
   int n = v.size();
   int next1=0;
   int next2=0;
   int curr;
   for (int i = n - 1; i >= 0; i--) {
      int incl = v[i] + next2;;
      int excl = next1;
       curr = max(incl, excl);
       next2=next1;
       next1=curr;
   }
   return curr;
}
int main(){
    vector<int>v;
    v.push_back(2);
    v.push_back(1);
    v.push_back(4);
    v.push_back(9);
    cout<<maxsum(v);
}


/////////////////////////////////// paint fences(gfg)//////////////////////////

// Given a fence with n posts and k colors, find out the number of ways of painting the fence such that at most 2 adjacent posts have the same color. Since the answer can be large return it modulo 10^9 + 7


#include<bits/stdc++.h>
#define MOD 1000000007
using namespace std;
long long f(int n , int k){
       if(n==1) return k%MOD;
       if(n==2) return k%MOD+k*(k-1)%MOD;
       return (f(n-2,k)%MOD+f(n-1,k)%MOD)*(k-1)%MOD;
}
int main(){
       int n=4 ;
       int k=3;
       cout<<f(n,k);
}

//////////////////////////// topdown////////////////////////////

#define MOD 1000000007
class Solution{
    public:
    long long f(int n , int k,vector<long long>&dp){
         if(n==1) return k%MOD;
         if(n==2 ) return k%MOD+k*(k - 1) % MOD;
         if(dp[n] !=-1) return dp[n];
         return  dp[n]=(f(n-2,k,dp)%MOD+f(n-1,k,dp)%MOD)*(k-1)%MOD;
    }
    long long countWays(int n, int k){
        vector<long long>dp(5005,-1);
        return f(n,k,dp);
    }
};



#define MOD 1000000007
class Solution{
    public:
    long long f(int n , int k){
         vector<long long>dp(5005,0);
         dp[1]=k%MOD;
         dp[2]=k%MOD+k*(k - 1) % MOD;
         
         for(int i =3;i<=n;i++){
             dp[i]=(dp[i-2]%MOD+dp[i-1]%MOD)*(k-1)%MOD;
         }
         return dp[n];
    }
    long long countWays(int n, int k){
        return f(n,k);
    }
};

////////////////////////0 - 1 Knapsack Problem (gfg)/////////////////////////////////////////
//////////////////// recursive code//////////////////////////////////

class Solution
{
public:
    int f(int wt[], int val[], int W, int n) {
        // BASE CASE
        if (W == 0 || n == 0) return 0;
        if (wt[n - 1] <= W) {
            return max(val[n - 1] + f(wt, val, W - wt[n - 1], n - 1), f(wt, val, W, n - 1));
        } else{
            return f(wt, val, W, n - 1);
        }
    }
    int knapSack(int W, int wt[], int val[], int n) {
        return f(wt, val, W, n);
    }
};

///////////////////////////// memoization///////////////////////

class Solution
{
public:
     vector<vector<int>> dp;
    int f(int wt[], int val[], int W, int n) {
        // BASE CASE
        if (W == 0 || n == 0) return 0;
        if(dp[W][n] !=-1) return dp[W][n];
        if (wt[n - 1] <= W) {
            return  dp[W][n]=max(val[n - 1] + f(wt, val, W - wt[n - 1], n - 1), f(wt, val, W, n - 1));
        } else{
            return dp[W][n]= f(wt, val, W, n - 1);
        }
    }
    int knapSack(int W, int wt[], int val[], int n) {
        dp.clear();
        dp.resize (1001,vector<int>(1001,-1));
        return f(wt, val, W, n);
    }
};

/////////////////////  tabulation////////////////////////////////

class Solution {
public:
    vector<vector<int>> dp;
    int knapSack(int W, int wt[], int val[], int n) {
        dp.resize(1001, vector<int>(1001, 0));

        for (int i = 1; i <= n; i++) {  // val
            for (int j = 0; j <= W; j++) {  // wt
                if (wt[i - 1] <= j) {
                    dp[i][j] = max(val[i - 1] + dp[i - 1][j - wt[i - 1]], dp[i - 1][j]);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[n][W];
    }
};


////////////////////// subset sum////////////////////////////////

class Solution{   
public:
    bool f(vector<int>&nums , int indx,int sum){
         int n = nums.size();
         if(indx>=n) return false;
         if(sum<0) return false;
         if(sum==0) return true;
         bool enclude = f(nums,indx+1,sum-nums[indx]);
         bool exclude = f(nums,indx+1,sum);
         return (enclude||exclude);
    }
    bool isSubsetSum(vector<int>arr, int sum){
      return f(arr,0,sum);
    }
};

class Solution {
public:
    vector<vector<int>> dp;

    bool f(vector<int>& nums, int i, int target) {
        int n = nums.size();
        if(i>=n) return target ==0 ;
        if(target == 0) return true; 
        if(target<0) return false; 
        if (dp[i][target] != -1) return dp[i][target];
        bool include = f(nums, i + 1, target - nums[i]);
        bool exclude = f(nums, i + 1, target);
        dp[i][target] = (include || exclude);
        return dp[i][target];
    }

    bool isSubsetSum(vector<int> arr, int sum) {
        dp.resize(arr.size(), vector<int>(sum + 1, -1));
        return f(arr, 0, sum);
    }
};


////////////////////////// partition equal sum//////////////////////////////////
                    /// reursive code//////////////////
class Solution {
public:
    bool f(vector<int>& nums,int i , int target){
         int n = nums.size();
         if(i>=n) return false;
         if(target == 0) return true; // also we add condi but not// if(target<0) return false;
         bool include = f(nums,i+1,target-nums[i]);
         bool exclude = f(nums,i+1,target);
         return (include || exclude);
    }
    bool canPartition(vector<int>& nums) {
        int sum =0;
        for(auto x:nums) sum +=x;
        if(sum%2 !=0) return false;
        int target = sum/2;
        return f(nums,0,target);
    }
};

                ////////////memoization/////////////////////////
class Solution {
public:
    vector<vector<int>> dp;
    bool f(vector<int>& nums,int i , int target){
         int n = nums.size();
         if(i>=n) return false;
         if(target == 0) return true; 
         if(target<0) return false;  // also we add condi but not neccesary
         if(dp[i][target]!=-1) return dp[i][target];
         bool include = f(nums,i+1,target-nums[i]);
         bool exclude = f(nums,i+1,target);
         dp[i][target] =(include || exclude);
         return  dp[i][target];
    }
    bool canPartition(vector<int>& nums) {
        int sum =0;
        for(auto x:nums) sum +=x;
        if(sum%2 !=0) return false;
        int target = sum/2;
        dp.resize(200,vector<int>(target+1,-1)); //also dp.resize(205,vector<int>(target+5,-1)); 
        return f(nums,0,target);
    }
};

             //////// tabulation///////////////////////

class Solution {
public:
    vector<vector<int>> dp;
    bool f(vector<int>& nums, int target){ 
        int n = nums.size();
    dp.resize(201,vector<int>(target+1,0));
    for(int i =0;i<n;i++){
         dp[i][0] =1;
    }
    for(int indx =n-1;indx>=0;indx--){ // opposite travesal;
         for(int t = 1;t<=target;t++){ // target will start from 1 becase col 0 already filled
         bool include = false;
         if (t - nums[indx] >= 0) include = dp[indx + 1][t - nums[indx]];
         bool exclude = dp[indx + 1][t];
         dp[indx][t] = (include || exclude);
       }
    }
     return  dp[0][target];
        
    }
    bool canPartition(vector<int>& nums) {
        int sum =0;
        for(auto x:nums) sum +=x;
        if(sum%2 !=0) return false;
        int target = sum/2;
        return f(nums,target);
    }
};

///////////////////// 1155. Number of Dice Rolls With Target Sum///////////////////////////
             // recursion///

#define MOD 1000000007
class Solution {
public:
    int f(int n, int k, int target){
         if(n==0 && target ==0) return 1;
         if(target<0) return 0;
         if(n!=0 && target==0) return 0;
         if(target !=0&& n==0) return 0;
         int ans =0;
         for(int i =1;i<=k;i++){
            ans = ans%MOD+f(n-1,k,target-i)%MOD;
         }
          return ans%MOD;
    }
    int numRollsToTarget(int n, int k, int target) {
        return f(n,k,target);
    }
};

            // memoization//

#define MOD 1000000007
class Solution {
public:
    vector<vector<int>> dp;
    int f(int n, int k, int target){
         if(n==0 && target ==0) return 1;
         if(target<0) return 0;
         if(n!=0 && target==0) return 0;
         if(target !=0&& n==0) return 0;
         if(dp[n][target] !=-1) return dp[n][target];
         int ans =0;
         for(int i =1;i<=k;i++){
            ans = ans%MOD+f(n-1,k,target-i)%MOD;
         }
          return  dp[n][target] = ans%MOD;
    }
    int numRollsToTarget(int n, int k, int target) {
        dp.clear();
        dp.resize(31,vector<int>(target+1,-1));
        return f(n,k,target);
    }
};

              /// tabulation///////
              
#define MOD 1000000007
class Solution {
public:
      vector<vector<int>> dp;
    int f(int n, int k, int target){
         dp.resize(n+1,vector<int>(target+1,0));
        dp[0][0]=1;
        for(int i =1;i<=n;i++){
            for(int j =0;j<=target;j++){
            int ans =0;
            for(int x =1;x<=k;x++){
                if(i-1>=0 && j-x>=0) ans = (ans+dp[i-1][j-x])%MOD;
         }
           dp[i][j] = ans;    
     }
 }
 return dp[n][target];
}
    int numRollsToTarget(int n, int k, int target) {
        return f(n,k,target);
    }
};



///////////################## 375. Guess Number Higher or Lower II#########################pragma endregion
                      // recursion///////////
class Solution {
public:
    int f(int start, int end){
        if(start>=end) return 0;
        int ans =INT_MAX;
        for(int i =start;i<=end;i++){
            ans = min(ans,i+max(f(start,i-1),f(i+1,end)));
        }
          return ans;
    }
    int getMoneyAmount(int n) {
        return f(1,n);
    }
};
                  // memoization/////////////

class Solution {
public:
    vector<vector<int>>dp;
    int f(int start, int end){
        if(start>=end) return 0;
        int ans =INT_MAX;
        if(dp[start][end] !=-1) return dp[start][end];
        for(int i =start;i<=end;i++){
            ans = min(ans,i+max(f(start,i-1),f(i+1,end)));
        }
         dp[start][end]= ans;
         return dp[start][end];
    }
    int getMoneyAmount(int n) {
        dp.resize(201,vector<int>(201,-1));
        return f(1,n);
    }
};
            /// tabulation//////

class Solution {
public:
    vector<vector<int>>dp;
    int f(int n){
        dp.resize(205,vector<int>(205,0));

        for(int start=n;start>=1;start--){
            for(int end =1;end<=n;end++){
                 if(start>=end) continue;
                 else {
                    int ans =INT_MAX;
                    for(int i =start;i<=end;i++){
                    ans = min(ans,i+max(dp[start][i-1],dp[i+1][end]));
                 }
                dp[start][end]= ans;
            }
           
        }
    } 
        return dp[1][n];
    }
    int getMoneyAmount(int n) {
        return f(n);
    }
};

///////////////////////////////  1130. Minimum Cost Tree From Leaf Values/////////////////////////
                  // recursion///
class Solution {
public:
    int f(vector<int>&arr,  map<pair<int,int>,int> &maxi,int left,int right){
         if(left==right) return 0;
          int ans =INT_MAX;
         for(int i = left;i<right;i++){
          ans = min(ans,maxi[{left,i}]*maxi[{i+1,right}]+f(arr,maxi,left,i)+f(arr,maxi,i+1,right));
              
         }
         return  ans;
    }

    int mctFromLeafValues(vector<int>& arr) {
        map<pair<int,int>,int> maxi;
        for(int i =0;i<arr.size();i++){
             maxi[{i,i}] =arr[i];
             for(int j =i+1;j<arr.size();j++){
                 maxi[{i,j}] = max(arr[j],maxi[{i,j-1}]);
             }
        }

          for(int i =0;i<arr.size();i++){
             for(int j =i;j<arr.size();j++){
                 cout<<"for range"<<i<<"->"<<j<<"maxi is"<<maxi[{i,j}]<<endl;
                //for range0->0maxi is6
                // for range0->1maxi is6
                // for range0->2maxi is6
                // for range1->1maxi is2
                // for range1->2maxi is4
                // for range2->2maxi is4
             }
        }


        int n = arr.size();
        return f(arr,maxi,0,n-1);
    }
};


            /// memoization//

class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&arr,  map<pair<int,int>,int> &maxi,int left,int right){
         if(left==right) return 0;
         if(dp[left][right] !=-1) return dp[left][right];
          int ans =INT_MAX;
         for(int i = left;i<right;i++){
          ans = min(ans,maxi[{left,i}]*maxi[{i+1,right}]+f(arr,maxi,left,i)+f(arr,maxi,i+1,right));
              
         }
         dp[left][right]= ans;
         return dp[left][right];
    }

    int mctFromLeafValues(vector<int>& arr) {
        map<pair<int,int>,int> maxi;
        for(int i =0;i<arr.size();i++){
             maxi[{i,i}] =arr[i];
             for(int j =i+1;j<arr.size();j++){
                 maxi[{i,j}] = max(arr[j],maxi[{i,j-1}]);
             }
        }

        int n = arr.size();
        dp.resize(45,vector<int>(45,-1));
        return f(arr,maxi,0,n-1);
    }
};



            // tabulation//

class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&arr,  map<pair<int,int>,int> &maxi){
          int n = arr.size();
          dp.resize(n+1,vector<int>(n+1,0));
          for(int left=n-1;left>=0;left--){
               for(int right = 0;right<=n-1;right++){
                    if(left>=right) continue; // base case
                    else{
                    int ans =INT_MAX;
                    for(int i = left;i<right;i++){
                    ans = min(ans,maxi[{left,i}]*maxi[{i+1,right}]
                    +dp[left][i]+dp[i+1][right]);
                    }
                     dp[left][right]= ans;
                  }
                   
               }
          }
          return dp[0][n-1];
      
    }

    int mctFromLeafValues(vector<int>& arr) {
        map<pair<int,int>,int> maxi;
        for(int i =0;i<arr.size();i++){
             maxi[{i,i}] =arr[i];
             for(int j =i+1;j<arr.size();j++){
                 maxi[{i,j}] = max(arr[j],maxi[{i,j-1}]);
             }
        }
        return f(arr,maxi);
    }
};


///////////////////////// LCS   ///////////////////////////////////
           /////////////////////// recursion ///////////////////////////
          // note when text1[i]!=text2[j] there 4 possibility but two we don't need to consider
          // f(text1,text2,i+1,j),f(text1,text2,i,j+1),f(text1,text2,i+1,j+1),f(text1,text2,i,j);

class Solution {
public: 
    int f(string &text1,string &text2,int i , int j){
         if(i==text1.size() || j==text2.size()) return 0;
         if(text1[i]==text2[j]) return 1+f(text1,text2,i+1,j+1);
         else return max(f(text1,text2,i+1,j),f(text1,text2,i,j+1));  
    }
    int longestCommonSubsequence(string text1, string text2) {
           return f(text1,text2,0,0);
    }
};


class Solution {
public: 
    int f(string &text1,string &text2,int i , int j){
         if(i==text1.size() || j==text2.size()) return 0;
         int ans =0;
         if(text1[i]==text2[j]) ans= 1+f(text1,text2,i+1,j+1);
         else  ans =0+max(f(text1,text2,i+1,j),f(text1,text2,i,j+1)); 
         return ans; 
    }
    int longestCommonSubsequence(string text1, string text2) {
           return f(text1,text2,0,0);
    }
};

       /////////////////////// memoization/top-down ////////////////////////
       
class Solution {
public: 
vector<vector<int>> dp;
    int f(string &text1,string &text2,int i , int j){
         if(i==text1.size() || j==text2.size()) return 0;
         if(dp[i][j] !=-1) return dp[i][j];
         if(text1[i]==text2[j])  return dp[i][j] = 1+f(text1,text2,i+1,j+1);
         else dp[i][j] = 0+max(f(text1,text2,i+1,j),f(text1,text2,i,j+1));
         return dp[i][j];
    }
    int longestCommonSubsequence(string text1, string text2) {
           dp.clear();
           dp.resize(1005,vector<int>(1005,-1));
           return f(text1,text2,0,0);
    }
};


  
class Solution {
public: 
vector<vector<int>> dp;
    int f(string &text1,string &text2,int i , int j){
         if(i==text1.size() || j==text2.size()) return 0;
         int ans =0;
         if(dp[i][j] !=-1) return  dp[i][j];
         if(text1[i]==text2[j])  ans = dp[i][j] = 1+f(text1,text2,i+1,j+1);
         else ans = 0+max(f(text1,text2,i+1,j),f(text1,text2,i,j+1));
         dp[i][j] = ans;
         return  dp[i][j];
    }
    int longestCommonSubsequence(string text1, string text2) {
           dp.clear();
           dp.resize(1005,vector<int>(1005,-1));
           return f(text1,text2,0,0);
    }
};


////////////////////////// tabulation///////////////////////////

   
class Solution {
public: 
vector<vector<int>> dp;
    int f(string &text1,string &text2,int i , int j){
         for(int i =text1.size()-1;i>=0;i--){
              for(int j = text2.size()-1;j>=0;j--){
               int ans =0;
               if(text1[i]==text2[j]) ans =  1+dp[i+1][j+1];
               else ans = max(dp[i+1][j],dp[i][j+1]);
               dp[i][j]=ans;
           }
      }
        
        return dp[0][0];
}
    int longestCommonSubsequence(string text1, string text2) {
           dp.clear();
           dp.resize(1005,vector<int>(1005,0));
           return f(text1,text2,0,0);
    }
};

//////////////////////////////  516. Longest Palindromic Subsequence//////////////////////
                 // recursion//
class Solution {
public:
    int f(string s1,string s2,int i ,int j){
        if(i==s1.size() || j==s2.size()) return 0;
        int ans =0;
        if(s1[i]==s2[j]){
             ans = 1+f(s1,s2,i+1,j+1);
        } else{
             ans =0+max(f(s1,s2,i+1,j),f(s1,s2,i,j+1));
        }
        return ans;
    }
    int longestPalindromeSubseq(string s) {
        string s1 = s;
        reverse(s1.begin(), s1.end());
        return f(s,s1,0,0);
    }
};

                 // top-down///
class Solution {
public:
    vector<vector<int>>dp;
    int f(string &s1,string &s2,int i ,int j){
        if(i==s1.size() || j==s2.size()) return 0;
        if(dp[i][j] !=-1) return dp[i][j];
        int ans =0;
        if(s1[i]==s2[j]){
             ans = 1+f(s1,s2,i+1,j+1);
        } else{
             ans =0+max(f(s1,s2,i+1,j),f(s1,s2,i,j+1));
        }
       dp[i][j] = ans;
       return dp[i][j];
    }
    int longestPalindromeSubseq(string s) {
        string s1 = s;
        reverse(s1.begin(), s1.end());
        dp.clear();
        dp.resize(1005,vector<int>(1005,-1));
        return f(s,s1,0,0);
    }
};
               /// buttom-up////

class Solution {
public:
    vector<vector<int>>dp;
    int f(string &s1,string &s2){
        dp.resize(1005,vector<int>(1005,0)); // base case done
        for(int i =s1.size()-1;i>=0;i--){
             for(int j =s2.size()-1;j>=0;j--){
                int ans =0;
                 if(s1[i]==s2[j]) ans = 1+dp[i+1][j+1];
                 else ans =0+max(dp[i+1][j],dp[i][j+1]);
                dp[i][j] = ans;
             }
        }
        return dp[0][0];
       
    }
    int longestPalindromeSubseq(string s) {
        string s1 = s;
        reverse(s1.begin(), s1.end());
        return f(s,s1);
    }
};


/////////////////////////// 72. Edit Distances///////////////////////

                        // recursion//
class Solution {
public: 
    int f(string &s1,string &s2,int i , int j){
         if(i==s1.size()) return s2.size()-j; 
         if(j==s2.size()) return s1.size()-i; 
         int ans =0;
         if(s1[i]==s2[j]){
              ans = 0+f(s1,s2,i+1,j+1);
         } else{
             int in = 1+f(s1,s2,i+1,j);
             int re = 1+f(s1,s2,i,1+j);
             int de = 1+f(s1,s2,i+1,1+j);
             ans = min(in,min(re,de));
         }
            return ans;
    }
    int minDistance(string word1, string word2) {
        if(word1.size()==0) return word2.size();
        if(word2.size()==0) return word1.size();
        return f(word1,word2,0,0);
    }
};


                //memoization//
class Solution {
public: 
    vector<vector<int>>dp;
    int f(string &s1,string &s2,int i , int j){
         if(i==s1.size()) return s2.size()-j; 
         if(j==s2.size()) return s1.size()-i; 
         if(dp[i][j] !=-1) return dp[i][j];
         int ans =0;
         if(s1[i]==s2[j]){
              ans = 0+f(s1,s2,i+1,j+1);
         } else{
             int in = 1+f(s1,s2,i+1,j);
             int re = 1+f(s1,s2,i,1+j);
             int de = 1+f(s1,s2,i+1,1+j);
             ans = min(in,min(re,de));
         }
         dp[i][j] = ans;
         return dp[i][j];
    }
    int minDistance(string word1, string word2) {
        if(word1.size()==0) return word2.size();
        if(word2.size()==0) return word1.size();
        dp.resize(505,vector<int>(505,-1));
        return f(word1,word2,0,0);
    }
};
                    // tabulation/buttom-up//
class Solution {
public:  
    vector<vector<int>>dp;
    int f(string &s1,string &s2){
         dp.resize(505,vector<int>(505,-1));
         for(int j =0;j<=s2.size();j++){
             dp[s1.size()][j] = s2.size()-j;
         }

           for(int i =0;i<=s1.size();i++){
             dp[i][s2.size()] = s1.size()-i;
         }

         for(int i =s1.size()-1;i>=0;i--){
              for(int j =s2.size()-1;j>=0;j--){
                    int ans =0;
                    if(s1[i]==s2[j]){
                        ans = 0+dp[i+1][j+1];
                    } else{
                        int in = 1+dp[i+1][j];
                        int re = 1+dp[i][1+j];
                        int de = 1+dp[i+1][1+j];
                        ans = min(in,min(re,de));
                    }
                    dp[i][j] = ans;

              }
         }
         return dp[0][0];
    }
    int minDistance(string word1, string word2) {
        if(word1.size()==0) return word2.size();
        if(word2.size()==0) return word1.size();
        return f(word1,word2);
    }
};

////////////////////////// LIS/////////////////////


//////////////////// method-1////////////////////////////

class Solution {
public:
    int f(int i,vector<int> & nums ){
         if(i==0) return 1;
         int mx = INT_MIN;
         for(int j = 0;j<=i-1;j++){
              if(nums[j]<nums[i]){
                 mx = max(mx,f(j,nums));
              }
         }
         if(mx==INT_MIN) return 1;
         return 1+mx;
    }
    int lengthOfLIS(vector<int>& nums) {
        int maxi = INT_MIN;
        for(int i =0;i<nums.size();i++){
              maxi = max(maxi,f(i,nums));
        }
        return maxi;
    }
};


class Solution {
public:
    vector<int> dp;
    int f(int i,vector<int> & nums ){
         if(i==0) return 1;
         if(dp[i] !=-1) return dp[i];
         int mx = INT_MIN;
         for(int j = 0;j<=i-1;j++){
              if(nums[j]<nums[i]){
                 mx = max(mx,f(j,nums));
              }
         }
         if(mx==INT_MIN) return 1; // imp 
         return dp[i] = 1+mx;
    }
    int lengthOfLIS(vector<int>& nums) {
        dp.clear();
        dp.resize(2500,-1);
        int maxi = INT_MIN;
        for(int i =0;i<nums.size();i++){
              maxi = max(maxi,f(i,nums));
        }

        return maxi;
    }
};


class Solution {
public:
    vector<int> dp;
    int lengthOfLIS(vector<int>& nums) {
        dp.clear();
        dp.resize(2500,-1);
        int maxi = INT_MIN;
        for(int i =0;i<nums.size();i++){
            for(int j =0;j<=i-1;j++){
                if(nums[j]<nums[i]){
                     dp[i]= max(dp[i],1+dp[j]);
                }
            }
            if(dp[i]==-1) dp[i]=1;
            maxi = max(maxi,dp[i]);  
        }

        return maxi;
    }
};

//////////////////// method-2////////////////////////////

class Solution {
public:
    int f(vector<int> & nums ,int curr ,int prev){
         if(curr==nums.size()) return 0;
        int include =0;
        if(prev==-1 || nums[prev]<nums[curr]){
             include =1+f(nums,curr+1,curr);
        }
        int exclude = 0+f(nums,curr+1,prev);
        return max(include,exclude);
    }
    int lengthOfLIS(vector<int>& nums) {
       int curr =0;
       int prev=-1;
       return f(nums,curr,prev);
    }
};



class Solution {
public:
    int f(vector<int> & nums ,int curr ,int prev , vector<vector<int>> &dp){
         if(curr==nums.size()) return 0;
         if(dp[curr][prev+1] !=-1) return dp[curr][prev+1];
        int include =0;
        if(prev==-1 || nums[prev]<nums[curr]){
             include =1+f(nums,curr+1,curr, dp);
        }
        int exclude = 0+f(nums,curr+1,prev,dp);
        return dp[curr][prev+1] = max(include,exclude);
    }
    int lengthOfLIS(vector<int>& nums) {
       int n = nums.size();
       int curr =0;
       int prev=-1;
       vector<vector<int>> dp(2500,vector<int>(2501,-1));
       // or  vector<vector<int>> dp(n,vector<int>(n+1,-1));
       return f(nums,curr,prev ,dp);
    }
};





class Solution {
public:
    int f(vector<int> & nums ,int curr ,int prev){
        int n = nums.size();
        vector<vector<int>> dp(n+1,vector<int>(n+1,0));
          for(int curr =n-1;curr>=0;curr--){
             for(int prev = curr-1;prev>=-1;prev--){
                    int include =0;
                    if(prev==-1 || nums[prev]<nums[curr]){
                    include =1+dp[curr+1][curr+1];
              }
                 int exclude = 0+dp[curr+1][prev+1];
                 dp[curr][prev+1] = max(include,exclude);
             }
        }
        return dp[0][0];
    }
    int lengthOfLIS(vector<int>& nums) {
       int n = nums.size();
       int curr =0;
       int prev=-1;
       return f(nums,curr,prev);
    }
};



//////////////////////////  dp with binary search///////////////

class Solution {
public:
    int f(vector<int> &nums){
        int n = nums.size();
        if(n==0) return 0;
        vector<int> ans;
        ans.push_back(nums[0]);
        for(int i =0;i<n;i++){
             if(nums[i]>ans.back()){
                 ans.push_back(nums[i]);

             }else{
                  // finding oerriding index
                  int indx = lower_bound(ans.begin(),ans.end(),nums[i])-ans.begin();
                  ans[indx] = nums[i];
             }
        }
      return ans.size();
        
    }
    int lengthOfLIS(vector<int>& nums) {
       return f(nums);
    }
};

//////////////// russian doll/////////////////


class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        for(auto &x:envelopes){   //[[5,4],[6,4],[6,7],[2,3]]
            sort(x.begin(),x.end());
            // 4 5 
            // 4 6 
            // 6 7 
            // 2 3 
        }

        sort(envelopes.begin(),envelopes.end());
        // 2 3 
        // 4 5 
        // 4 6 
        // 6 7 

              // printing //
        // for(int i =0;i<envelopes.size();i++){
        //     for(int j =0;j<envelopes[0].size();j++){
        //          cout<<envelopes[i][j]<<" ";
        //     }cout<<endl;
        // }

        return 0;
    }
};


////////// Time Limit Exceeded 85 / 87 testcases passed // memoization//

class Solution {
public:
    int f(int curr,int prev,vector<vector<int>>& envelopes,vector<vector<int>> &dp){
        if(curr==envelopes.size()){
            return 0;
        }
        if(dp[curr][prev+1]!=-1){
            return dp[curr][prev+1];
        }
        int notTake=0+f(curr+1,prev,envelopes,dp);
        int take=-1e9;
        if(prev==-1 or (envelopes[prev][0]<envelopes[curr][0] and envelopes[prev][1]     <envelopes[curr][1])){
                take=1+f(curr+1,curr,envelopes,dp);
        }
        return dp[curr][prev+1]=max(take,notTake);
    }
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(),envelopes.end());
        int n=envelopes.size();
        vector<vector<int>> dp(n+1,vector<int>(n+1,-1));
        return f(0,-1,envelopes,dp);
    }
};

////////// Time Limit Exceeded 85 / 87 testcases passed // tabulation

class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(),envelopes.end());
        int n=envelopes.size();
        vector<vector<int>> dp(n+1,vector<int>(n+1,0));
        for(int curr=n-1;curr>=0;curr--){
            for(int prev=curr-1;prev>=-1;prev--){
                int notTake=0+dp[curr+1][prev+1];
                int take=-1e9;
                if(prev==-1 or (envelopes[prev][0]<envelopes[curr][0] and envelopes[prev][1]<envelopes[curr][1])){
                        take=1+dp[curr+1][curr+1];
                }
             dp[curr][prev+1]=max(take,notTake); 
            }
        }
        return dp[0][0];
          
    }
};

//////////////////////////// final soln//////////////////////

class Solution {
public:
    static bool comp(vector<int> &a,vector<int>&b){
        if(a[0]==b[0]) return a[1]>b[1];
        else return a[0]<b[0];
    }
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(),envelopes.end(),comp);

        vector<int> temp;
        temp.push_back(envelopes[0][1]);

        for(int i=1;i<envelopes.size();i++){
            if(temp.back()<envelopes[i][1]) temp.push_back(envelopes[i][1]);
            else{
                auto ind=lower_bound(temp.begin(),temp.end(),envelopes[i][1])-temp.begin();
                temp[ind]=envelopes[i][1];
            }
        }
        return temp.size();
    }
};


///////////////1691 max height by stacking cuboids/////////////

                     /// basic//

class Solution {
public:
    int maxHeight(vector<vector<int>>& cuboids) {
        // sort every  array inside 2D array;
        for(auto &x:cuboids){ //[[50,45,20],[95,37,53],[45,23,12]]
            sort(x.begin(),x.end());  // 
            // 20 45 50 
            // 37 53 95 
            // 12 23 45 
        }
        sort(cuboids.begin(),cuboids.end()); // sort cuboids 
        // 12 23 45 
        // 20 45 50 
        // 37 53 95 
                          // printing the sorted cuboids
        // for(int i =0;i<cuboids.size();i++){
        //      for(int j =0;j<cuboids[0].size();j++){
        //           cout<<cuboids[i][j]<<" ";
        //      }
        //      cout<<endl;
        // }

        return 0;

    }
};

                    // recursion//
class Solution {
public:
    int find(vector<vector<int>>&nums,int curr,int prev){
        if(curr==nums.size()) return 0;
        int inc=0;
        if(prev==-1||(nums[prev][0]<=nums[curr][0]&&nums[prev][1]<=nums[curr][1]&&nums[prev][2]<=nums[curr][2])){ // width,length,height
            inc=nums[curr][2]+find(nums,curr+1,curr);
        }
        int exc=find(nums,curr+1,prev);
        return max(inc,exc);
    }
    int maxHeight(vector<vector<int>>&nums){
        for(auto &x:nums){
            sort(x.begin(),x.end());
        }
        sort(nums.begin(),nums.end());
        int curr =0,prev =-1;
        return find(nums,curr,prev);
    }
};

                // top-down//
class Solution {
public: 
    vector<vector<int>>dp;
    int find(vector<vector<int>>&nums,int curr,int prev){
        if(curr==nums.size()) return 0;
        if(dp[curr][prev+1] !=-1) return dp[curr][prev+1];
        int inc=0;
        if(prev==-1||(nums[prev][0]<=nums[curr][0]&&nums[prev][1]<=nums[curr][1]&&nums[prev][2]<=nums[curr][2])){ // width,length,height
            inc=nums[curr][2]+find(nums,curr+1,curr);
        }
        int exc=find(nums,curr+1,prev);
        dp[curr][prev+1] = max(inc,exc);
        return dp[curr][prev+1];
    }
    int maxHeight(vector<vector<int>>&nums){
        for(auto &x:nums){
            sort(x.begin(),x.end());
        }
        sort(nums.begin(),nums.end());
        int curr =0,prev =-1;
        dp.clear();
        dp.resize(105,vector<int>(105,-1));
        return find(nums,curr,prev);
    }
};

//*********************************Linear DP****************************************
////////////////////////////1.Perfect Squares Leetcode/////////////////////////////////////////

class Solution {
public:
    int f(int n){
        if(n==0) return 1;
        if(n<0) return 0;
        int ans =INT_MAX;
        for(int i =1;i<=sqrt(n);i++){
            int sq = i*i;
            int ans1 = 1+f(n-sq);
             ans =min(ans1,ans);
            
        }
        return ans;
    }
    int numSquares(int n) {
       return f(n)-1;
    }
};

class Solution {
public:
    vector<int>dp;
    int f(int n){
        if(n==0) return 1;
        if(n<0) return 0;
        if(dp[n] !=-1) return dp[n];
        int ans =INT_MAX;
        for(int i =1;i<=sqrt(n);i++){
            int sq = i*i;
            int ans1 = 1+f(n-sq);
             ans =min(ans1,ans);
            
        }
        dp[n]= ans;
        return dp[n];
    }
    int numSquares(int n) {
       dp.clear();
       dp.resize(10005,-1);
       return f(n)-1;
    }
};

class Solution {
public:
    vector<int>dp;
    int f(int n){
         dp.resize(10005,0);
         dp[0]=1;
        for(int i =1;i<=n;i++){
        int ans =INT_MAX;
        for(int start =1;start<=sqrt(i);start++){
            int sq = start*start;
            int ans1 = 1+dp[i-sq];
             ans =min(ans1,ans);
            
        }
        dp[i]= ans;

         }
          return dp[n];
    }
    int numSquares(int n) {
       return f(n)-1;
    }
};


/////////////////////////////2.Min Cost for Tickets Leetcode//////////////////////////////




class Solution {
public:
    int f(vector<int>& days, vector<int>& costs,int i){
       if(i==days.size()) return 0;
       int cost1 = costs[0]+f(days,costs,i+1);
       int endPass7 = days[i]+7-1;
       int j =i;
       while(j<days.size() && days[j]<=endPass7) j++;
       int cost7 = costs[1]+f(days,costs,j);

        int endPass30 = days[i]+30-1;
        j =i;
       while(j<days.size() && days[j]<=endPass30) j++;
       int cost30 = costs[2]+f(days,costs,j);
      return min(cost1,min(cost7,cost30));
    }
    int mincostTickets(vector<int>& days, vector<int>& costs) {
        return f(days,costs,0);
    }
};


class Solution {
public:
    vector<int> dp;
    int f(vector<int>& days, vector<int>& costs,int i){
       if(i==days.size()) return 0;
       if(dp[i] !=-1) return dp[i];
       int cost1 = costs[0]+f(days,costs,i+1);
       int endPass7 = days[i]+7-1;
       int j =i;
       while(j<days.size() && days[j]<=endPass7) j++;
       int cost7 = costs[1]+f(days,costs,j);

        int endPass30 = days[i]+30-1;
        j =i;
       while(j<days.size() && days[j]<=endPass30) j++;
       int cost30 = costs[2]+f(days,costs,j);
       dp[i] = min(cost1,min(cost7,cost30));
       return dp[i];
    }
    int mincostTickets(vector<int>& days, vector<int>& costs) {
        dp.clear();
        dp.resize(366,-1);
        return f(days,costs,0);
    }
};


class Solution {
public:
    vector<int> dp;
    int f(vector<int>& days, vector<int>& costs){
       dp.resize(366,0);
       for(int i =days.size()-1;i>=0;i--){
       int cost1 = costs[0]+dp[i+1];
       int endPass7 = days[i]+7-1;
       int j =i;
       while(j<days.size() && days[j]<=endPass7) j++;
       int cost7 = costs[1]+dp[j];
       int endPass30 = days[i]+30-1;
        j =i;
       while(j<days.size() && days[j]<=endPass30) j++;
       int cost30 = costs[2]+dp[j];
       dp[i] = min(cost1,min(cost7,cost30));
       }
       return dp[0];
         
    }
    int mincostTickets(vector<int>& days, vector<int>& costs) {
        return f(days,costs);
    }
};

//***********************************DP on Strings**************************************8
///////////////////////////// 3.Longest Palindromic SubString Leetcode/////////////////////
                       // polimdrome using recursion
    ////////////////////////////////////////////////////////////////////////////////////

    bool isPalindrome(string &s, int i, int j) {
            if(i>=j) return true;
            if (s[i] != s[j]) return false;
            return isPalindrome(s,i+1,j-1);
      }


class Solution {
public:
  vector<vector<int>>dp;
  bool isPalindrome(string &s, int i, int j) {
            if(i>=j) return true;
            if(dp[i][j] !=-1) return dp[i][j];
            if (s[i] != s[j]) return false;
            return dp[i][j] = isPalindrome(s,i+1,j-1);
      }

   string longestPalindrome(string s) {
         
        int maxLen = 0;
        string ans="";
        dp.clear();
        dp.resize(1005,vector<int>(1005,-1));
        for (int i = 0; i < s.size(); i++) {
             for (int j = i; j < s.size(); j++) {
                  if (isPalindrome(s, i, j)) {
                         if (j - i + 1 > maxLen) {
                            maxLen = j - i + 1;
                            ans = s.substr(i, maxLen);  // substr(start,length);
                         }
                  }
             }
        }
        return ans;
    }
};

// 4.Distinct Subsequences Leetcode

class Solution {
public:
    int f(string & s,string &t,int i, int j){
         if(j==t.size()) return 1;
         if(i==s.size()) return 0;
         int ans1=0;
         if(s[i]==t[j]){
             ans1 += f(s,t,i+1,j+1);
             ans1 +=f(s,t,i+1,j);
         }else {
               ans1 += f(s,t,i+1,j);
         }
         return ans1;
    }
    int numDistinct(string s, string t) {
        return f(s,t,0,0);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(string & s,string &t,int i, int j){
         if(j==t.size()) return 1;
         if(i==s.size()) return 0;
         if(dp[i][j]!=-1) return dp[i][j];
         int ans1=0;
         if(s[i]==t[j]){
             ans1 += f(s,t,i+1,j+1);
             ans1 +=f(s,t,i+1,j);
         }else {
               ans1 +=f(s,t,i+1,j);
         }
          dp[i][j] = ans1;
          return dp[i][j];
    }
    int numDistinct(string s, string t) {
        dp.clear();
        dp.resize(1005,vector<int>(1005,-1));
        return f(s,t,0,0);
    }
};

class Solution {
public:
    vector<vector<int>> dp;

    int f(string &s, string &t) {
        dp.resize(s.size()+1, vector<int>(t.size()+1, 0));

        for (int i = 0; i <= s.size(); i++) {
            dp[i][t.size()] = 1;
        }

        for (int i = s.size() - 1; i >= 0; i--) {
            for (int j = t.size() - 1; j >= 0; j--) {
                long long  ans1 = 0;
                if (s[i] == t[j]) {
                    ans1 += dp[i + 1][j + 1];
                    ans1 += dp[i + 1][j];
                } else {
                    ans1 += dp[i + 1][j];
                }
                dp[i][j] = ans1;
            }
        }
        return dp[0][0];
    }

    int numDistinct(string s, string t) {
        return f(s, t);
    }
};


// 5.Min ASCII Delete Sum for 2 Strings Leetcode


class Solution {
public: 
    int f(string &s1,string &s2,int i, int j ){
       int cost =0;
       if(i==s1.size() || j== s2.size()){
          for(int x =i;x<s1.size();x++) cost +=s1[x];
          for(int x =j;x<s2.size();x++) cost +=s2[x];
        
       }
       else if(s1[i]==s2[j]) {
           cost = f(s1,s2,i+1,j+1);
       }
       else {
          int cost1 = s1[i]+f(s1,s2,i+1,j);
          int cost2 = s2[j]+f(s1,s2,i,j+1);
          cost = min(cost1,cost2);
       }
        return cost;
    }
    int minimumDeleteSum(string s1, string s2) {
        return f(s1,s2,0,0);
    }
};

class Solution {
public: 
    vector<vector<int>>dp;
    int f(string &s1,string &s2,int i, int j ){
       int cost =0;
        if(dp[i][j] !=-1) return dp[i][j];
       if(i==s1.size() || j== s2.size()){
          for(int x =i;x<s1.size();x++) cost +=s1[x];
          for(int x =j;x<s2.size();x++) cost +=s2[x];
        
       }
       else if(s1[i]==s2[j]) {
           cost = f(s1,s2,i+1,j+1);
       }
       else {
          int cost1 = s1[i]+f(s1,s2,i+1,j);
          int cost2 = s2[j]+f(s1,s2,i,j+1);
          cost = min(cost1,cost2);
       }
        return dp[i][j]=cost;
    }
    int minimumDeleteSum(string s1, string s2) {
        dp.resize(1000,vector<int>(1000,-1));
        return f(s1,s2,0,0);
    }
};

/////////////////////////////// Word Break 1////////////////////////////////////

class Solution {
public:
    bool check (string &s, vector<string>& wordDict){
       for(auto x:wordDict) if(x==s) return true;
       return false;
    }
    bool f(string &s, vector<string>& wordDict,int start){
       if(start==s.size()) return true;
       bool flag = false;
       string word = "";
       for(int i = start;i<s.size();i++){
          word +=s[i];
          if(check(word,wordDict)){
           flag = flag || f(s,wordDict,i+1);
          }
       }
       return flag;
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        return f(s,wordDict,0);
    }
};


class Solution {
public:
    vector<int>dp;
    bool check (string &s, vector<string>& wordDict){
       for(auto x:wordDict) if(x==s) return true;
       return false;
    }

    bool f(string &s, vector<string>& wordDict,int start){
       if(start==s.size()) return true;
       if(dp[start] !=-1) return dp[start];
       bool flag = false;
       string word = "";
       for(int i = start;i<s.size();i++){
          word +=s[i];
          if(check(word,wordDict)){
           flag = flag || f(s,wordDict,i+1);
          }
       }
       dp[start] =flag;
       return dp[start];
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        dp.resize(301,-1);
        return f(s,wordDict,0);
    }
};




class Solution {
public:
    vector<int>dp;
    bool check (string &s, vector<string>& wordDict){
       for(auto x:wordDict) if(x==s) return true;
       return false;
    }

    bool f(string &s, vector<string>& wordDict){
       dp.resize(301,true);
      for(int start =s.size()-1;start>=0;start--){
       bool flag = false;
       string word = "";
       for(int i = start;i<s.size();i++){
          word +=s[i];
          if(check(word,wordDict)){
           flag = flag || dp[i+1];
          }
       }
       dp[start] =flag;
      }
        return dp[0];
     
    }
    bool wordBreak(string s, vector<string>& wordDict) {
        return f(s,wordDict);
    }
};

////////////////// Word Break  2 Leetcode//////////////////////////////

class Solution {
public:
    vector<string> f(string &s, unordered_map<string,bool> &mp,int i ){
      if(i==s.size()) return {""};
       string word;
       vector<string>ans;
       for(int j =i;j<s.size();j++){
          word.push_back(s[j]);
          if(mp.find(word)==mp.end()) continue;
          auto right = f(s,mp,j+1);
          for(auto rightpart:right){
            string endpart;
            if(rightpart.size()>0) endpart = " "+rightpart;
            ans.push_back(word+endpart);
          }
       }
       return ans;
    }
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_map<string,bool> mp;
        for(auto word:wordDict) mp[word]=1;
        return f(s,mp,0);
    }
};


class Solution {
public:
    unordered_map<int , vector<string>>dp;
    vector<string> f(string &s, unordered_map<string,bool> &mp,int i ){
      if(i==s.size()) return {""};
      if(dp.find(i) !=dp.end()) return dp[i];
       string word;
       vector<string>ans;
       for(int j =i;j<s.size();j++){
          word.push_back(s[j]);
          if(mp.find(word)==mp.end()) continue;
          auto right = f(s,mp,j+1);
          for(auto rightpart:right){
            string endpart;
            if(rightpart.size()>0) endpart = " "+rightpart;
            ans.push_back(word+endpart);
          }
       }
      return dp[i] = ans;
    }
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_map<string,bool> mp;
        for(auto word:wordDict) mp[word]=1;
        return f(s,mp,0);
    }
};


//******************************************DP on Trees*****************************
// ////////////////////7.House Robber iii Leetcode//////////////////////////


class Solution {
public:
    int f(TreeNode* root){
        if(!root) return 0;
        int rob =0,notrob=0;

        rob +=root->val; // rob
        if(root->left) rob +=f(root->left->left)+f(root->left->right);
        if(root->right) rob +=f(root->right->left)+f(root->right->right);
        notrob = f(root->left)+f(root->right);
        return max(rob,notrob);
    }
    int rob(TreeNode* root) {
        return f(root,dp);
    }
};


class Solution {
public:
    int f(TreeNode* root,unordered_map<TreeNode*,int>&dp){
        if(!root) return 0;
        int rob =0,notrob=0;
        if(dp.find(root) !=dp.end()) return dp[root];
        rob +=root->val; // rob
        if(root->left) rob +=f(root->left->left,dp)+f(root->left->right,dp);
        if(root->right) rob +=f(root->right->left,dp)+f(root->right->right,dp);
        notrob = f(root->left,dp)+f(root->right,dp);
        dp[root] = max(rob,notrob);
        return dp[root];
    }
    int rob(TreeNode* root) {
        unordered_map<TreeNode*,int>dp;
        return f(root,dp);
    }
};


//////////////////////////// 8.Unique BST ii Leetcode////////////////////////////


class Solution {
public:
     vector<TreeNode*> f(int start,int end){
        // base case 
        if(start>end) return {0};
        if(start==end) return {new TreeNode(start)};
        vector<TreeNode*>ans;
        for(int i =start;i<=end;i++){
        vector<TreeNode*> left = f(start,i-1);
        vector<TreeNode*> right = f(i+1,end);
        for(int j =0;j<left.size();j++){
           for(int k =0;k<right.size();k++){
              TreeNode* root = new TreeNode(i);
              root->left= left[j];
              root->right= right[k];
              ans.push_back(root);
           }
        }
     }
     return ans;
 }
    vector<TreeNode*> generateTrees(int n) {
       if(n==0) return {0};
        return f(1,n);
    }
};



class Solution {
public:
     map<pair<int,int>,vector<TreeNode*>>dp;
     vector<TreeNode*> f(int start,int end){
        // base case 
        if(start>end) return {0};
        if(start==end) return {new TreeNode(start)};
        if(dp.find({start,end}) !=dp.end()) return dp[{start,end}];
        vector<TreeNode*>ans;
        for(int i =start;i<=end;i++){
        vector<TreeNode*> left = f(start,i-1);
        vector<TreeNode*> right = f(i+1,end);
        for(int j =0;j<left.size();j++){
           for(int k =0;k<right.size();k++){
              TreeNode* root = new TreeNode(i);
              root->left= left[j];
              root->right= right[k];
              ans.push_back(root);
           }
        }
     }
     dp[{start,end}]=ans;
     return dp[{start,end}];
 }
    vector<TreeNode*> generateTrees(int n) {
       if(n==0) return {0};
        return f(1,n);
    }
};


//***********************************DP on Intervals************************************
////////////////////////////////// .Stone Games Leetcode I ////////////////////////////////

class Solution {
public:
    bool stoneGame(vector<int>& piles) {
        // there will beat least one way by which alice will win.
        // that means , we should return true;
        // because the problem is asking is there any way that alice can win.
        return true;
    }
};


//////////////////////////////////// Stone Games II////////////////////////////

class Solution {
public:
    int f(vector<int>&piles,int i,int M,bool alice){
       if(i==piles.size()) return 0;
       int total =0;
       int ans =alice?INT_MIN:INT_MAX;
       for(int X =1;X<=2*M;X++){
           if(i+X-1>=piles.size()) break;
           total +=piles[i+X-1];
           if(alice) ans = max(ans,total+f(piles,i+X,max(X,M),!alice)); 
           else  ans = min(ans,f(piles,i+X,max(X,M),!alice)); 
       }
       return ans;
    }
    int stoneGameII(vector<int>& piles) {
        return f(piles,0,1,true); //{piles,indx,M==1}
    }
};


class Solution {
public:
    vector<vector<vector<int>>>dp;
    int f(vector<int>&piles,int i,int M,bool alice){
       if(i==piles.size()) return 0;
       if( dp[i][M][alice] !=-1) return dp[i][M][alice];
       int total =0;
       int ans =alice?INT_MIN:INT_MAX;
       for(int X =1;X<=2*M;X++){
           if(i+X-1>=piles.size()) break;
           total +=piles[i+X-1];
           if(alice) ans = max(ans,total+f(piles,i+X,max(X,M),!alice)); 
           else  ans = min(ans,f(piles,i+X,max(X,M),!alice)); 
       }
       dp[i][M][alice]= ans;
       return dp[i][M][alice];
    }
    int stoneGameII(vector<int>& piles) {
        dp.resize(101,vector<vector<int>>(101,vector<int>(2,-1)));
        return f(piles,0,1,true); //{piles,indx,M==1}
    }
};



class Solution {
public:
    vector<vector<vector<int>>>dp;
    int f(vector<int>&piles){
       dp.resize(101,vector<vector<int>>(101,vector<int>(2,0)));
       for(int i = piles.size()-1;i>=0;i--){
          for(int M = piles.size()-1;M>=1;M--){
             for(int alice = 0;alice<=1;alice++){
                 int total =0;
                int ans =alice?INT_MIN:INT_MAX;
                for(int X =1;X<=2*M;X++){
                    if(i+X-1>=piles.size()) break;
                    total +=piles[i+X-1];
                    if(alice) ans = max(ans,total+dp[i+X][max(X,M)][!alice]); 
                    else  ans = min(ans,0+dp[i+X][max(X,M)][!alice]); 
                }
                dp[i][M][alice]= ans;
             }
          }
       }
      return dp[0][1][true];
    }
    int stoneGameII(vector<int>& piles) {
        if(piles.size()==1) return piles[0];
        return f(piles); //{piles,indx,M==1}
    }
};
/////////////////////////// 10.Burst balloons Leetcode//////////////////////////////////////

class Solution {
public:
    int f(vector<int>&nums,int start,int end){
       if(start>end) return 0;
       int cost = INT_MIN;
       for(int i = start;i<=end;i++){
        cost = max(nums[start-1]*nums[i]*nums[end+1]+f(nums,start,i-1)+f(nums,i+1,end),cost);

       }
       return cost;
    }
    int maxCoins(vector<int>& nums) {
        nums.insert(nums.begin(),1);
        nums.insert(nums.end(),1);
        return f(nums,1,nums.size()-2);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&nums,int start,int end){
       if(start>end) return 0;
       if( dp[start][end] !=-1) return dp[start][end];
       int cost = INT_MIN;
       for(int i = start;i<=end;i++){
        cost = max(nums[start-1]*nums[i]*nums[end+1]+f(nums,start,i-1)+f(nums,i+1,end),cost);

       }
       dp[start][end] = cost;
       return dp[start][end];
    }
    int maxCoins(vector<int>& nums) {
        nums.insert(nums.begin(),1);
        nums.insert(nums.end(),1);
        dp.clear();
        dp.resize(301,vector<int>(301,-1));
        return f(nums,1,nums.size()-2);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&nums){
         dp.resize(302,vector<int>(302,0));
         for(int start = nums.size()-2;start>=1;start--){
             for(int end = start;end<=nums.size()-2;end++){
                  int cost = INT_MIN;
                  for(int i = start;i<=end;i++){
                 cost = max(nums[start-1]*nums[i]*nums[end+1]+dp[start][i-1]+dp[i+1][end],cost);

                  }
                  dp[start][end] = cost;
             }
         }
          return dp[1][nums.size()-2];
    }
    int maxCoins(vector<int>& nums) {
        nums.insert(nums.begin(),1);
        nums.insert(nums.end(),1);
        return f(nums);
    }
};

//*********************************LIS / LCS Variants***************************************8

// ///////////////////////11.Intervleaving Strings Leetcode///////////////////////////////

class Solution {
public:
    bool f(string s1,string s2, string s3,int i, int j, int k){
         if(i==s1.size() && j==s2.size() && k==s3.size()) return true;
          bool flag1 =false;
          bool flag2 =false;
         if( i<s1.size() && s1[i]==s3[k]){
               flag1 = f(s1,s2,s3,i+1,j,k+1);
         }
          if( j<s2.size() && s2[j]==s3[k]){
              flag2 = f(s1,s2,s3,i,j+1,k+1);
         }

         return (flag1 || flag2);
    }
    bool isInterleave(string s1, string s2, string s3) {
         if(s1.size()==0 && s2.size()==0&& s3.size()==0) return true;
         return f(s1,s2,s3,0,0,0);
    }
};


class Solution {
public:
    vector<vector<vector<int>>>dp;
    bool f(string &s1,string &s2, string &s3,int i, int j, int k){
         if(i==s1.size() && j==s2.size() && k==s3.size()) return true;
         if(dp[i][j][k] !=-1) return dp[i][j][k];
          bool flag1 =false;
          bool flag2 =false;
         if( i<s1.size() && s1[i]==s3[k]){
               flag1 = f(s1,s2,s3,i+1,j,k+1);
         }
          if( j<s2.size() && s2[j]==s3[k]){
              flag2 = f(s1,s2,s3,i,j+1,k+1);
         }

        dp[i][j][k] = (flag1 || flag2);
        return dp[i][j][k];
    }
    bool isInterleave(string s1, string s2, string s3) {
         if(s1.size()==0 && s2.size()==0&& s3.size()==0) return true;
          dp.resize(s1.size()+1,vector<vector<int>>(s2.size()+1,vector<int>(s3.size()+1,-1)));
         return f(s1,s2,s3,0,0,0);
    }
};



class Solution {
public:
    vector<vector<vector<int>>> dp;
    
    bool f(string& s1, string& s2, string& s3) {
        dp.resize(s1.size() + 1, vector<vector<int>>(s2.size() + 1, vector<int>(s3.size()+ 1, false)));
        
        // Base case: Empty strings match.
        dp[s1.size()][s2.size()][s3.size()] = true;
        
        for (int i = s1.size(); i >= 0; i--) {
            for (int j = s2.size(); j >= 0; j--) {
                for (int k = s3.size(); k >= 0; k--) {
                    if (i < s1.size() && s1[i] == s3[k]) {
                        dp[i][j][k] = dp[i][j][k] || dp[i + 1][j][k + 1];
                    }
                    if (j < s2.size() && s2[j] == s3[k]) {
                        dp[i][j][k] = dp[i][j][k] || dp[i][j + 1][k + 1];
                    }
                }
            }
        }
        
        return dp[0][0][0];
    }

    bool isInterleave(string s1, string s2, string s3) {
        if (s1.size() ==0 && s2.size() == 0 && s3.size()==0) return true;
        return f(s1, s2, s3);
    }
};

////////////////////////// 12.Min Insertion steps to make a string palindrome Leetcode//////////////////
/// minInsertions=s.size()-LPS(length);

class Solution {
public:
        int f(string &s1,string &s2,int i ,int j){
        if(i==s1.size() || j==s2.size()) return 0;
        int ans =0;
        if(s1[i]==s2[j]) ans = 1+f(s1,s2,i+1,j+1);
        else ans =0+max(f(s1,s2,i+1,j),f(s1,s2,i,j+1));
        return dp[i][j] = ans;
   }
    int minInsertions(string s) {
        string s1 =s;
        reverse(s1.begin(),s1.end());
         int LPS = f(s,s1,0,0);
         return s.size()-LPS;

    }
};

class Solution {
public:
        vector<vector<int>>dp;
        int f(string &s1,string &s2,int i ,int j){
        if(i==s1.size() || j==s2.size()) return 0;
        if(dp[i][j] !=-1) return dp[i][j];
        int ans =0;
        if(s1[i]==s2[j]) ans = 1+f(s1,s2,i+1,j+1);
        else ans =0+max(f(s1,s2,i+1,j),f(s1,s2,i,j+1));
        return dp[i][j] = ans;
   }
    int minInsertions(string s) {
        string s1 =s;
        reverse(s1.begin(),s1.end());
        dp.clear();
        dp.resize(505,vector<int>(505,-1));
         int LPS = f(s,s1,0,0);
         return s.size()-LPS;

    }
};

class Solution {
public:
        vector<vector<int>>dp;
        int f(string &s1,string &s2){
        dp.resize(505,vector<int>(505,0));
        for(int i =s1.size()-1;i>=0;i--){
          for(int j =s2.size()-1;j>=0;j--){
                    int ans =0;
                    if(s1[i]==s2[j]) ans = 1+dp[i+1][j+1];
                    else ans =0+max(dp[i+1][j],dp[i][j+1]);
                    dp[i][j] = ans;
          }
     }

      return dp[0][0];
   }
    int minInsertions(string s) {
        string s1 =s;
        reverse(s1.begin(),s1.end());
         int LPS = f(s,s1);
         return s.size()-LPS;

    }
};

////////////// 14.Min Number of Removals to make Mountain Array Leetcode//////////////////////



class Solution {
public:
    int f(vector<int>& nums, vector<int>& LIS) {
        int n = nums.size();
        vector<int> ans;
        ans.push_back(nums[0]);
        for (int i = 1; i < n; i++) {
            if (nums[i] > ans.back()) ans.push_back(nums[i]);
                 else {
                int indx = lower_bound(ans.begin(), ans.end(), nums[i]) - ans.begin();
                ans[indx] = nums[i];
            }
            LIS[i] = ans.size();  // Store the length of the increasing subsequence at index i
        }
        return ans.size();
    }

    int minimumMountainRemovals(vector<int>& nums) {
        int n = nums.size();
        vector<int> LIS(n,1), LDS(n,1);
        int increasingLength = f(nums, LIS);
        reverse(nums.begin(), nums.end());
        int decreasingLength = f(nums, LDS);
        int maxMountain = INT_MIN;
        for (int i = 1; i < n - 1; i++) {
            if (LIS[i] > 1 && LDS[n - i - 1] > 1) {
                // Calculate the total length of the mountain using LIS and LDS
                maxMountain = max(maxMountain, LIS[i] + LDS[n - i - 1] - 1);
            }
        }
        // Calculate the minimum number of elements to remove
        return n - maxMountain;
    }
};


////////////////////// 15.Make Array Strictly increasing Leetcode/////////////////////////////


#define INF (1e9+1)
class Solution {
public:
    int f(vector<int>& arr1, vector<int>& arr2,int prev,int i){
         if(i==arr1.size()) return 0;
         int op1 = INF;
         if(prev<arr1[i]) op1= 0+f(arr1,arr2,arr1[i],i+1); // no ops
          int op2 = INF;
          auto it = upper_bound(arr2.begin(),arr2.end(),prev);
          if(it !=arr2.end()) { // it is present 
            int indx = it-arr2.begin();
            op2 =1+f(arr1,arr2,arr2[indx],i+1);

          }
           int ans = min(op1,op2);
          return  ans;

    }
    int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2) {
        sort(arr2.begin(),arr2.end());
        int ans = f(arr1,arr2,-1,0);
        return ans == INF?-1:ans;
    
    }
};

#define INF (1e9+1)
class Solution {
public:
     map<pair<int,int>,int>dp; 
     // use map for dp becasue 1 <= arr1.length, arr2.length <= 20000<= arr1[i], arr2[i] <= 10^9
    int f(vector<int>& arr1, vector<int>& arr2,int prev,int i){
         if(i==arr1.size()) return 0;
         if(dp.find({prev,i}) !=dp.end()) return dp[{prev,i}];
         int op1 = INF;
         if(prev<arr1[i]) op1= 0+f(arr1,arr2,arr1[i],i+1); // no ops
          int op2 = INF;
          auto it = upper_bound(arr2.begin(),arr2.end(),prev);
          if(it !=arr2.end()) { // it is present 
            int indx = it-arr2.begin();
            op2 =1+f(arr1,arr2,arr2[indx],i+1);

          }
          dp[{prev,i}] = min(op1,op2);
          return  dp[{prev,i}];

    }
    int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2) {
        sort(arr2.begin(),arr2.end());
        int ans = f(arr1,arr2,-1,0);
        return ans == INF?-1:ans;
    
    }
};

//************************Buy & Sell Stocks Variants [all 5 variants] Leetcode************************
/////////////////////////////////////////// Buy & Sell Stocks 1/////////////////////////////
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        //[7,1,5,3,6,4]
         int n = prices.size();
         int mini = prices[0];
         int maxprofit=INT_MIN; //0
         for(int each:prices){
              mini = min(mini,each);
              int stock=each-mini;
              maxprofit=max(maxprofit,stock);
         }
         if(maxprofit<0) return 0;
         else return maxprofit;
    }
};


class Solution {
public:
   void solve(vector<int>& prices,int &maxProfit,int &minPrice,int indx){
         if(indx==prices.size()) return;  // base case
         if(minPrice>prices[indx]) minPrice = prices[indx];  // one case
         int maxP = prices[indx]-minPrice;
         if(maxP>maxProfit) maxProfit = maxP;
         solve(prices, maxProfit,minPrice,indx+1); // recursive case
    }
    int maxProfit(vector<int>& prices) {
        int maxProfit =INT_MIN;
        int minPrice = INT_MAX;
        solve(prices,maxProfit , minPrice,0);
        return maxProfit;
    }
};


/////////////////////////////// 


//17.1st Variant [121. Best Time to Buy and Sell Stock] - Already Covered in Recursion Week HomeWork.


/// //////////////////// buy and sell stock (dp)//////////////////////// 


class Solution {
public:
    int maxProfit(vector<int>& prices) {
        //[7,1,5,3,6,4]
         int n = prices.size();
         int mini = prices[0];
        int maxprofit=INT_MIN; //0
         for(int each:prices){
              mini = min(mini,each);
              int stock=each-mini;
              maxprofit=max(maxprofit,stock);
         }
         if(maxprofit<0) return 0;
         else return maxprofit;
    }
};


class Solution {
public:
   void solve(vector<int>& prices,int &maxProfit,int &minPrice,int indx){
         if(indx==prices.size()) return;  // base case
         if(minPrice>prices[indx]) minPrice = prices[indx];  // one case
         int maxP = prices[indx]-minPrice;
         if(maxP>maxProfit) maxProfit = maxP;
         solve(prices, maxProfit,minPrice,indx+1); // recursive case
    }
    int maxProfit(vector<int>& prices) {
        int maxProfit =INT_MIN;
        int minPrice = INT_MAX;
        solve(prices,maxProfit , minPrice,0);
        return maxProfit;
    }
};

//############################### buy and sell stock II#############################3 

class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&prices,int i, bool buy){
        if(i>=prices.size()) return 0;
        if(dp[i][buy] !=-1) return dp[i][buy]; 
        int profit =0;
        if(buy){
          int buyItProfit = -prices[i]+f(prices,i+1,0);
          int skipProfit =f(prices,i+1,1);
          profit =max(buyItProfit, skipProfit);
           
        }else {
           int sellProfit = prices[i]+f(prices,i+1,1);
           int skipProfit = f(prices,i+1,0);
           profit = max(sellProfit, skipProfit);
        }
       return dp[i][buy] = profit;
    }
    int maxProfit(vector<int>& prices) {
        dp.resize(30001,vector<int>(2,-1));
        return f(prices,0,true);
    }
};

//////////////////////////////////////////////////////////////////////////////////

class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&prices){
         dp.resize(30001,vector<int>(2,0));
         for(int i =prices.size()-1;i>=0;i--){
            for(int j = 0;j<=1;j++){
                int profit =0;
                if(j){
                  int buyItProfit = -prices[i]+dp[i+1][0];
                  int skipProfit =dp[i+1][1];
                  profit =max(buyItProfit, skipProfit);
                  
                }else {
                  int sellProfit = prices[i]+dp[i+1][1];
                  int skipProfit = dp[i+1][0];
                  profit = max(sellProfit, skipProfit);
                }
                  dp[i][j] = profit;
            }
         }
          return dp[0][1];
    }
    int maxProfit(vector<int>& prices) {
        return f(prices);
    }
};

/////////////////////// exact same as buy and sell stock1 problem//////////////////////
//////////////// 2016. Maximum Difference Between Increasing Elements///////////////////

class Solution {
public:
    void solve(vector<int>& prices,int &maxProfit,int &minPrice,int indx){
         if(indx==prices.size()) return;  // base case
         if(minPrice>prices[indx]) minPrice = prices[indx];  // one case
         int maxP = prices[indx]-minPrice;
         if(maxP>maxProfit) maxProfit = maxP;
         solve(prices, maxProfit,minPrice,indx+1); // recursive case
    }

    int maximumDifference(vector<int>& nums) {
        int maxProfit =INT_MIN;
        int minPrice = INT_MAX;
        solve(nums,maxProfit , minPrice,0);
        if(maxProfit>0) return maxProfit ;
        else return -1;
    }
};


////////////////////////////// buy and sell stocks III///////////////////////////

class Solution {
public:
    int f(vector<int>&prices,int i,bool buy , int limit){
       if(limit==0 || i==prices.size()) return 0;
       // if buy is true;
       int profit =0;
       if(buy){
           int buykaro = -prices[i]+f(prices,i+1,false, limit);
           int dontbuy = 0+f(prices,i+1,true,limit);
           profit =max(buykaro,dontbuy);
       }
       // if buy is false;
        else{
           int sellkaro = prices[i]+f(prices,i+1,true,limit-1);
           int dontsell = 0+f(prices,i+1,false,limit);
           profit =max(sellkaro,dontsell);
       }
       return profit;
    }
    int maxProfit(vector<int>& prices) {
        return f(prices,0,true,2);
    }
};

//////////////////////////////////////////////////////////////////////////////////
class Solution {
public:
    vector<vector<vector<int>>>dp;
    int f(vector<int>&prices,int i,bool buy , int limit){
       if(limit==0 || i==prices.size()) return 0;
       if(dp[i][buy][limit] !=-1)  return dp[i][buy][limit];
       // if buy is true;
       int profit =0;
       if(buy){
           int buykaro = -prices[i]+f(prices,i+1,false, limit);
           int dontbuy = 0+f(prices,i+1,true,limit);
           profit =max(buykaro,dontbuy);
       }
       // if buy is false;
        else{
           int sellkaro = prices[i]+f(prices,i+1,true,limit-1);
           int dontsell = 0+f(prices,i+1,false,limit);
           profit =max(sellkaro,dontsell);
       }
       dp[i][buy][limit] = profit;
       return dp[i][buy][limit];
    }
    int maxProfit(vector<int>& prices) {
        dp.resize(prices.size()+1,vector<vector<int>>(3,vector<int>(3,-1)));
        return f(prices,0,true,2);
    }
};

/////////////////////////////////////////////////////////////////////////////

class Solution {
public:
    vector<vector<vector<int>>>dp;
    int f(vector<int>&prices){
       dp.resize(prices.size()+1,vector<vector<int>>(3,vector<int>(3,0)));
       // if buy is true;
       for(int i =prices.size()-1;i>=0;i--){
          for(int buy = 0;buy<=1;buy++){
             for(int limit=1;limit<=2;limit++){
                   int profit =0;
                  if(buy){
                      int buykaro = -prices[i]+dp[i+1][false][limit];
                      int dontbuy = 0+dp[i+1][true][limit];
                      profit =max(buykaro,dontbuy);
                  }
                  // if buy is false;
                    else{
                      int sellkaro = prices[i]+dp[i+1][true][limit-1];
                      int dontsell = 0+dp[i+1][false][limit];
                      profit =max(sellkaro,dontsell);
                  }
                  dp[i][buy][limit] = profit;
             }
          }
       }
       return dp[0][1][2];
     
    }
    int maxProfit(vector<int>& prices) {
        return f(prices);
    }
};

///////////////////////////////// space optimation///////////////////////////////


//////////////////////////// buy and sell stocks IV//////////////////////////////////

class Solution {
public:
    int f(vector<int>&prices,int limit,int i,bool buy){
      if(i==prices.size() || limit ==0) return 0;
      int profit =0;
      // if buy true do buy or skip
      if(buy){
         int buykaro = -prices[i]+f(prices,limit,i+1,false);
         int dontbuy = 0+f(prices,limit,i+1,true);
         profit  = max(buykaro,dontbuy);
      }
         // if buy false do sell or skip
      else {
         int sellkaro = prices[i]+f(prices,limit-1,i+1,true);
         int dontsell = 0+f(prices,limit,i+1,false);
         profit  = max(sellkaro,dontsell);
      }
    return profit;
    }
    int maxProfit(int k, vector<int>& prices) {
        return f(prices,k,0,true);
    }
};


class Solution {
public:
    vector<vector<vector<int>>>dp;
    int f(vector<int>&prices,int limit,int i,bool buy){
      if(i==prices.size() || limit ==0) return 0;
      if( dp[i][limit][buy] !=-1) return dp[i][limit][buy];
      int profit =0;
      // if buy true do buy or skip
      if(buy){
         int buykaro = -prices[i]+f(prices,limit,i+1,false);
         int dontbuy = 0+f(prices,limit,i+1,true);
         profit  = max(buykaro,dontbuy);
      }
         // if buy false do sell or skip
      else {
         int sellkaro = prices[i]+f(prices,limit-1,i+1,true);
         int dontsell = 0+f(prices,limit,i+1,false);
         profit  = max(sellkaro,dontsell);
      }
        dp[i][limit][buy] = profit;
       return dp[i][limit][buy];
    }
    int maxProfit(int k, vector<int>& prices) {
        dp.resize(prices.size()+1,vector<vector<int>>(k+1,vector<int>(3,-1)));
        return f(prices,k,0,true);
    }
};


///////////////714. Best Time to Buy and Sell Stock with Transaction Fee V ///////////////////////////// 

class Solution {
public:
    int f(vector<int>&prices,int i, bool buy,int &fee){
       if(i==prices.size()) return 0;
       // if buy is true then we can buy the stocks;
       int profit =0;
       if(buy){
          int buykaro = -prices[i]+f(prices,i+1,false,fee);    
          int dontbuy = 0+f(prices,i+1,true,fee);  
          profit = max(buykaro,dontbuy);  
       }
         // buy is false then we can only sell stocks;
       else {
           int sellkaro = prices[i]+f(prices,i+1,true,fee)-fee;     
           int dontsell = 0+f(prices,i+1,false,fee);  
           profit = max(sellkaro,dontsell);  
       }
      return profit;

    }
    int maxProfit(vector<int>& prices, int fee) {
        return f(prices,0,true,fee);
    }
 };


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&prices,int i, bool buy,int &fee){
       if(i==prices.size()) return 0;
       // if buy is true then we can buy the stocks;
       if(dp[i][buy] !=-1) return dp[i][buy];
       int profit =0;
       if(buy){
          int buykaro = -prices[i]+f(prices,i+1,false,fee);    
          int dontbuy = 0+f(prices,i+1,true,fee);  
          profit = max(buykaro,dontbuy);  
       }
         // buy is false then we can only sell stocks;
       else {
           int sellkaro = prices[i]+f(prices,i+1,true,fee)-fee;     
           int dontsell = 0+f(prices,i+1,false,fee);  
           profit = max(sellkaro,dontsell);  
       }
           dp[i][buy] = profit;
           return dp[i][buy];

    }
    int maxProfit(vector<int>& prices, int fee) {
        dp.resize(prices.size()+1,vector<int>(2,-1));
        return f(prices,0,true,fee);
    }
};



class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&prices,int &fee){
       dp.resize(prices.size()+1,vector<int>(2,0));
       for(int i =prices.size()-1;i>=0;i--){
          for(int buy =0;buy<=1;buy++){
              int profit =0;
              if(buy){
                  int buykaro = -prices[i]+dp[i+1][false];    
                  int dontbuy = 0+dp[i+1][true];  
                  profit = max(buykaro,dontbuy);  
              }
                // buy is false then we can only sell stocks;
              else {
                  int sellkaro = prices[i]+dp[i+1][true]-fee;     
                  int dontsell = 0+dp[i+1][false];  
                  profit = max(sellkaro,dontsell);  
              }
                  dp[i][buy] = profit;
          }
       }
       return dp[0][true];

    }
    int maxProfit(vector<int>& prices, int fee) {

        return f(prices,fee);
    }
};

///////////////309. Best Time to Buy and Sell Stock with Cooldown////////////////////////////


class Solution {
public:
    int f(vector<int>&prices,int i , bool buy){
       if(i>=prices.size()) return 0;
       int profit =0;
       if(buy){
          int buykaro = -prices[i]+f(prices,i+1,false);  
          int dontbuy = 0+f(prices,i+1,true);  
          profit = max(buykaro,dontbuy);
       }else {
        //  After you sell your stock, you cannot buy stock 
         // on the next day (i.e., cooldown one day).
           int sellkaro  = prices[i]+f(prices,i+2,true);
           int dontsell  =  0+f(prices,i+1,false);
           profit = max(sellkaro,dontsell);
       }
        return profit;
    }
    int maxProfit(vector<int>& prices) {
        return f(prices,0,true);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&prices,int i , bool buy){
       if(i>=prices.size()) return 0;
       if(dp[i][buy] !=-1) return dp[i][buy];
       int profit =0;
       if(buy){
          int buykaro = -prices[i]+f(prices,i+1,false);  
          int dontbuy = 0+f(prices,i+1,true);  
          profit = max(buykaro,dontbuy);
       }else {
        //  After you sell your stock, you cannot buy stock 
         // on the next day (i.e., cooldown one day).
           int sellkaro  = prices[i]+f(prices,i+2,true);
           int dontsell  =  0+f(prices,i+1,false);
           profit = max(sellkaro,dontsell);
       }
        dp[i][buy] = profit;
        return dp[i][buy];
    }
    int maxProfit(vector<int>& prices) {
        dp.resize(prices.size()+1,vector<int>(2,-1));
        return f(prices,0,true);
    }
};



class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&prices){
       dp.resize(5001,vector<int>(3,0));
       for(int i =prices.size()-1;i>=0;i--){
         for(int buy =0;buy<=1;buy++){
              int profit =0;
                if(buy){
                    int buykaro = -prices[i]+dp[i+1][false];  
                    int dontbuy = 0+dp[i+1][true]; 
                    profit = max(buykaro,dontbuy);
                }else {
                  //  After you sell your stock, you cannot buy stock 
                  // on the next day (i.e., cooldown one day).
                    int sellkaro  = prices[i]+dp[i+2][true];
                    int dontsell  =  0+dp[i+1][false];
                    profit = max(sellkaro,dontsell);
                }
                  dp[i][buy] = profit;
         }
       }
        return dp[0][true];
     
    }
    int maxProfit(vector<int>& prices) {
        return f(prices);
    }
};

//*******************************Knapsack DP********************************

// /////////////////18.Target Sum Leetcode//////////////////////////////////////

class Solution {
public:
    int f(vector<int>& nums, int target,int indx){
        //  if(indx==nums.size() && target ==0 ) return 1;
        //  if(indx==nums.size() && target !=0) return 0;
         if(indx==nums.size()) return target==0 ? 1 : 0;
         int plus = f(nums,target-nums[indx],indx+1);
         int minus = f(nums,target+nums[indx],indx+1);
         int ans = plus+minus;
         return ans;
    }
    int findTargetSumWays(vector<int>& nums, int target) {
        return f(nums,target,0);
    }
};

class Solution {
public:
    int f(vector<int>& nums, int target, int indx, map<pair<int,int>,int>&dp) {
        if (indx == nums.size()) return target == 0 ? 1 : 0;
        if(dp.find({indx,target}) !=dp.end()) return dp[{indx,target}];
        int plus = f(nums, target - nums[indx], indx + 1,dp);
        int minus = f(nums, target + nums[indx], indx + 1,dp);
        return  dp[{indx,target}] = plus+minus;
    }

    int findTargetSumWays(vector<int>& nums, int target) {
        map<pair<int,int>,int>dp;
        return f(nums, target, 0,dp);
    }
};



class Solution {
public:
    int f(vector<int>& nums, int target) {
        map<pair<int,int>,int>dp;
        dp[{nums.size(),0}] =1;
        int total =0;
        for(auto num:nums) total+=num;

        for(int i =nums.size()-1;i>=0;i--){
            for(int sum = -total;sum<=total;sum++ ){
                int plus = dp.find({i+1,sum-nums[i]}) !=dp.end() ? dp[{i+1,sum-nums[i]}]:0;
                int minus = dp.find({i+1,sum+nums[i]}) !=dp.end() ? dp[{i+1,sum+nums[i]}]:0;
                dp[{i,sum}] =plus+minus;

             }

        }
        return  dp[{0,target}];
    }

    int findTargetSumWays(vector<int>& nums, int target) {
        return f(nums, target);
    }
};


//////////////////////// 19.Min Swaps to make Sequences increasing Leetcode//////////////////////////

class Solution {
public:
    int f(vector<int>&n1,vector<int>&n2, int i, int prev1,int prev2){
         if(i==n1.size()) return 0;
          int swap =INT_MAX,noswap =INT_MAX;
          if(prev1<n2[i] && prev2<n1[i]){
            swap =1+f(n1,n2,i+1,n2[i],n1[i]);
          }
          if(prev1<n1[i] && prev2<n2[i]) {
              noswap = 0+f(n1,n2,i+1,n1[i],n2[i]);
          }
          return min(swap, noswap);
    }
    int minSwap(vector<int>& nums1, vector<int>& nums2) {
        return f(nums1,nums2,0,-1,-1);
    }
};



/////// HERE only two state are varying indx and swap and noswap

class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&n1,vector<int>&n2, int i, int prev1,int prev2,bool flag){ 
         if(i==n1.size()) return 0;
         if( dp[i][flag] !=-1) return  dp[i][flag];
          int swap =INT_MAX,noswap =INT_MAX;
          if(prev1<n2[i] && prev2<n1[i]){
            swap =1+f(n1,n2,i+1,n2[i],n1[i],1);
          }
          if(prev1<n1[i] && prev2<n2[i]) {
              noswap = 0+f(n1,n2,i+1,n1[i],n2[i],0);
          }
          dp[i][flag] = min(swap, noswap);
          return dp[i][flag];
    }
    int minSwap(vector<int>& nums1, vector<int>& nums2) {
        dp.clear();
        dp.resize(nums1.size()+1,vector<int>(2,-1));
        return f(nums1,nums2,0,-1,-1,0);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>&n1,vector<int>&n2){ 
        dp.resize(n1.size()+1,vector<int>(2,0));
        for(int i = n1.size()-1;i>=1;i--){
            for(int j=1;j>=0;j--){
                 int p1 =n1[i-1];
                 int p2 =n2[i-1];
                 if(j) swap(p1,p2);
                int swap =INT_MAX,noswap =INT_MAX;
                if(p1<n2[i] && p2<n1[i]){
                    swap =1+dp[i+1][1];
                }
                if(p1<n1[i] && p2<n2[i]) {
                    noswap = 0+dp[i+1][0];
                }
                dp[i][j] = min(swap, noswap);
            }
        }
         return dp[1][0];
         
    }
    int minSwap(vector<int>& nums1, vector<int>& nums2) {
        nums1.insert(nums1.begin(),-1);
        nums2.insert(nums2.begin(),-1);
        return f(nums1,nums2);
    }
};

////////////////////////////////// 20.Reducing Dishes Leetcode////////////////////////

class Solution {
public:
    int f(vector<int>& sat,int i, int time){
         if(i==sat.size()) return 0;
         int include = sat[i]*time+f(sat,i+1,time+1);
         int exclude = 0+f(sat,i+1,time);
         return max(include ,exclude);
    }
    int maxSatisfaction(vector<int>& satisfaction) {
        sort(satisfaction.begin(),satisfaction.end());
        return f(satisfaction,0,1);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>& sat,int i, int time){
         if(i==sat.size()) return 0;
         if(dp[i][time] !=-1) return dp[i][time];
         int include = sat[i]*time+f(sat,i+1,time+1);
         int exclude = 0+f(sat,i+1,time);
         dp[i][time] = max(include ,exclude);
         return dp[i][time];
    }
    int maxSatisfaction(vector<int>& satisfaction) {
        sort(satisfaction.begin(),satisfaction.end());
        dp.clear();
        dp.resize(501,vector<int>(satisfaction.size()+1,-1));
        return f(satisfaction,0,1);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int>& sat){
         dp.resize(505,vector<int>(sat.size()+5,0));
         for(int i =sat.size()-1;i>=0;i--){
              for(int j = sat.size();j>=1;j--){
                    int include = sat[i]*j+dp[i+1][j+1];
                    int exclude = 0+dp[i+1][j];
                    dp[i][j] = max(include ,exclude);
              }
         }

          return dp[0][1];
       
    }
    int maxSatisfaction(vector<int>& satisfaction) {
        sort(satisfaction.begin(),satisfaction.end());
        return f(satisfaction);
    }
};


///////////////// 21.Ones and Zeroes Leetcode////////////////////////

class Solution {
public:
    void f1(vector<string>& strs, vector<pair<int,int>> &p){
        
         for(int i =0;i<strs.size();i++){
              int zero=0,ones=0;
              for(auto ch:strs[i]){
                if(ch=='0') zero++;
                else ones++;
              }
              p.push_back({zero,ones});
         }
    }
    int f(vector<string>& strs,int i, int m, int n, vector<pair<int,int>> &p){
         if(i==strs.size()) return 0;
         int zeros = p[i].first;
        int ones = p[i].second;
        int include =0 , exclude=0;
         if(m-zeros >=0 && n-ones>=0){
              include =1+f(strs,i+1,m-zeros,n-ones,p);
         } 
         exclude = 0+f(strs,i+1,m,n,p);
         return max(include,exclude);
    }
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<pair<int,int>> p; //{no of 0s and no of 1'};
        f1(strs,p);
        return f(strs,0,m,n,p);
    }
};


class Solution {
public:
    vector<vector<vector<int>>>dp;
    void f1(vector<string>& strs, vector<pair<int,int>> &p){
        
         for(int i =0;i<strs.size();i++){
              int zero=0,ones=0;
              for(auto ch:strs[i]){
                if(ch=='0') zero++;
                else ones++;
              }
              p.push_back({zero,ones});
         }
    }
    int f(vector<string>& strs,int i, int m, int n, vector<pair<int,int>> &p){
         if(i==strs.size()) return 0;
         if(dp[i][m][n]!=-1) return dp[i][m][n];
         int zeros = p[i].first;
        int ones = p[i].second;
        int include =0 , exclude=0;
         if(m-zeros >=0 && n-ones>=0){
              include =1+f(strs,i+1,m-zeros,n-ones,p);
         } 
         exclude = 0+f(strs,i+1,m,n,p);
         dp[i][m][n] = max(include,exclude);
         return dp[i][m][n];
    }
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<pair<int,int>> p; //{no of 0s and no of 1'};
        f1(strs,p);
        dp.resize(601,vector<vector<int>>(m+1,vector<int>(n+1,-1)));
        return f(strs,0,m,n,p);
    }
};


class Solution {
public:
    vector<vector<vector<int>>>dp;
    void f1(vector<string>& strs, vector<pair<int,int>> &p){
        
         for(int i =0;i<strs.size();i++){
              int zero=0,ones=0;
              for(auto ch:strs[i]){
                if(ch=='0') zero++;
                else ones++;
              }
              p.push_back({zero,ones});
         }
    }

    int f(vector<string>& strs,int m, int n, vector<pair<int,int>> &p){
       dp.resize(601,vector<vector<int>>(m+1,vector<int>(n+1,0)));
       for(int i =strs.size()-1;i>=0;i-- ){
            for(int j = 0;j<=m;j++){
                 for(int k =0;k<=n;k++){
                         int zeros = p[i].first;
                         int ones = p[i].second;
                         int include =0 , exclude=0;
                         if(j-zeros >=0 && k-ones>=0){
                              include =1+dp[i+1][j-zeros][k-ones];
                         } 
                         exclude = 0+dp[i+1][j][k];
                         dp[i][j][k] = max(include,exclude);
                 }
            }
       }
           return dp[0][m][n];
    }
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<pair<int,int>> p; //{no of 0s and no of 1'};
        f1(strs,p);
        return f(strs,m,n,p);
    }
};






//*************************************MinMax DP***************************************

//############################ 22.Predict the Winner Leetcod#########################pragma endregion
class Solution {
public:
    int f(vector<int> &nums,int start,int end){
        if(start==end) return nums[start];
        int diffByStart = nums[start]-f(nums,start+1,end);
        int diffByEnd = nums[end]-f(nums,start,end-1);
         return max(diffByStart,diffByEnd);
    }
    bool predictTheWinner(vector<int>& nums) {
        // player1 will start we can choose from {nums[0],nums[n-1]} we can choose optimally
        return f(nums,0,nums.size()-1)>=0;
    }
};

class Solution {
public:
    vector<vector<int>>dp;
    int f(vector<int> &nums,int start,int end){
        if(start==end) return nums[start];
        if(dp[start][end]!=-1) return dp[start][end];
        int diffByStart = nums[start]-f(nums,start+1,end);
        int diffByEnd = nums[end]-f(nums,start,end-1);
         dp[start][end] = max(diffByStart,diffByEnd);
         return dp[start][end];
    }
    bool predictTheWinner(vector<int>& nums) {
        // player1 will start we can choose from {nums[0],nums[n-1]} we can choose optimally
        dp.resize(25,vector<int>(25,-1));
        return f(nums,0,nums.size()-1)>=0;
    }
};

////////////////////////////////// 44. Wildcard Matching/////////////////////////
class Solution {
public:
    bool f(string &s,string &p,int i, int j){
         // base case 
         if(i==s.size() && j==p.size()) return true; // outof bound
         if(i==s.size() && j<p.size()){ // fron recusive tree or logic
              while(j<p.size()){
                  if(p[j]!='*') return false;
                  j++;
              }
              return true;
         }
         if(s[i]==p[j] || p[j]=='?') return f(s,p,i+1,j+1);
         if(p[j]=='*'){
              bool case1 = f(s,p,i,j+1); // assume '*'==""
              bool case2 = f(s,p,i+1,j); // assume '*' == any charge which matches with s[i];
              return (case1||case2);
         }
         return  false;
    }
    bool isMatch(string s, string p) {
      return f(s,p,0,0);
    }
};


class Solution {
public:
    vector<vector<int>>dp;
    bool f(string &s,string &p,int i, int j){
         // base case 
         if(i==s.size() && j==p.size()) return true; // outof bound
         if(i==s.size() && j<p.size()){ // fron recusive tree or logic
              while(j<p.size()){
                  if(p[j]!='*') return false;
                  j++;
              }
              return true;
         }
         if(dp[i][j]!=-1) return dp[i][j];

         if(s[i]==p[j] || p[j]=='?')  {
             dp[i][j] = f(s,p,i+1,j+1);
         }
         else if (p[j]=='*'){
              bool case1 = f(s,p,i,j+1); // assume '*'==""
              bool case2 = f(s,p,i+1,j); // assume '*' == any charge which matches with s[i];
              dp[i][j] = (case1||case2);
         }
         else{
              dp[i][j]=false;
         }
         return dp[i][j];
    }
    bool isMatch(string s, string p) {
      dp.clear();
      dp.resize(s.size()+1,vector<int>(p.size()+1,-1));
      return f(s,p,0,0);
    }
};



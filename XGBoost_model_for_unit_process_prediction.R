library(bcv)
library(Matrix)
library(readxl)
library(pracma)
library(MLmetrics)
library("xgboost")
library("randomForest")

#load the data (supporting information_sheet 3)
data=read.csv("data.csv",header=FALSE)

######data mormalization
normalize_min=apply(data, 1, FUN=min)
normalize_max=apply(data, 1, FUN=max)
max_row=apply(cbind(abs(normalize_min),normalize_max),1,FUN=max)
data_normalize=data/max_row
normalize_min=apply(data_normalize, 2, FUN=min)
normalize_max=apply(data_normalize, 2, FUN=max)
max_col=apply(cbind(abs(normalize_min),normalize_max),1,FUN=max)
data_normalize=t(t(data_normalize)/max_col)

############# basic info
num_nonzero=colSums(data != 0)
num_nonzero_flow=rowSums(data != 0)

index_2_row=which(num_nonzero_flow>1,arr.ind=TRUE) ###flows appear in more than one process

index_5=which(num_nonzero>4&num_nonzero<1732,arr.ind=TRUE)  ###processes with more than 5 flows
index_10=which(num_nonzero>9&num_nonzero<1732,arr.ind=TRUE)
index_20=which(num_nonzero>19&num_nonzero<1732,arr.ind=TRUE)


#XGBoost model based on normalization dataset
data=data_normalize

R2_nonzero_XG<-vector() ###store the R2 for each test process
RMSE_nonzero_XG<-vector()  ###store the RMSE for each test process
MPE_nonzero_XG<-vector()   ###store the MPE for each test process

pre_nonzero=vector() ###store the prediction results for each test process
true_nonzero=vector()  ###store the true value for each test process
simple_flow_flag=vector()
flow_flag=vector()
process_flag=vector()
num_flow=vector()
num_flow_predicted=vector()

### parameters used for cross-validation
eta_best=vector()
nrounds_best=vector()
max_depth_best=vector()
eta_range=c(0.01, 0.05, 0.1,0.2)
nrounds_range = seq(from=200, to=400, by=100)
max_depth_range = seq(from=2, to=4, by=1)


### iteratively construct the model
for (j in 1:length(index_10)){
  print(j)
  nonzero=which(data[,index_10[j]]!=0, arr.ind=TRUE)  #the non-zero index for the target process
  nonzeroidx=intersect(nonzero,index_2_row_feasible)
  if (length(nonzeroidx)!=0){
    zeroidx=which(data[,index_10[j]]==0, arr.ind=TRUE)  #the zero index for the target process
    smp_size_withvalue <- ceiling(0.1 * length(nonzeroidx)) # %missing part for the process with value
    smp_size_zero <- ceiling(0.1 * length(zeroidx)) # %missing part for the process with value of 0
    miss_withvalue <- sample(length(nonzeroidx), size = smp_size_withvalue)
    miss_zero <- sample(length(zeroidx), size = smp_size_zero)
    miss_index_withvalue <-nonzeroidx[miss_withvalue]
    miss_index_zero <-zeroidx[miss_zero]
    test_idx=c(miss_index_withvalue,miss_index_zero)
    test <- data[test_idx,]
    num_flow=cbind(num_flow,length(nonzeroidx))
    num_flow_predicted=cbind(num_flow_predicted,length(miss_index_withvalue))
    pre=vector()
    true=vector()
    eta_value=vector()
    nrounds_value=vector()
    max_depth_value=vector()
    
    for (k in 1:length(miss_index_withvalue)){
      ### data preprocessing for training set
      train <- t(data[-test_idx,-index_10[j]])
      traindata1 <- data.matrix(train)  
      traindata2 <- Matrix(traindata1,sparse=T)
      traindata3 <- t(data[miss_index_withvalue[k],-index_10[j]])
      traindata4 <- list(data=traindata2,label=traindata3)
      dtrain <- xgb.DMatrix(data = traindata4$data, label = traindata4$label)
      ####data preprocessing for test set
      testset1 <- t(data.matrix(data[-test_idx,index_10[j]]))
      testset2 <- Matrix(testset1,sparse=T)
      testset3 <- t(data[miss_index_withvalue[k],index_10[j]])
      testset4 <- list(data=testset2,label=testset3)
      dtest <- xgb.DMatrix(data = testset4$data, label = testset4$label)
      min_index=vector()
      min_value=vector()
      a_value=vector()
      b_value=vector()
      c_value=vector()
      number=0
      for (a in eta_range){
        for (b in nrounds_range){
          for (c in max_depth_range){
            cv <- xgb.cv(data = dtrain, nrounds = b, nfold = 10, 
                         max_depth = c, eta = a, verbose=FALSE)
            number=number+1
            print(number)
            min_index=c(min_index,which.min(cv$evaluation_log$test_rmse_mean))
            min_value=c(min_value,min(cv$evaluation_log$test_rmse_mean))
            a_value=c(a_value,a)
            b_value=c(b_value,b)
            c_value=c(c_value,c)
          }
        }
      }
      
      index_best=which.min(min_value)
      xgb_best <- xgboost(data = dtrain,max_depth=c_value[index_best], eta=a_value[index_best], nrounds = b_value[index_best], verbose=FALSE)
      eta_value=c(eta_value,a_value[index_best])
      max_depth_value=c(max_depth_value,c_value[index_best])
      nrounds_value=c(nrounds_value,b_value[index_best])
      
      pre_best=predict(xgb_best,dtest)
      pre=c(pre,pre_best)
      true=c(true,data[miss_index_withvalue[k],index_10[j]])
      flow_flag=c(flow_flag,miss_index_withvalue[k])
      process_flag=c(process_flag,index_10[j])
    }
    pre_nonzero=c(pre_nonzero,pre)
    true_nonzero=c(true_nonzero,true)
    simple_flow_flag=c(simple_flow_flag,index_10[j])
    
    r_square_nonzero_XG<-R2_Score(pre,true)
    rmse_nonzero_XG<-RMSE(pre,true)
    mpe_nonzero_XG<-sum('^'((pre-true),2))/sum('^'(true,2))
    
    
    R2_nonzero_XG<-cbind(R2_nonzero_XG,r_square_nonzero_XG)
    RMSE_nonzero_XG<-cbind(RMSE_nonzero_XG,rmse_nonzero_XG)
    MPE_nonzero_XG<-cbind(MPE_nonzero_XG,mpe_nonzero_XG)
  }
}


r_square_nonzero<-R2_Score(pre_nonzero,true_nonzero)
rmse_nonzero<-RMSE(pre_nonzero,true_nonzero)
mpe_nonzero<-sum('^'((pre_nonzero-true_nonzero),2))/sum('^'(true_nonzero,2))
prediction<-cbind(process_flag,flow_flag,true_nonzero,pre_nonzero,eta_value,max_depth_value,nrounds_value)
prediction_performance<-cbind(simple_flow_flag,t(R2_nonzero_XG),t(RMSE_nonzero_XG),t(MPE_nonzero_XG),t(num_flow),t(num_flow_predicted))
colnames(prediction_performance)<-c("process","R2","RMSE","MPEs","number of flows","number of flows predicted")
performance<-c(r_square_nonzero,rmse_nonzero,mpe_nonzero,sqrtmpe_nonzero)

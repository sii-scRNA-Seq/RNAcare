setwd("/home/mt229a/Documents/djangoproject/IMID/static/temp/");
#args='mt229a'#
args<-commandArgs(trailingOnly = TRUE);
filename=paste(args[1],'corrected','clusters.csv',sep='_');#args[1]
#print(filename);
df<-read.csv(filename);
n<-ncol(df);
df1<-df[,-c(n-3,n-1,n-2)];
set.seed(1234);
fold<-sample(rep(seq(5),length=nrow(df1)));

library('glmnet')
library('plotmo')
x<-model.matrix(cluster~.,data=df1)
y<-factor(df1$cluster)
lasso.cv<-cv.glmnet(x,y,alpha=1,foldid=fold,family='multinomial',type.measure='class')

lasso.lam<-lasso.cv$lambda.min

lasso.mod<-glmnet(x,y,alpha=1,family='multinomial',type.measure='class')

png(filename=paste(args[1],args[3],'lasso.png',sep='_'))
plot_glmnet(lasso.mod,xvar='rlambda',label=TRUE,nresponse=as.numeric(args[2]))#args[2]
abline(v=log(lasso.lam),lty=2)
dev.off()


#z<-coef(lasso.mod,s=lasso.lam)[1][[1]]
#z.abs<-z %>% lapply(abs)
#first.five<-dimnames(z)[[1]][order(unlist(z))[c(1:5)]]
#last.five<-dimnames(z)[[1]][order(unlist(z),decreasing=TRUE)[c(1:5)]]

#z1<-coef(lasso.mod)[1][[1]]
#allnames<-names(z1[,ncol(z1)][order(z1[,ncol(z1)],decreasing=TRUE)])

#cols<-rep('0',length(allnames))
#cols[allnames %in% first.five]<-'red'
#cols[allnames %in% last.five]<-'green'

#library(plotmo)
#plot_glmnet(lasso.mod,xvar='rlambda',label=TRUE,nresponse=1,col=cols)#args[2]
#abline(v=log(lasso.lam),lty=2)

##########  Tabplot Demo  ##############            
library(xlsx)
library(tabplot)   
library(RColorBrewer)
## 
setwd("... your path...")
data <-read.xlsx('EXAFS.xlsx',1)
data <- unique(data)
feature<-data[,-c(26,28,30:33)] 
sum(is.na(feature)) 
which(rowSums(is.na(feature))> 0)

f_color<-brewer.pal(11,"Paired")
pal<-colorRampPalette(f_color)
f_color<-pal(11)

#27 features
tabplot::tableplot(feature,sortcol=Mineral,decreasing =T,
                   nBins=100,
                   select=c(Mineral,SSA, Fe_wt, Fe.O_at, 
                            PZC, Crystallinity, Cella, Cellb,
                            Cellc)
                   ,scales='lin')


tabplot::tableplot(feature,sortcol=Metal,decreasing =T,
                   nBins=100,
                   select=c(Metal, Radius, O_Num, Charge, Valence,
                            Electronegativity,  C_metal,
                            Electrolyte, ligand),
                   pals=list(Metal=f_color,
                             ligand=brewer.pal(3,"Spectral")), 
                   scales='lin')


tabplot::tableplot(feature,sortcol=Reaction,decreasing =T,
                   nBins=100,
                   select=c(Reaction, pHe, Oxygen, Recrystallization, 
                            C_mineral, C_OM, Time, Temperature, 
                            IonicStrength),
                    scales='lin')



###################  Feature selection  ###################
library(randomForest)
library(Metrics)  
library(randomForestExplainer)
library(ggplot2)
library(pheatmap)
library(xlsx)

impframe  <- data.frame()
corframe  <- data.frame()
setwd("...your path...")

### imp_Me-O CN Demo
data <-read.xlsx('Dummy.xlsx',1) 
EXAFS.data <- feature<-data[,c(1:31,33)]
EXAFS.data <- unique(EXAFS.data)
EXAFS.data[1:31] <- scale(EXAFS.data[1:31])
set.seed(0)
rf <- randomForest(CN~. , data = EXAFS.data, proximity = F,
                   importance =T)

imp_CN <- as.data.frame(rf$importance)
imp_CN[1] <- imp_CN[1]/max(imp_CN[1])

EXAFS.data <- feature<-data[,c(1:31,33)]
EXAFS.data <- unique(EXAFS.data)
EXAFS.data[1:31] <- scale(EXAFS.data[1:31])
pearson <-  cor(EXAFS.data)
 
### feature-corelation Demo
setwd("...your path...")
data <-read.xlsx('Dummy.xlsx',1) 
EXAFS.data <- feature<-data[,c(1:31)]
EXAFS.data <- unique(EXAFS.data)
EXAFS.data[1:31] <- scale(EXAFS.data[1:31])
cor <- as.data.frame(cor(EXAFS.data))
bk <- c(seq(-1,-0.01,by=0.001),seq(0,1,by=0.001))
pheatmap(as.matrix(cor),
         legend_breaks=seq(-1,1,0.5),
         breaks=seq(-1,1,0.02),
         cluster_col = FALSE, cluster_row = FALSE) 
write.csv(cor, '...your path.../Me-O Featcor.csv')


############# Clustering #############
#########  Coordination similarity network Demo  
library(randomForest)
library(Metrics)  
library(randomForestExplainer)
library(ggplot2)
library(pheatmap)
library(xlsx)
library(igraph)
library(RColorBrewer)

### Me-O/Fe coordination
setwd("...your path...")
data <-read.xlsx('Dummy.xlsx',1) 
EXAFS.data <- data[,c(1,2,4,9,11,12,15:22,31:32)]
EXAFS.data[sapply(EXAFS.data, is.character)] <- lapply(EXAFS.data[sapply(EXAFS.data, is.character)],as.factor)

EXAFS.data[1:15] <- scale(EXAFS.data[1:15])
set.seed(0)
rf <- randomForest(Bond~. , data = EXAFS.data,  ntree =400, proximity = T,
                   importance =F)  # keep the same parameter as gcForest
proximity_Bond<-rf$proximity 

write.csv(proximity_Bond,'Results/similarity/proximity_Bond_15.csv')


data1<-read.csv("Results/similarity/proximity_Bond_15.csv")
data_ck<-as.matrix(data1[,2:456])
data_nw<-as.matrix(data1[,2:456])
diag(data_nw) <- 0
data_nw[data_nw < 2*mean(data_ck)] <- 0 
data_nw[data_nw >= 2*mean(data_ck)] <- 1 

net<-graph.adjacency(adjmatrix=data_nw,mode="undirected",diag= FALSE) #聚类作图？
set.seed(12345) 
graph.density(net)


plot(net,layout= layout_with_kk,
     vertex.size=9,
     vertex.color=as.character(data1$ver.mineral), # as character() 
     vertex.label="",                           # as character() 
     vertex.label.size=0.5,
     vertex.label.cex=0.2,
     vertex.label.dist=0,
     vertex.label.color="black")

legend("bottom",c("Ferrihydrite","Goethite","Hematite","Lepidocrocite", 
                  "Maghemite", "Magnetite", "Schwertmannite"),
       pch=3, cex=0.1,
       col=rainbow(7))


plot(net,layout= layout_with_kk,
     vertex.size=9,
     vertex.color=as.character(data1$ver.metal), # as character()
     vertex.label="",                           # as character() 
     vertex.label.size=0.5,
     vertex.label.cex=0.2,
     vertex.label.dist=0,
     vertex.label.color="black")

legend("bottom",c("As","Cd","Co","Cr","Cu","Hg"
                  ,"Ni","Pb","Sb","Se", "Zn"),
       pch=16, cex=0.1,
       col=f_color)



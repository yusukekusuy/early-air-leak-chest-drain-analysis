## 01_explore.R  データ把握
suppressMessages({library(readxl); library(dplyr)})
skewness <- function(x){x<-x[!is.na(x)];m<-mean(x);mean((x-m)^3)/(mean((x-m)^2)^1.5)}
options(width=200)
d <- read_excel("C:/Users/Owner/Desktop/LiNGAM解析/データ.xlsx")
cat("N =", nrow(d), " cols =", ncol(d), "\n")
names(d)
cat("\n--- air_leak table ---\n"); print(table(d$`術中所見_air_leak`, useNA="always"))
y <- d$`Y=ドレーン抜去日数`
cat("\n--- Y (drain days) summary ---\n")
print(summary(y)); cat("SD:",sd(y),"\n")
cat("min,max:",min(y),max(y),"  any<=0:",any(y<=0),"  zeros:",sum(y==0),"\n")
cat("quantiles:\n"); print(quantile(y, c(.05,.25,.5,.75,.9,.95,.99)))
cat("skewness raw:", skewness(y), " skewness log:", skewness(log(y)), "\n")
cat("\n--- missingness per column ---\n")
print(sort(colSums(is.na(d)), decreasing=TRUE))
cat("\n--- binary var prevalence ---\n")
bin <- names(d)[sapply(d, function(x) is.numeric(x) && all(na.omit(x) %in% c(0,1)))]
for(b in bin) cat(sprintf("%-32s n1=%3d (%.1f%%)\n", b, sum(d[[b]]==1,na.rm=T), 100*mean(d[[b]]==1,na.rm=T)))
cat("\n--- sex table ---\n"); print(table(d$`性別`))

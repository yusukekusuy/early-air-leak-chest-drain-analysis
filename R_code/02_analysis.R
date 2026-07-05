## 02_analysis.R  主解析（対数変換線形回帰）
## ---- portable paths (GitHub-ready; NO absolute paths, NO patient data in repo) ----
## Run this script from the R_code/ directory (e.g. `Rscript 02_analysis.R`).
## Patient-level data are NOT distributed. To regenerate the .rds files locally,
## place the source spreadsheet at ./data/データ.xlsx ; both ./data/ and
## ./figures/ are git-ignored so no patient-level data can be committed.
data_dir <- "data"; fig_dir <- "figures"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
suppressMessages({library(readxl); library(dplyr); library(car); library(MASS); library(lmtest); library(survival)})
options(width=200, scipen=6)
d <- read_excel(file.path(data_dir, "データ.xlsx"))
names(d) <- c("age","sex","bmi","smk","copd","ip","neoadj","vc","fev1","dlco",
              "tumor","alb","hb","egfr","bs","hba1c","crp","asa","robot","lobectomy",
              "lymph","adhesion","fissure","airleak","vio","sealant","suture","optime","sedation","y")
d$male <- as.integer(d$sex=="男")
y <- d$y; ly <- log(y)

cat("====== Y distribution ======\n")
cat(sprintf("median %.0f (IQR %.0f-%.0f), mean %.2f, range %d-%d, n=%d\n",
            median(y),quantile(y,.25),quantile(y,.75),mean(y),min(y),max(y),length(y)))

cat("\n====== method check: log-OLS residuals & lognormal AFT equivalence ======\n")
# null + a few predictors to check residual normality of log(y)
m0 <- lm(ly ~ copd+fev1+alb+lobectomy+sealant+suture, data=d)
sw <- shapiro.test(resid(m0)); bp <- bptest(m0)
cat(sprintf("Shapiro W=%.3f p=%.3g ; Breusch-Pagan p=%.3g\n", sw$statistic, sw$p.value, bp$p.value))
aft <- survreg(Surv(y, rep(1,length(y))) ~ copd+fev1+alb+lobectomy+sealant+suture, data=d, dist="lognormal")
cmp <- cbind(logOLS=coef(m0), AFT_lognormal=coef(aft)[names(coef(m0))])
cat("coef comparison (log-OLS vs lognormal AFT, should match):\n"); print(round(cmp,4))

## ---- univariate ----
cont <- c("age","bmi","smk","vc","fev1","dlco","tumor","alb","hb","egfr","bs","hba1c","crp","asa","optime","sedation")
bins <- c("male","copd","ip","neoadj","robot","lobectomy","lymph","adhesion","fissure","airleak","vio","sealant","suture")
uni <- list()
for(v in cont){
  z <- scale(d[[v]])[,1]; fit <- lm(ly ~ z)
  ci <- confint(fit)[2,]; b<-coef(fit)[2]
  uni[[v]] <- data.frame(var=v,type="cont(perSD)",ratio=exp(b),lo=exp(ci[1]),hi=exp(ci[2]),p=summary(fit)$coef[2,4])
}
for(v in bins){
  fit <- lm(ly ~ d[[v]]); ci <- confint(fit)[2,]; b<-coef(fit)[2]
  uni[[v]] <- data.frame(var=v,type="binary",ratio=exp(b),lo=exp(ci[1]),hi=exp(ci[2]),p=summary(fit)$coef[2,4])
}
U <- do.call(rbind, uni); U <- U[order(U$p),]
cat("\n====== UNIVARIATE (ratio of geometric-mean drain days) ======\n")
print(data.frame(var=U$var,type=U$type,ratio=round(U$ratio,3),CI=paste0(round(U$lo,3),"-",round(U$hi,3)),p=signif(U$p,3)), row.names=FALSE)

## ---- multivariable full (clinically+LiNGAM candidates) ----
cand <- c("age","smk","copd","fev1","dlco","alb","bmi","lobectomy","robot","sealant","suture","fissure","adhesion","airleak","optime")
f_full <- as.formula(paste("ly ~", paste(cand, collapse="+")))
mfull <- lm(f_full, data=d)
cat("\n====== MULTIVARIABLE (full candidate model) ======\n"); print(round(summary(mfull)$coef,4))
cat("adj R2:", round(summary(mfull)$adj.r.squared,3),"\n")
cat("\nVIF:\n"); print(round(vif(mfull),2))

## ---- stepwise AIC as guide ----
cat("\n====== stepAIC (both) ======\n")
mstep <- stepAIC(mfull, direction="both", trace=FALSE)
cat("selected:", paste(names(coef(mstep))[-1],collapse=", "),"\n")
print(round(summary(mstep)$coef,4)); cat("adj R2:",round(summary(mstep)$adj.r.squared,3),"\n")
saveRDS(d, file.path(data_dir, "data_clean.rds"))

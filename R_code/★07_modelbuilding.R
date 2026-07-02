## 07 変数選択過程・適合度・残差診断・候補モデル(補足S3)
suppressMessages({library(car); library(MASS); library(lmtest); library(survival)})
options(width=200, scipen=6)
d <- readRDS("C:/Users/Owner/Desktop/LiNGAM解析/R/data_clean.rds"); d$ly<-log(d$y)
outdir <- "C:/Users/Owner/Desktop/LiNGAM解析/figures"

jp <- c(age="年齢/age",smk="喫煙指数/smoking",copd="COPD",fev1="FEV1.0%",dlco="DLCO",alb="アルブミン/albumin",
        bmi="BMI",lobectomy="葉切除以上/lobectomy+",robot="ロボット/robot",sealant="シーラント/sealant",
        suture="肺縫合/lung suture",fissure="不全葉間/incomplete fissure",adhesion="癒着/adhesion",
        airleak="術中空気漏れ/intraop air leak",optime="手術時間/operative time")
cand <- names(jp)
contv <- c("age","smk","fev1","dlco","alb","bmi","optime")  # continuous -> per SD
ds <- d
for(v in contv) ds[[v]] <- scale(d[[v]])[,1]
f <- as.formula(paste("ly ~", paste(cand, collapse="+")))
mf <- lm(f, data=ds)
co<-summary(mf)$coef; ci<-confint(mf); v<-vif(mf)
unit <- ifelse(cand %in% contv, "/SD", "")
cat("===== CANDIDATE MULTIVARIABLE MODEL (cont per SD) =====\n")
tab<-data.frame(var=paste0(jp[cand],unit),
   ratio=sprintf("%.3f",exp(co[cand,1])),
   CI=sprintf("%.3f-%.3f",exp(ci[cand,1]),exp(ci[cand,2])),
   p=signif(co[cand,4],3), VIF=sprintf("%.2f",v[cand]))
print(tab,row.names=FALSE)
cat(sprintf("Candidate: adj R2=%.3f, AIC=%.1f, n=%d, max VIF=%.2f\n",
   summary(mf)$adj.r.squared, AIC(mf), nobs(mf), max(v)))
write.csv(tab, file.path(outdir,"tableS3_candidate.csv"), row.names=FALSE, fileEncoding="UTF-8")

cat("\n===== stepAIC (both) selected =====\n")
ms<-stepAIC(mf, direction="both", trace=FALSE)
cat("selected:", paste(names(coef(ms))[-1],collapse=", "),"\n")
cat(sprintf("stepAIC: adj R2=%.3f, AIC=%.1f\n", summary(ms)$adj.r.squared, AIC(ms)))

cat("\n===== FINAL 6-var model (binary clinical cutoffs) =====\n")
## Final bedside model: determinants entered at pre-specified clinical cutoffs so the
## model is consistent with the weighted bedside score (COPD label -> spirometry
## FEV1.0%<70 + DLCO<80; operative time excluded as mediator).
d$rf_fev1<-d$fev1<70; d$rf_dlco<-d$dlco<80; d$rf_alb<-d$alb<3.8
d$rf_fissure<-d$fissure==1; d$rf_airleak<-d$airleak==1; d$rf_suture<-d$suture==1
fin<-lm(ly~rf_airleak+rf_fev1+rf_dlco+rf_alb+rf_fissure+rf_suture, data=d)
cat(sprintf("Final: adj R2=%.3f, AIC=%.1f, F=%.1f on %d/%d df, overall p=%.3g\n",
  summary(fin)$adj.r.squared, AIC(fin),
  summary(fin)$fstatistic[1], summary(fin)$fstatistic[2], summary(fin)$fstatistic[3],
  pf(summary(fin)$fstatistic[1],summary(fin)$fstatistic[2],summary(fin)$fstatistic[3],lower.tail=FALSE)))
cat("null AIC:", AIC(lm(ly~1,data=d)),"\n")

## joint significance of the pulmonary-function block (FEV1.0%<70 + DLCO<80, 2 df)
red<-lm(ly~rf_airleak+rf_alb+rf_fissure+rf_suture, data=d)
cat(sprintf("Joint F test (FEV1.0%%<70+DLCO<80, 2 df): p=%.3f\n", anova(red,fin)[2,"Pr(>F)"]))

cat("\n===== diagnostics (final) =====\n")
sw<-shapiro.test(resid(fin)); bp<-bptest(fin)
cat(sprintf("Shapiro-Wilk W=%.3f p=%.3g ; Breusch-Pagan p=%.3g ; max VIF(final)=%.2f\n",
  sw$statistic, sw$p.value, bp$p.value, max(vif(fin))))
aft<-survreg(Surv(y,rep(1,nrow(d)))~rf_airleak+rf_fev1+rf_dlco+rf_alb+rf_fissure+rf_suture,data=d,dist="lognormal")
cat("log-OLS vs lognormal-AFT coef max abs diff:",
  max(abs(coef(fin)-coef(aft)[names(coef(fin))])),"\n")

cat("\n===== weighting: integer points from final-model coefficients =====\n")
b<-coef(fin)[-1]; pts<-round(b/min(b))
print(data.frame(factor=sub("^rf_","",names(b)), coef=round(b,3), points=as.integer(pts)), row.names=FALSE)

cat("\n===== sensitivity analyses =====\n")
d$rf_copd<-d$copd==1
sens1<-lm(ly~rf_airleak+rf_copd+rf_alb+rf_fissure+rf_suture, data=d)            # COPD label instead of spirometry cutoffs
sens2<-lm(ly~rf_airleak+rf_fev1+rf_dlco+rf_alb+rf_fissure+rf_suture+optime, data=d)  # add operative time (mediator) back
cat(sprintf("COPD-replacement adj R2=%.3f ; +operative-time adj R2=%.3f\n",
  summary(sens1)$adj.r.squared, summary(sens2)$adj.r.squared))

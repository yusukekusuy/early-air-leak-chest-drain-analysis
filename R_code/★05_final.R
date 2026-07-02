## 05_final.R  CANONICAL final analysis for Manuscript_English.docx / Supplement_English.docx.
## Final 6-factor model (operative time excluded as a mediator; COPD label replaced by
## measured FEV1.0% + DLCO). Weighted integer risk points, and 3-group risk strata by
## TERTILES of the weighted score (Low 0-5 / Intermediate 6-9 / High >=10).
## Generates: Table 1 (baseline), Table 2 (final model + points), Fig 3 (drain-day
## distribution), Fig 5 (composite KM), Fig 6 (box plot), and the per-factor KM panel.
suppressMessages({library(dplyr); library(survival); library(survminer); library(pROC)})
options(width=200, scipen=6)
outdir <- "C:/Users/Owner/Desktop/LiNGAM解析/figures"
d <- readRDS("C:/Users/Owner/Desktop/LiNGAM解析/R/data_clean.rds")
d$ly <- log(d$y); d$prolong <- as.integer(d$y>=7)

## ---- 6 pre-specified clinical-cutoff risk factors ----
## Continuous variables (FEV1.0%, DLCO, Albumin) are entered into the bedside model
## as their pre-specified CLINICAL-CUTOFF binaries, so the multivariable model and the
## bedside risk-stratification score are mutually consistent. Continuous granularity is
## retained only in the LiNGAM causal map (Fig 4).
d$rf_fev1<-d$fev1<70; d$rf_dlco<-d$dlco<80; d$rf_alb<-d$alb<3.8
d$rf_fissure<-d$fissure==1; d$rf_airleak<-d$airleak==1; d$rf_suture<-d$suture==1
rfs<-c("rf_airleak","rf_fev1","rf_dlco","rf_alb","rf_fissure","rf_suture")
lab<-c("Intraoperative air leak","FEV1.0% <70%","DLCO <80%","Albumin <3.8 g/dL","Incomplete fissure","Lung suture")

## ---- FINAL MODEL (6 binary factors) ----
## optime (mediator/surrogate of operative difficulty) excluded from the structure;
## COPD diagnostic label replaced by measured spirometry (FEV1.0% + DLCO).
fin <- lm(ly ~ rf_airleak + rf_fev1 + rf_dlco + rf_alb + rf_fissure + rf_suture, data=d)
co<-summary(fin)$coef; ci<-confint(fin)
## effect size: adjusted ratio of geometric-mean drain duration (factor present vs absent)
ratio<-exp(co[,1]); lo<-exp(ci[,1]); hi<-exp(ci[,2])
fintab<-data.frame(Variable=c("Intercept",lab), Ratio=sprintf("%.2f",ratio),
                   CI=sprintf("%.2f-%.2f",lo,hi), p=signif(co[,4],3))[-1,]
cat("===== FINAL MODEL (adjusted GM-duration ratio; binary clinical-cutoff factors) =====\n"); print(fintab,row.names=FALSE)
cat("adj R2:",round(summary(fin)$adj.r.squared,3)," AIC:",round(AIC(fin),1),
    " F:",round(summary(fin)$fstatistic[1],1)," n:",nobs(fin),"\n")
write.csv(fintab, file.path(outdir,"table_finalmodel.csv"), row.names=FALSE, fileEncoding="UTF-8")

## ---- integer-point weights from the multivariable coefficients ----
## Points proportional to each factor's adjusted log geometric-mean ratio, scaled to the
## smallest coefficient and rounded (Framingham-style weighting). This is an IN-SAMPLE,
## DESCRIPTIVE weighting for risk stratification, NOT a validated prediction rule
## (external/temporal validation is positioned as future work).
bcoef <- co[-1,1]
pts <- round(bcoef/min(bcoef)); names(pts)<-rfs
wtab <- data.frame(Factor=lab, Ratio=sprintf("%.2f",exp(bcoef)), Points=as.integer(pts))
cat("\n===== weighted risk points (max =",sum(pts),") =====\n"); print(wtab,row.names=FALSE)
write.csv(wtab, file.path(outdir,"table_riskpoints.csv"), row.names=FALSE, fileEncoding="UTF-8")

## ---- per-factor descriptive table ----
pf<-data.frame()
for(i in seq_along(rfs)){g<-d[[rfs[i]]]; rt<-exp(coef(lm(ly~g,data=d))[2]); rci<-exp(confint(lm(ly~g,data=d))[2,])
  pf<-rbind(pf,data.frame(Factor=lab[i], "n(%)"=sprintf("%d (%.1f)",sum(g),100*mean(g)),
    Points=as.integer(pts[i]),
    median_present=median(d$y[g]), median_absent=median(d$y[!g]),
    prolong_present=sprintf("%.1f",100*mean(d$prolong[g])), prolong_absent=sprintf("%.1f",100*mean(d$prolong[!g])),
    Ratio=sprintf("%.2f (%.2f-%.2f)",rt,rci[1],rci[2]), check.names=FALSE))}
cat("\n===== per-factor =====\n"); print(pf,row.names=FALSE)
write.csv(pf, file.path(outdir,"table_riskfactors.csv"), row.names=FALSE, fileEncoding="UTF-8")

## ---- WEIGHTED composite score + descriptive (in-sample) discrimination ----
d$score <- as.numeric(as.matrix(d[,rfs]) %*% pts)   # weighted points, range 0 - sum(pts)
cat("\nweighted score dist:\n"); print(table(d$score))
## reported as a descriptive in-sample c-statistic only (NOT a prediction claim)
cat("in-sample c-statistic (score vs >=7 d), descriptive:",
    round(as.numeric(auc(roc(d$prolong,d$score,quiet=TRUE))),3),"\n")
print(d %>% group_by(score) %>% summarise(n=n(),medianDrain=median(y),pctProlong=round(100*mean(prolong),1)) %>% as.data.frame())

## ---- 3-group strata (MAIN) by tertiles of the weighted score ----
qs <- as.numeric(quantile(d$score, c(1/3,2/3)))
cat(sprintf("\ntertile cutpoints of weighted score: %g / %g\n", qs[1], qs[2]))
lo_lbl<-sprintf("Low risk (0-%g points)",qs[1])
in_lbl<-sprintf("Intermediate (%g-%g points)",qs[1]+1,qs[2])
hi_lbl<-sprintf("High risk (≥%g points)",qs[2]+1)
d$risk3 <- cut(d$score, breaks=c(-Inf,qs[1],qs[2],Inf), labels=c(lo_lbl,in_lbl,hi_lbl))
d$highrisk<-factor(ifelse(d$score>qs[2],"High risk","Low risk"),levels=c("Low risk","High risk")) # 参考2群
pal3 <- c("#0072B2","#E69F00","#D55E00")

stat <- d %>% group_by(risk3) %>%
  summarise(n=n(), median=median(y), q1=quantile(y,.25), q3=quantile(y,.75),
            gm=exp(mean(log(y))), pct7=round(100*mean(prolong),1)) %>% as.data.frame()
cat("\n===== 3-group strata =====\n"); print(stat,row.names=FALSE)
kw<-kruskal.test(y~risk3,data=d); cat(sprintf("Kruskal-Wallis p=%.3g\n",kw$p.value))
lr3<-survdiff(Surv(y,rep(1,nrow(d)))~risk3,data=d); lr3p<-1-pchisq(lr3$chisq,length(lr3$n)-1)
cat(sprintf("log-rank(3grp) chi2=%.1f p=%.3g\n",lr3$chisq,lr3p))
pw<-pairwise.wilcox.test(d$y,d$risk3,p.adjust.method="holm"); cat("pairwise Wilcoxon(Holm):\n"); print(signif(pw$p.value,3))
sp<-suppressWarnings(cor.test(d$score,d$y,method="spearman")); cat(sprintf("Spearman trend rho=%.3f p=%.3g\n",sp$estimate,sp$p.value))

kwfmt<-if(kw$p.value<.001)"Kruskal-Wallis P < .001" else sprintf("Kruskal-Wallis P = %.3f",kw$p.value)
lrfmt<-if(lr3p<.001)"log-rank P < .001" else sprintf("log-rank P = %.3f",lr3p)

## figure strata labels (weighted points)
flab <- c(sprintf("Low risk (0-%g pts)",qs[1]), sprintf("Intermediate (%g-%g pts)",qs[1]+1,qs[2]),
          sprintf("High risk (≥%g pts)",qs[2]+1))

## ---- composite KM (3-group, MAIN = Figure 5) ----  (enlarged fonts)
fit3<-survfit(Surv(y,rep(1,nrow(d)))~risk3,data=d)
g<-ggsurvplot(fit3,data=d,fun="event",risk.table=TRUE,pval=lrfmt,pval.coord=c(12.5,0.22),pval.size=6,conf.int=TRUE,
   palette=pal3,legend.labs=flab,fontsize=5,
   xlab="Days after surgery",ylab="Cumulative incidence of drain removal",
   legend.title="",break.time.by=5,xlim=c(0,21),risk.table.height=0.30,
   font.x=16,font.y=16,font.tickslab=14,font.legend=15,
   risk.table.title="Number at risk",risk.table.fontsize=5,
   tables.theme=theme_survminer(font.main=15,font.x=15,font.y=14,font.tickslab=13))
png(file.path(outdir,"fig_km_composite.png"),width=2000,height=1800,res=210); print(g); dev.off()

## ---- box plot (3-group, MAIN = Figure 6) ----  (enlarged fonts)
ymax_disp<-20
med_lab<-stat %>% mutate(txt=sprintf("median %d d\n(IQR %g-%g)",median,q1,q3))
xlabs<-c(sprintf("Low risk\n(0-%g pts)",qs[1]), sprintf("Intermediate\n(%g-%g pts)",qs[1]+1,qs[2]),
         sprintf("High risk\n(≥%g pts)",qs[2]+1))
pb<-ggplot(d,aes(risk3,y,fill=risk3,color=risk3))+
  geom_jitter(width=.18,height=.15,alpha=.22,size=1.0)+
  geom_boxplot(width=.55,alpha=.55,outlier.shape=NA,color="grey20",linewidth=.5)+
  stat_summary(fun=median,geom="point",shape=23,size=3.4,fill="white",color="grey20")+
  geom_text(data=med_lab,aes(x=risk3,y=ymax_disp*0.95,label=txt),inherit.aes=FALSE,size=4.6,lineheight=.9)+
  annotate("text",x=2,y=ymax_disp*0.74,label=kwfmt,size=5.2)+
  scale_fill_manual(values=pal3,guide="none")+scale_color_manual(values=pal3,guide="none")+
  scale_x_discrete(labels=xlabs)+coord_cartesian(ylim=c(0,ymax_disp))+
  labs(x=NULL,y="Days to chest drain removal")+theme_classic(base_size=16)+
  theme(axis.text.x=element_text(face="bold"))
ggsave(file.path(outdir,"fig_box_risk.png"),pb,width=6.6,height=5.4,dpi=210)

## ---- per-factor KM panel (Multimedia Appendix 6) ----
## AMA/JMIR style P-value label: italic P, no leading zero, <.001
lab_p<-function(p){
  if(p<.001) return(bquote(italic(P) ~ "<" ~ ".001"))
  ps<-if(p<.01) sub("^0","",sprintf("%.3f",p)) else sub("^0","",sprintf("%.2f",p))
  bquote(italic(P) ~ "=" ~ .(ps))
}
d$status<-1
png(file.path(outdir,"figS_km_perfactor.png"),width=2000,height=1400,res=160)
par(mfrow=c(2,3),mar=c(4,4,2.2,1))
for(i in seq_along(rfs)){ d$grp<-d[[rfs[i]]]
  f<-survfit(Surv(y,status)~grp,data=d)
  plot(f,fun="event",col=c("#0072B2","#D55E00"),lwd=2,xlim=c(0,21),xlab="Days after surgery",ylab="Cumulative incidence of drain removal",main=lab[i])
  p<-1-pchisq(survdiff(Surv(y,status)~grp,data=d)$chisq,1)
  legend("bottomright",c("Absent","Present"),col=c("#0072B2","#D55E00"),lwd=2,bty="n",cex=.95)
  legend("right",legend=lab_p(p),bty="n",cex=.95)}
dev.off()
## ---- Table 1: baseline characteristics of the early-air-leak cohort (n=556) ----
## (model-independent descriptive table; moved here from the retired 04_figures_tables.R)
contv<-c("age","bmi","smk","vc","fev1","dlco","tumor","alb","hb","egfr","bs","hba1c","crp","asa","optime","sedation","y")
binv<-c("male","copd","ip","neoadj","robot","lobectomy","lymph","adhesion","fissure","airleak","vio","sealant","suture")
t1<-data.frame()
for(v in contv) t1<-rbind(t1,data.frame(variable=v, summary=sprintf("%.1f [%.1f-%.1f]",median(d[[v]]),quantile(d[[v]],.25),quantile(d[[v]],.75))))
for(v in binv) t1<-rbind(t1,data.frame(variable=v, summary=sprintf("%d (%.1f%%)",sum(d[[v]]==1),100*mean(d[[v]]==1))))
write.csv(t1, file.path(outdir,"table1_baseline.csv"), row.names=FALSE, fileEncoding="UTF-8")
cat("\nTable 1 baseline written. rows:",nrow(t1),"\n")

## ---- Figure 3: chest-drain duration distribution ----
## (model-independent; moved here from the retired 04_figures_tables.R)
png(file.path(outdir,"fig_distribution.png"), width=1500, height=900, res=200)
par(mar=c(4.2,4.2,2,1))
hist(d$y, breaks=seq(0,42,1), col="grey70", border="white",
     xlab="Chest drain duration (days)", ylab="Number of patients", main="")
abline(v=median(d$y), col="red", lwd=2, lty=2)
legend("topright", sprintf("Median = %d days",median(d$y)), col="red", lty=2, lwd=2, bty="n")
dev.off()

cat("\nDONE. figures+tables updated.\n")
saveRDS(d,"C:/Users/Owner/Desktop/LiNGAM解析/R/data_risk.rds")

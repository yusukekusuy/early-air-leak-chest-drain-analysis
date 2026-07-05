## 06 単変量全結果テーブル（補足S1）日本語ラベル
## ---- portable paths (GitHub-ready; NO absolute paths, NO patient data in repo) ----
## Run this script from the R_code/ directory (e.g. `Rscript 02_analysis.R`).
## Patient-level data are NOT distributed. To regenerate the .rds files locally,
## place the source spreadsheet at ./data/データ.xlsx ; both ./data/ and
## ./figures/ are git-ignored so no patient-level data can be committed.
data_dir <- "data"; fig_dir <- "figures"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
suppressMessages(library(dplyr)); options(scipen=6)
d <- readRDS(file.path(data_dir, "data_clean.rds")); d$ly<-log(d$y)
jp <- c(age="年齢(/SD)",bmi="BMI(/SD)",smk="喫煙指数(/SD)",vc="%VC(/SD)",fev1="FEV1.0%(/SD)",dlco="DLCO(/SD)",
  tumor="推計腫瘍径(/SD)",alb="アルブミン(/SD)",hb="Hb(/SD)",egfr="eGFR(/SD)",bs="血糖(/SD)",hba1c="HbA1c(/SD)",
  crp="CRP(/SD)",asa="ASA-PS(/SD)",optime="手術時間(/SD)",sedation="覚醒スコア(/SD)",
  male="男性",copd="COPD",ip="間質性肺炎",neoadj="術前治療",robot="ロボット支援",lobectomy="葉切除以上",
  lymph="リンパ節郭清",adhesion="胸膜癒着",fissure="不全葉間",airleak="術中空気漏れ",vio="焼灼(VIO)",
  sealant="シーラント",suture="肺縫合")
cont<-c("age","bmi","smk","vc","fev1","dlco","tumor","alb","hb","egfr","bs","hba1c","crp","asa","optime","sedation")
bins<-c("male","copd","ip","neoadj","robot","lobectomy","lymph","adhesion","fissure","airleak","vio","sealant","suture")
res<-data.frame()
for(v in cont){z<-scale(d[[v]])[,1];f<-lm(d$ly~z);ci<-confint(f)[2,];b<-coef(f)[2]
  res<-rbind(res,data.frame(変数=jp[v],比=sprintf("%.2f",exp(b)),`95%CI`=sprintf("%.2f-%.2f",exp(ci[1]),exp(ci[2])),p=signif(summary(f)$coef[2,4],3),check.names=FALSE))}
for(v in bins){f<-lm(d$ly~d[[v]]);ci<-confint(f)[2,];b<-coef(f)[2]
  res<-rbind(res,data.frame(変数=jp[v],比=sprintf("%.2f",exp(b)),`95%CI`=sprintf("%.2f-%.2f",exp(ci[1]),exp(ci[2])),p=signif(summary(f)$coef[2,4],3),check.names=FALSE))}
res<-res[order(as.numeric(res$p)),]
write.csv(res,file.path(fig_dir, "tableS1_univariate.csv"),row.names=FALSE,fileEncoding="UTF-8")
print(res,row.names=FALSE); cat("\nrows:",nrow(res),"\n")

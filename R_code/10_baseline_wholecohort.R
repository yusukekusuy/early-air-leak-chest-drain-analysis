## 10_baseline_wholecohort.R
## ---- portable paths (GitHub-ready; NO absolute paths, NO patient data in repo) ----
## Run this script from the R_code/ directory (e.g. `Rscript 02_analysis.R`).
## Patient-level data are NOT distributed. To regenerate the .rds files locally,
## place the source spreadsheet at ./data/データ.xlsx ; both ./data/ and
## ./figures/ are git-ignored so no patient-level data can be committed.
data_dir <- "data"; fig_dir <- "figures"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
## 全集団(LiNGAM 第1段階, n=1139)のベースライン比較表
## LiNGAM 投入変数(Multimedia Appendix 5 の ✓ セット)を 早期空気漏れ(POD1まで) 有/無 で比較。
## 出力: figures/table_baseline_wholecohort.csv, manuscript/_baseline_ja.md, manuscript/_baseline_en.md
suppressMessages({library(readxl)})
options(stringsAsFactors=FALSE)
root <- "."   # unused; paths below are portable
d <- read_excel(file.path(data_dir,"データ.xlsx"), sheet="全集団")
d <- as.data.frame(d)
grp <- d[["初期リーク（POD1まで）"]]              # 1 = early leak, 0 = no
stopifnot(all(grp %in% c(0,1)))
cat(sprintf("n total=%d | early-leak yes=%d | no=%d\n", length(grp), sum(grp==1), sum(grp==0)))

## ---- AMA/JMIR P-value formatter ----
fmtP <- function(p){
  if(is.na(p)) return("—")
  if(p < .001) return("<.001")
  if(p < .01)  return(sub("^0","",sprintf("%.3f",p)))
  sub("^0","",sprintf("%.2f",p))
}
endash <- "–"
fmt_cont <- function(x, dec){
  q <- quantile(x, c(.5,.25,.75), na.rm=TRUE)
  sprintf(paste0("%.",dec,"f [%.",dec,"f",endash,"%.",dec,"f]"), q[1], q[2], q[3])
}
fmt_bin <- function(b){ sprintf("%d (%.1f)", sum(b==1,na.rm=TRUE), 100*mean(b==1,na.rm=TRUE)) }

## ---- variable spec: key, ja, en, type, column, decimals ----
spec <- list(
  list("age","年齢（歳）","Age (years)","cont","年齢",0),
  list("sex","性別（男性）","Male sex","binsex","性別",0),
  list("bmi","BMI","BMI","cont","BMI",1),
  list("smk","喫煙指数","Smoking index","cont","喫煙指数",0),
  list("copd","COPD","COPD","bin","併存症COPD",0),
  list("ip","間質性肺炎","Interstitial pneumonia","bin","併存症IP",0),
  list("asa","ASA-PS","ASA-PS","cont","術前全身評価_ASA PS",0),
  list("vc","%VC","%VC","cont","肺機能.%VC",1),
  list("fev","FEV1.0%","FEV1.0%","cont","肺機能.FEV1.0%",1),
  list("dlco","DLCO","DLCO","cont","肺機能.DLCO",1),
  list("tum","推計腫瘍径（cm）","Estimated tumor size (cm)","cont","推計腫瘍サイズ",1),
  list("alb","アルブミン（g/dL）","Albumin (g/dL)","cont","術前_Alb",1),
  list("hb","Hb（g/dL）","Hemoglobin (g/dL)","cont","術前_Hb",1),
  list("egfr","eGFR","eGFR","cont","術前_eGFR",1),
  list("bs","血糖（mg/dL）","Blood glucose (mg/dL)","cont","術前_BS",0),
  list("hba1c","HbA1c（%）","HbA1c (%)","cont","術前_HbA1c",1),
  list("crp","CRP（mg/dL）","CRP (mg/dL)","cont","術前_CRP",1),
  list("lob","葉切除以上","Lobectomy or greater","bin","T=lobectomy_or_more",0),
  list("robot","ロボット支援","Robot-assisted","bin","ロボット操作",0),
  list("lnd","リンパ節郭清","Lymph-node dissection","bin","術中所見_alymph_dissection",0),
  list("adh","胸膜癒着","Pleural adhesion","bin","術中所見_aadhesion",0),
  list("fis","不全葉間","Incomplete fissure","bin","術中所見_aincomplete_fissure",0),
  list("leak","術中空気漏れ","Intraoperative air leak","bin","術中所見_air_leak",0)
)

rows <- data.frame()
for(s in spec){
  key<-s[[1]]; ja<-s[[2]]; en<-s[[3]]; type<-s[[4]]; col<-s[[5]]; dec<-s[[6]]
  x <- d[[col]]
  if(type=="cont"){
    xn <- as.numeric(x)
    allc <- fmt_cont(xn, dec); noc <- fmt_cont(xn[grp==0], dec); yec <- fmt_cont(xn[grp==1], dec)
    p <- suppressWarnings(wilcox.test(xn ~ grp)$p.value)
  } else {
    b <- if(type=="binsex") as.integer(x=="男") else as.integer(x==1)
    allc <- fmt_bin(b); noc <- fmt_bin(b[grp==0]); yec <- fmt_bin(b[grp==1])
    p <- suppressWarnings(chisq.test(table(grp, b))$p.value)
  }
  rows <- rbind(rows, data.frame(ja=ja, en=en, all=allc, no=noc, yes=yec, P=fmtP(p)))
}

## ---- CSV ----
write.csv(rows, file.path(fig_dir,"table_baseline_wholecohort.csv"), row.names=FALSE, fileEncoding="UTF-8")

## ---- markdown (JA / EN) ----
mk <- function(lang){
  if(lang=="ja"){
    h <- "| 変数 | 全体（n=1139） | 早期リークなし（n=583） | 早期リークあり（n=556） | *P*値 |"
    lab <- rows$ja
  } else {
    h <- "| Variable | Overall (n=1139) | No early leak (n=583) | Early leak (n=556) | *P* value |"
    lab <- rows$en
  }
  out <- c(h, "|---|---|---|---|---|")
  for(i in seq_len(nrow(rows)))
    out <- c(out, sprintf("| %s | %s | %s | %s | %s |", lab[i], rows$all[i], rows$no[i], rows$yes[i], rows$P[i]))
  out
}
writeLines(mk("ja"), file.path(fig_dir,"_baseline_ja.md"), useBytes=TRUE)
writeLines(mk("en"), file.path(fig_dir,"_baseline_en.md"), useBytes=TRUE)
cat("DONE: baseline whole-cohort table written.\n")

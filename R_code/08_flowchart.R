## 08 inclusion/exclusion flowchart (CONSORT-style, side branches)
## ---- portable paths (GitHub-ready; NO absolute paths, NO patient data in repo) ----
## Run this script from the R_code/ directory (e.g. `Rscript 02_analysis.R`).
## Patient-level data are NOT distributed. To regenerate the .rds files locally,
## place the source spreadsheet at ./data/データ.xlsx ; both ./data/ and
## ./figures/ are git-ignored so no patient-level data can be committed.
data_dir <- "data"; fig_dir <- "figures"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
outdir <- fig_dir
png(file.path(outdir,"fig_flow.png"), width=2300, height=1620, res=210)
par(mar=c(0,0,0,0)); plot(0,0,type="n",xlim=c(0,100),ylim=c(8,98),axes=FALSE,xlab="",ylab="")
box2<-function(x,y,w,h,txt,cex=1.18){
  rect(x-w/2,y-h/2,x+w/2,y+h/2,col="grey97",border="grey30",lwd=1.7)
  text(x,y,txt,cex=cex)
}
vseg<-function(x,y0,y1) arrows(x,y0,x,y1,length=0.12,lwd=2.0,col="grey30")   # downward
hbr <-function(x0,x1,y) arrows(x0,y,x1,y,length=0.11,lwd=1.8,col="grey30")    # rightward branch
mx<-29                                   # main column x
# main boxes
box2(mx,89,56,14,"Patients undergoing surgery for primary lung cancer\n(admitted January 2017 – November 2025)\nn = 1241")
box2(mx,56,56,14,"VATS, including robot-assisted surgery\n(whole analysis cohort; Step 1, layered LiNGAM map, Fig. 4)\nn = 1139")
box2(mx,23,56,16,"Early air leak by postoperative day 1\n(landmark cohort for the duration analysis;\nStep 2, Tables 1–2, Figs 5–6)\nn = 556")
# vertical flow with side branches
vseg(mx,82,63)                           # A -> B
hbr(mx,61.6,74); box2(80,74,42,11,"Excluded:\nopen thoracotomy or conversion\nto thoracotomy (n = 102)",cex=1.18)
vseg(mx,48,31)                           # B -> C
hbr(mx,61.6,41); box2(80,41,42,10,"No air leak by\npostoperative day 1 (n = 583)",cex=1.18)
dev.off()
cat("flowchart saved\n")

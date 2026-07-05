## 09_pipeline_figure.R  Figure 2: end-to-end pipeline (with conceptual stage icons)
## ---- portable paths (GitHub-ready; NO absolute paths, NO patient data in repo) ----
## Run this script from the R_code/ directory (e.g. `Rscript 02_analysis.R`).
## Patient-level data are NOT distributed. To regenerate the .rds files locally,
## place the source spreadsheet at ./data/データ.xlsx ; both ./data/ and
## ./figures/ are git-ignored so no patient-level data can be committed.
data_dir <- "data"; fig_dir <- "figures"
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,  showWarnings = FALSE, recursive = TRUE)
## records -> GPT structures free text -> LiNGAM causal discovery (DAG) ->
##   physician-curated risk model -> stratified bedside score
## Emphasis: AI is used TWICE -- (1) LLM structures free-text data, (2) layered LiNGAM causal discovery.
suppressMessages({library(ggplot2); library(grid)})
outdir <- fig_dir

## Okabe-Ito palette
grey <- "#666666"; ai <- "#E69F00"; blue <- "#0072B2"; green <- "#009E73"; verm <- "#D55E00"
hdr_col <- c(grey, ai, blue, green, verm)

## ---- canvas / aspect ----
fig_w <- 16.6; fig_h <- 7.7
xlim0 <- c(0,100); ylim0 <- c(26,82)
asp <- (fig_h/diff(ylim0)) / (fig_w/diff(xlim0))   # x-units per y-unit for a true circle

## ---- 5 stage boxes ----
n <- 5; mar <- 2
gaps <- c(3.5, 3.5, 9, 3.5)
bw <- (100 - 2*mar - sum(gaps))/n
left <- mar + c(0, cumsum(bw + gaps[1:(n-1)])); right <- left + bw; cx <- left + bw/2
y_bot <- 28; y_top <- 74; y_hdr <- 63          # body 28-63, header 63-74

title <- c("Real-world\nsurgical\nrecords","AI\nstructuring","Causal\ndiscovery",
           "Risk\nmodel","Bedside\nrisk score")
tag   <- c("", "Generative AI (LLM)", "n = 1139 (whole cohort)",
           "n = 556 (early-leak cohort)", "")
body  <- c(
"• Free-text operative\n  notes\n• Structured EMR:\n  demographics, labs,\n  pulmonary function",
"GPT-5.4 (Azure OpenAI)\nextracts intraoperative\nfindings from free text:\n• pleural adhesion\n• incomplete fissure\n• air leak\n→ analyzable variables",
"Layered LiNGAM\n+ regression\n• temporal ordering of\n  variables\n• key intermediate event\n  = early air leak (POD1):\n  gateway to prolongation",
"Physician-curated\nmultivariable regression\non pre-specified clinical\ncutoffs:\n FEV1.0%<70, DLCO<80,\n Alb<3.8, fissure,\n air leak, lung suture\n→ adjusted ratio per factor",
"• coefficient-weighted\n  risk points (0–23)\n• 3-group risk\n  stratification\n  (low / intermediate /\n  high)\n• Kaplan–Meier + box plot\n→ chest-drain management")

hdr <- data.frame(xmin=left, xmax=right, ymin=y_hdr, ymax=y_top)
bdy <- data.frame(xmin=left, xmax=right, ymin=y_bot, ymax=y_hdr)

p <- ggplot() +
  geom_rect(data=bdy, aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax), fill="white",
            colour=hdr_col, linewidth=.7) +
  geom_rect(data=hdr, aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax), fill=hdr_col, colour=NA)

## ---- conceptual icon helpers (white line-art inside header) ----
icon_cy <- (y_hdr + y_top)/2 + 0.2
icon_dx <- 3.2                      # icon centre offset from box left
icon_s  <- 3.5                      # icon half-size (y-units)
lwI <- 0.9

add_icon <- function(p, type, ix, iy, s, col="white"){
  mx <- function(ux) ix + asp*s*ux
  my <- function(uy) iy + s*uy
  pth <- function(ux,uy, lw=lwI)
    annotate("path", x=mx(ux), y=my(uy), colour=col, linewidth=lw, lineend="round", linejoin="round")
  seg <- function(x0,y0,x1,y1, lw=lwI)
    annotate("segment", x=mx(x0), y=my(y0), xend=mx(x1), yend=my(y1), colour=col, linewidth=lw, lineend="round")
  arr <- function(x0,y0,x1,y1, lw=lwI)
    annotate("segment", x=mx(x0), y=my(y0), xend=mx(x1), yend=my(y1), colour=col, linewidth=lw,
             lineend="round", arrow=arrow(length=unit(0.09,"cm"), type="closed"))
  dot <- function(ux,uy, sz=1.7)
    annotate("point", x=mx(ux), y=my(uy), colour=col, fill=col, size=sz, shape=21, stroke=0.5)
  th <- seq(0, 2*pi, length.out=48)

  if(type=="docs"){            ## stack of operative notes
    p <- p + pth(c(-0.30,-0.30,0.62,0.62,-0.30)+0.12, c(-0.62,0.96,0.96,-0.62,-0.62)+0.10) +
             pth(c(-0.58,-0.58,0.18,0.52,0.52,-0.58), c(-0.92,0.72,0.72,0.38,-0.92,-0.92)) +
             pth(c(0.18,0.18,0.52), c(0.72,0.38,0.38)) +
             seg(-0.34,0.34,0.30,0.34) + seg(-0.34,0.04,0.30,0.04) + seg(-0.34,-0.26,0.08,-0.26)
  } else if(type=="text2tab"){ ## GPT: free text -> structured table
    p <- p + seg(-0.95,0.55,-0.50,0.55) + seg(-0.95,0.20,-0.55,0.20) +
             seg(-0.95,-0.15,-0.50,-0.15) + seg(-0.95,-0.50,-0.62,-0.50) +
             arr(-0.38,0.02,-0.02,0.02) +
             pth(c(0.12,0.12,0.95,0.95,0.12), c(-0.60,0.62,0.62,-0.60,-0.60)) +
             seg(0.535,-0.60,0.535,0.62) + seg(0.12,0.21,0.95,0.21) + seg(0.12,-0.20,0.95,-0.20)
  } else if(type=="dag"){      ## LiNGAM: upstream vars -> landmark -> outcome
    arr(-0.78,0.55,-0.10,0.10); arr(-0.78,0.00,-0.18,0.00); arr(-0.78,-0.55,-0.10,-0.10)
    p <- p + arr(-0.78,0.55,-0.12,0.08) + arr(-0.78,0.00,-0.18,0.00) + arr(-0.78,-0.55,-0.12,-0.08) +
             pth(0.15+0.20*cos(th), 0.20*sin(th)) +              # landmark ring
             arr(0.40,0.0,0.78,0.0) +
             dot(-0.82,0.55) + dot(-0.82,0.0) + dot(-0.82,-0.55) +
             dot(0.15,0.0,1.4) + dot(0.92,0.0)
  } else if(type=="clipcheck"){ ## physician checks/curates variables
    p <- p + pth(c(-0.60,-0.60,0.60,0.60,-0.60), c(-0.88,0.78,0.78,-0.88,-0.88)) +
             pth(c(-0.20,-0.20,0.20,0.20,-0.20), c(0.78,0.98,0.98,0.78,0.78)) +
             pth(c(-0.46,-0.33,-0.12), c(0.40,0.27,0.52)) + seg(0.02,0.40,0.45,0.40) +
             pth(c(-0.46,-0.33,-0.12), c(0.02,-0.11,0.14)) + seg(0.02,0.02,0.45,0.02) +
             pth(c(-0.46,-0.33,-0.12), c(-0.36,-0.49,-0.24)) + seg(0.02,-0.36,0.45,-0.36)
  } else if(type=="strata"){   ## 3 risk groups, low -> high
    p <- p + seg(-0.88,-0.72,0.88,-0.72) +
             seg(-0.58,-0.72,-0.58,-0.18,2.0) + seg(0.0,-0.72,0.0,0.18,2.0) + seg(0.58,-0.72,0.58,0.56,2.0) +
             dot(-0.58,0.02,1.6) + dot(0.0,0.38,1.6) + dot(0.58,0.76,1.6)
  }
  p
}

icons <- c("docs","text2tab","dag","clipcheck","strata")
for(i in 1:n) p <- add_icon(p, icons[i], left[i]+icon_dx, icon_cy, icon_s)

## ---- header titles (right of icon) + tag + body ----
p <- p +
  annotate("text", x=left+icon_dx*2+0.2, y=(y_hdr+y_top)/2, label=title, colour="white",
           fontface="bold", size=5.4, hjust=0, vjust=0.5, lineheight=.9) +
  annotate("text", x=left+0.7, y=y_hdr-1.5, label=tag, colour=hdr_col,
           fontface="italic", size=4.0, hjust=0, vjust=1) +
  annotate("text", x=left+0.7, y=y_hdr-4.6, label=body, colour="grey15",
           size=5.0, hjust=0, vjust=1, lineheight=1.32)

## ---- connector arrows ----
ay <- 45
for(i in 1:4){
  p <- p + annotate("segment", x=right[i]+0.2, xend=left[i+1]-0.2, y=ay, yend=ay,
                    arrow=arrow(length=unit(0.18,"cm"), type="closed"),
                    linewidth=.9, colour="grey30")
}
p <- p + annotate("text", x=(right[3]+left[4])/2, y=ay+6, label="restrict to\nearly-leak cohort",
                  size=3.8, fontface="italic", colour="grey25", lineheight=.85)

## ---- phase brackets ----
brk <- function(x0,x1,y,txt,col)
  list(annotate("segment", x=x0, xend=x1, y=y, yend=y, colour=col, linewidth=.8),
       annotate("segment", x=x0, xend=x0, y=y, yend=y-1.4, colour=col, linewidth=.8),
       annotate("segment", x=x1, xend=x1, y=y, yend=y-1.4, colour=col, linewidth=.8),
       annotate("text", x=(x0+x1)/2, y=y+1.7, label=txt, colour=col, fontface="bold", size=5.2))
p <- p + brk(left[1], right[3], 78, "Phase 1  ·  whole cohort", blue) +
         brk(left[4], right[5], 78, "Phase 2  ·  early-leak cohort", verm)

p <- p + coord_cartesian(xlim=xlim0, ylim=ylim0, expand=FALSE) + theme_void()
ggsave(file.path(outdir,"fig_pipeline.png"), p, width=fig_w, height=fig_h, dpi=300, bg="white")
cat("DONE fig_pipeline.png\n")

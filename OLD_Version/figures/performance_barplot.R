library("ggplot2")

results_testset1 <- read.delim("~/Desktop/collembola_ai/results_testset1.txt")

ggplot(results_testset1, aes(x=species, y=proportion, fill=prediction)) + 
  geom_bar(position="dodge", stat="identity", colour="black") + 
  theme_test() + 
  theme(legend.title = element_blank(), axis.text.x = element_text(size = 12, angle = 45, hjust = 1)) +
  scale_fill_manual(labels = c("correct", "false positive", "not detected", "wrong"), values=c("#4CB944", "#246EB9", "#F5EE9E", '#F06543')) +
  geom_text(aes(label=proportion), position=position_dodge(width=0.9),  angle = 90, hjust=-0.1)
ggsave("/home/stephan/Desktop/collembola_ai/performance_barplot_testset1.png")

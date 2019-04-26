library(OneR)
library(dplyr)
library(tidyverse)

data_17 <- read.table("data/world-happiness-2017.csv", sep = ",", header = TRUE)
data_16 <- read.table("data/world-happiness-2016.csv", sep = ",", header = TRUE)
data_15 <- read.table("data/world-happiness-2015.csv", sep = ",", header = TRUE)

common_feats <- colnames(data_16)[which(colnames(data_16) %in% colnames(data_15))]

# features and response variable for modeling
feats <- setdiff(common_feats, c("Country", "Happiness.Rank", "Happiness.Score"))
response <- "Happiness.Score"

# combine data from 2015 and 2016

data_15_16 <- rbind(dplyr::select(data_15, one_of(c(feats, response))),
                    dplyr::select(data_16, one_of(c(feats, response))))

data_15_16_17 <- rbind(dplyr::select(data_15, one_of(c(feats, response))),
                       dplyr::select(data_16, one_of(c(feats, response))),
                       dplyr::select(data_17, one_of(c(feats, response))))

# configure multicore
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

library(caret)

set.seed(42)
data_15_16$Happiness.Score.l <- bin(data_15_16$Happiness.Score, nbins = 3, method = "content")
index <- createDataPartition(data_15_16$Happiness.Score.l, p = 0.7, list = FALSE)
train_data <- data_15_16[index, ]
test_data  <- data_15_16[-index, ]

set.seed(42)
model_mlp <- caret::train(Happiness.Score.l ~ .,
                          data = train_data,
                          method = "mlp",
                          trControl = trainControl(method = "repeatedcv", 
                                                   number = 10, 
                                                   repeats = 5, 
                                                   verboseIter = FALSE))



library(lime)

explainer <- lime(train_data, model_mlp, bin_continuous = TRUE, n_bins = 5, n_permutations = 1000)

pred <- data.frame(sample_id = 1:nrow(test_data),
                   predict(model_mlp, test_data, type = "prob"),
                   actual = test_data$Happiness.Score.l)
pred$prediction <- colnames(pred)[3:5][apply(pred[, 3:5], 1, which.max)]
pred$correct <- ifelse(pred$actual == pred$prediction, "correct", "wrong")


library(tidyverse)
pred_cor <- filter(pred, correct == "correct")
pred_wrong <- filter(pred, correct == "wrong")

test_data_cor <- test_data %>%
  mutate(sample_id = 1:nrow(test_data)) %>%
  filter(sample_id %in% pred_cor$sample_id) %>%
  sample_n(size = 3) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-Happiness.Score.l)

test_data_wrong <- test_data %>%
  mutate(sample_id = 1:nrow(test_data)) %>%
  filter(sample_id %in% pred_wrong$sample_id) %>%
  sample_n(size = 3) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-Happiness.Score.l)



explanation_cor <- explain(test_data_cor, explainer, n_labels = 3, n_features = 5)
explanation_wrong <- explain(test_data_wrong, explainer, n_labels = 3, n_features = 5)


plot_features(explanation_cor, ncol = 3)


plot_features(explanation_wrong, ncol = 3)

tibble::glimpse(explanation_cor)

pred %>%
  filter(sample_id == 22)
##   sample_id        low   medium       high actual prediction correct
## 1        22 0.02906327 0.847562 0.07429938 medium     medium correct

train_data %>%
  gather(x, y, Economy..GDP.per.Capita.:Dystopia.Residual) %>%
  ggplot(aes(x = Happiness.Score.l, y = y)) +
  geom_boxplot(alpha = 0.8, color = "grey") + 
  geom_point(data = gather(test_data[22, ], x, y, Economy..GDP.per.Capita.:Dystopia.Residual), color = "red", size = 3) +
  facet_wrap(~ x, scales = "free", ncol = 4)
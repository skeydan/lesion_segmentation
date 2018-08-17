library(keras)
library(neurobase)
library(KernSmooth)
library(reticulate)
library(pracma)

sc <- import("scipy")


# Functions ---------------------------------------------------------------

normalize_image <- function(vol, contrast) {
  temp <- vol[vol != 0]
  q <- quantile(temp, .99)
  temp <- temp[temp <= q]
  bw <- q / 80
  gridsize <- 128 ## scipy rounds up from 80 to 128
  cat("99th quantile is",
      q,
      ", bandwidth =",
      bw,
      ", gridsize =",
      gridsize,
      "\n")
  kde <- bkde(temp, bandwidth = bw, gridsize = gridsize)
  x <- kde$y
  y <- kde$x
  indx <- findpeaks(x)[, 2]
  heights <- x[indx]
  peaks <- y[indx]
  cat(length(peaks), "peaks found.", "\n")
  if (contrast == "T1") {
    peak <- peaks[length(peaks)]
  } else if (contrast %in% c("T2", "PD", "FL")) {
    peak_height <- which.max(heights)
    peak <- peaks[which.max(heights)]
  }
  cat("Peak found at", peak, "for", contrast, "\n")
  peak
}

pad_image <- function(vol, padsize) {
  dim_orig <- dim(vol)
  dim_pad <- dim_orig + 2 * padsize
  padded <- array(0, dim = dim_pad)
  padded[(padsize + 1):(dim_orig[1] + padsize), (padsize + 1):(dim_orig[2] + padsize),
         (padsize + 1):(dim_orig[3] + padsize)] <- vol
  padded
}

get_patches <- function(invol1, invol2, mask, patchsize) {
  dsize <- floor(patchsize / 2)
  indx <- which(mask != 0, arr.ind = TRUE) %>% t()
  num_patches <- ncol(indx)
  cat("Number of patches used: ", num_patches, "\n")
  matsize <- c(num_patches, patchsize[1], patchsize[2], 1)
  randindx <- sample(1:num_patches, num_patches, replace = FALSE)
  newindx <- matrix(0, nrow = 3, ncol = num_patches)
  for (i in 1:num_patches) {
    for (j in 1:3) {
      newindx[j, i] <- indx[j, randindx[i]]
    }
  }
  blurmask <-
    sc$ndimage$filters$gaussian_filter(mask, sigma = c(1, 1, 1))
  blurmask[blurmask < 0.0001] <- 0
  stopifnot(!all(blurmask == 0))
  blurmask <- blurmask * 100
  t1_patches <- array(0, dim = matsize)
  fl_patches <- array(0, dim = matsize)
  mask_patches <- array(0, dim = matsize)
  for (i in 1:num_patches) {
    x <- newindx[1, i]
    y <- newindx[2, i]
    z <- newindx[3, i]
    t1_patches[i, , , 1] <-
      invol1[(x - dsize[1]):(x + dsize[1]), (y - dsize[2]):(y + dsize[2]), z]
    fl_patches[i, , , 1] <-
      invol2[(x - dsize[1]):(x + dsize[1]), (y - dsize[2]):(y + dsize[2]), z]
    mask_patches[i, , , 1] <-
      blurmask[(x - dsize[1]):(x + dsize[1]), (y - dsize[2]):(y + dsize[2]), z]
  }
  list(t1_patches, fl_patches, mask_patches)
}


# Configuration -----------------------------------------------------------

atlas_dir <- "lesion_challenge"
psize <- 35
model_dir <- "saved_models"

patchsize <- c(psize, psize)
padsize <- max(patchsize + 1) / 2

measurements <- c(
  "training01_01",
  "training01_02",
  "training01_03",
  "training01_04",
  "training02_01",
  "training02_02",
  "training02_03",
  "training02_04",
  "training03_01",
  "training03_02",
  "training03_03",
  "training03_04",
  "training03_05",
  "training04_01",
  "training04_02",
  "training04_03",
  "training04_04",
  "training05_01",
  "training05_02",
  "training05_03",
  "training05_04")

# Read data ---------------------------------------------------------------

saved_patches_exist <- TRUE

if(!saved_patches_exist) {
  num_patches <- 0
  mask_files <-  list.files(atlas_dir, pattern = "mask1.nii")
  for (f in mask_files) {
    p <-
      readnii(file.path(atlas_dir, f))
    num_patches <- num_patches + img_data(p) %>% sum()
  }

  cat("Total number of lesion patches =" , num_patches, "\n")

  matsize <- c(num_patches, patchsize[1], patchsize[2], 1)

  t1_patches <- array(0, dim = matsize)
  fl_patches <- array(0, dim = matsize)
  mask_patches <- array(0, dim = matsize)

  count2 <- 1
  count1 <- 1

  for (i in seq_along(measurements)) {
    t1name <- file.path(atlas_dir, paste0(measurements[i], "_mprage_pp.nii"))
    cat("Reading", t1name, "\n")
    t1 <- readnii(t1name) %>% img_data()
    flname <- file.path(atlas_dir, paste0(measurements[i], "_flair_pp.nii"))
    cat("Reading", flname, "\n")
    fl <- readnii(flname) %>% img_data()
    maskname <-
      file.path(atlas_dir, paste0(measurements[i], "_mask1.nii"))
    cat("Reading", maskname, "\n")
    mask <- readnii(maskname) %>% img_data()

    t1 <- t1 / normalize_image(t1, 'T1')
    fl <- fl / normalize_image(fl, 'FL')

    padded_t1 <- pad_image(t1, padsize)
    padded_fl <- pad_image(fl, padsize)
    padded_mask <- pad_image(mask, padsize)

    cat("T1 orig dim: ", dim(t1), "\n")
    cat("T1 padded to dim: ", dim(padded_t1), "\n")

    c(t1_patches_a, fl_patches_a, mask_patches_a) %<-% get_patches(padded_t1, padded_fl, padded_mask, patchsize)

    pdim <- dim(t1_patches_a)
    count2 <- count1 + pdim[1] - 1
    cat("Atlas", i, "indices:", count1, count2, "\n")

    t1_patches[count1:count2, , , ] <- t1_patches_a
    fl_patches[count1:count2, , , ] <- fl_patches_a
    mask_patches[count1:count2, , , ] <- mask_patches_a
    count1 <- count1 + pdim[1]

  }

  cat("Total number of patches collected = ", count2, "\n")
  cat("Size of the input matrix is ", dim(mask_patches), "\n")

  saveRDS(t1_patches, "t1_patches.rds")
  saveRDS(fl_patches, "fl_patches.rds")
  saveRDS(mask_patches, "mask_patches.rds")

} else {

  t1_patches <- readRDS("t1_patches.rds")
  fl_patches <- readRDS("fl_patches.rds")
  mask_patches <- readRDS("mask_patches.rds")

  num_patches <- nrow(t1_patches)

}


# Train-test split --------------------------------------------------------

if(!saved_patches_exist) {
  train_indx <- sample(1:num_patches, num_patches * 0.7)
  saveRDS(train_indx, "train_index.rds")
} else {
  train_indx <- readRDS("train_index.rds")
}

c(t1_train, t1_test) %<-% list(t1_patches[train_indx, , , , drop = FALSE], t1_patches[-train_indx, , ,  , drop = FALSE])
c(fl_train, fl_test) %<-% list(fl_patches[train_indx, , ,  , drop = FALSE], fl_patches[-train_indx, , , , drop = FALSE])
c(mask_train, mask_test) %<-% list(mask_patches[train_indx, , , , drop = FALSE], mask_patches[-train_indx, , ,  , drop = FALSE])



# Model -------------------------------------------------------------------

model_exists <- TRUE
restore_path <- "weights.05-10.58.hdf5"

ds <- 2
num_filters <- 128
kernel_size_1 <- 3
kernel_size_2 <- 5

batch_size <- 1

conv_chain <- function(prev_layer,
                       ds,
                       num_filters,
                       kernel_size_1,
                       kernel_size_2,
                       prefix = NULL) {
  prev_layer %>%
    layer_conv_2d(
      filters = num_filters,
      kernel_size = kernel_size_1,
      activation = "relu",
      padding = "same",
      name = paste0(prefix, "_conv1")
    ) %>%
    layer_conv_2d(
      filters = num_filters / ds,
      kernel_size = kernel_size_2,
      activation = "relu",
      padding = "same",
      name = paste0(prefix, "_conv2")
    ) %>%
    layer_conv_2d(
      filters = num_filters / (ds * 2),
      kernel_size = kernel_size_1,
      activation = "relu",
      padding = "same",
      name = paste0(prefix, "_conv3")
    ) %>%
    layer_conv_2d(
      filters = num_filters / (ds ^ 3),
      kernel_size = kernel_size_2,
      activation = "relu",
      padding = "same",
      name = paste0(prefix, "_conv4")
    ) %>%
    layer_conv_2d(
      filters = num_filters / (ds ^ 4),
      kernel_size = kernel_size_1,
      activation = "relu",
      padding = "same",
      name = paste0(prefix, "_conv5")
    )

}

t1_input <- layer_input(shape = shape(NULL, NULL, 1))
t1 <- t1_input %>%
  conv_chain(
    ds = ds,
    num_filters = num_filters,
    kernel_size_1 = kernel_size_1,
    kernel_size_2 = kernel_size_2,
    prefix = "t1"
  )

fl_input <- layer_input(shape = shape(NULL, NULL, 1))
fl <- fl_input %>%
  conv_chain(
    ds = ds,
    num_filters = num_filters,
    kernel_size_1 = kernel_size_1,
    kernel_size_2 = kernel_size_2,
    prefix = "fl"
  )

concat <- layer_concatenate(list(t1, fl), axis = -1)

combined <- concat %>%
  conv_chain(
    ds = ds,
    num_filters = num_filters,
    kernel_size_1 = kernel_size_1,
    kernel_size_2 = kernel_size_2,
    prefix = "combined"
  ) %>%
  layer_conv_2d(
    filters = 1,
    kernel_size = 3,
    activation = "relu",
    padding = "same",
    name = "conv_final"
  )

model <-
  keras_model(inputs = list(t1_input, fl_input), outputs = combined)

model %>% compile(
  optimizer = optimizer_adam(lr =  0.0001),
  loss = "mean_squared_error",
  metrics = c("mean_squared_error")
)

if(!model_exists) {
  history <- model %>% fit(
    x = list(t1_train, fl_train),
    y = mask_train,
    batch_size = batch_size,
    epochs = 5,
    validation_split = 0.7,
    callbacks = list(callback_model_checkpoint(filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                     callback_early_stopping(patience = 1)))
    saveRDS(history, "history.rds")
} else {
  model <- load_model_hdf5(restore_path)
  history <- readRDS("history.rds")
}

plot(history, metrics = "loss")

par(mfrow = c(6,6))
par(mar = c(0.5, 0.5, 0.5, 0.5),
    xaxs = 'i',
    yaxs = 'i')
for (i in 1:36) {
  preds <- model %>% predict(list(t1_test[i, , , , drop = FALSE],
                         fl_test[i, , , , drop = FALSE]),
                    batch_size = 1)
  true <- mask_test[i, , , , drop = FALSE]
  cor(true, preds) %>% print()
  preds[1, , , ] %>% image(col = grey.colors(n=10),  xaxt = 'n',
                           yaxt = 'n')
  true[1, , , ] %>% image(col = grey.colors(n=10),  xaxt = 'n',
                          yaxt = 'n')
}


model %>% evaluate(x = list(t1_test,
                       fl_test),
                   y = mask_test,
                  batch_size = 1)

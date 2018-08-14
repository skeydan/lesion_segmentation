library(keras)
library(neurobase)
library(KernSmooth)
library(reticulate)
library(pracma)

sc <- import("scipy")



# The data consists of magnetic resonance (MR) images (MRI) divided into two cohorts:
# 1) Training Set;
# 2) Test Set.
# The Training Set consists of five subjects, four of which had four time-points,
# while the fifth subject had five time-points.
# The Test Setincludes fourteen subjects, ten of which had four time-points,
# three had five time-points, and one had six time-points.
# Two consecutive time-points are separated by approximately one year for all subjects.

# Atlas directory containing atlasXX_T1.nii, atlasXX_FL.nii, atlasXX_mask.nii.gz etc.
# XX=1,2,3...
atlas_dir <- "lesion_challenge"
n_atlas <- 1
psize <- 35
model_dir <- "saved_models"

##

patchsize <- c(psize, psize)
padsize <- max(patchsize + 1) / 2
num_patches <- 0

for (i in 1:n_atlas) {
  p <-
    readnii(file.path(atlas_dir, paste0("atlas", i, "_mask.nii.gz")))
  num_patches <- num_patches + img_data(p) %>% sum()
}

cat("Total number of lesion patches =" , num_patches, "\n")

matsize <- c(num_patches, patchsize[1], patchsize[2], 1)


#for i in range(0, numatlas):
# remove
i <- 1
t1name <- file.path(atlas_dir, paste0("atlas", i, "_T1.nii.gz"))
cat("Reading", t1name, "\n")
t1 <- readnii(t1name) %>% img_data()
flname <- file.path(atlas_dir, paste0("atlas", i, "_FL.nii.gz"))
cat("Reading", flname, "\n")
fl <- readnii(flname) %>% img_data()
maskname <- file.path(atlas_dir, paste0("atlas", i, "_mask.nii.gz"))
cat("Reading", maskname, "\n")
mask <- readnii(maskname) %>% img_data()

dim(t1)

normalize_image <- function(vol, contrast) {
  temp <- vol[vol != 0]
  length(temp)
  q <- quantile(temp, .99)
  temp <- temp[temp <= q]
  length(temp)
  bw <- q / 80
  gridsize <- 128 ## scipy rounds up 80 to 128
  cat("99th quantile is",
      q,
      ", bandwidth =",
      bw,
      "gridsize =",
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

t1 <- t1 / normalize_image(t1, 'T1')
fl <- fl / normalize_image(fl, 'FL')

pad_image <- function(vol, padsize) {
  dim_orig <- dim(vol)
  dim_pad <- dim_orig + 2 * padsize
  padded <- array(0, dim = dim_pad)
  padded[(padsize + 1):(dim_orig[1] + padsize), (padsize + 1):(dim_orig[2] + padsize),
         (padsize + 1):(dim_orig[3] + padsize)] <- vol
  padded
}


padded_t1 <- pad_image(t1, padsize)
padded_fl <- pad_image(fl, padsize)
padded_mask <- pad_image(mask, padsize)

cat("T1 padded to dim: ", dim(padded_t1), "\n")

get_patches <- function(invol1, invol2, mask, patchsize) {
  dsize <- floor(patchsize / 2)
  indx <- which(mask != 0, arr.ind = TRUE) %>% t()
  num_patches <- ncol(indx)
  cat("Number of patches used: ", num_patches, "\n")
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

c(t1_patches_a, fl_patches_a, mask_patches_a) %<-% get_patches(padded_t1, padded_fl, padded_mask, patchsize)

cat("Dim of T1 patches:", dim(t1_patches_a), "\n")

pdim <- dim(t1_patches_a)
count2 <- pdim[1]
count1 <- 1

cat("Atlas ", i, "indices:", count1, count2)

t1_patches <- array(0, dim = matsize)
fl_patches <- array(0, dim = matsize)
mask_patches <- array(0, dim = matsize)
t1_patches[count1:count2, , , ] <- t1_patches_a
fl_patches[count1:count2, , , ] <- fl_patches_a
mask_patches[count1:count2, , , ] <- mask_patches_a
count1 < count1 + pdim[1]

cat("Total number of patches collected = ", count2)
cat("Size of the input matrix is ", dim(mask_patches))

##
ds <- 2
num_filters <- 128
kernel_size_1 <- 3
kernel_size_2 <- 5

##
batch_size <- 128


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
    padding = "same"
  )

model <-
  keras_model(inputs = list(t1_input, fl_input), outputs = combined)
model %>% compile(optimizer = optimizer_adam(lr =  0.0001), loss = "mean_squared_error")

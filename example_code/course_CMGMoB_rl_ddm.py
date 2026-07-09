import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyddm

def fill_rddm_pars(pars = {}, defaults = None):
    if defaults is None:
        defaults = {
            "alpha": 0.25,
            "v_intercept": 0,
            "v_scale": 1,
            "a"   : 3,
            "w"   : 0.5,
            "t0"  : 0.25,
            "sv"  : 0,
            "sw"  : 0,
            "st0" : 0
        }
        out = defaults
        out.update(pars)
    return out

def value_update(value, outcome, alpha):
    pred_error = outcome - value
    value = value + alpha * pred_error
    return value

def softmax(values, beta):
    numerator = np.exp(beta * (values - np.max(values)))
    denominator = np.sum(numerator)
    probabilities = numerator / denominator
    return probabilities

def simulate_choice(probabilities):
    n_options = len(probabilities)
    choice = np.random.choice(np.arange(n_options), p=probabilities)
    return choice

# # Helper functions ------------------------------------------------------------

# simulate_task_environment <- function(n_trials, means, sds) {
#   n_options <- length(means)
#   outcomes <- matrix(data = NA_real_, nrow = n_trials, ncol = n_options)
#   for (i in seq_len(n_options)) {
#     outcomes[ , i] <- rnorm(n = n_trials, mean = means[i], sd = sds[i])
#   }
#   colnames(outcomes) <- paste("option", seq_len(n_options), sep = "_")
#   return(as.data.frame(outcomes))
# }

# # DDM-related helpers
# ddm_sample_trial <- function(pars) {
#   sim <- WienR::rWDM(
#     N = 1,
#     a = pars["a"],
#     v = pars["v"],
#     w = pars["w"],
#     t0 = pars["t0"],
#     sv = pars["sv"],
#     sw = pars["sw"],
#     st0 = pars["st0"],
#     response = "both"
#   )
#   choice <- ifelse(sim$response == "lower", 1L, 2L)
#   rt <- sim$q
#   out <- list(choice = choice, rt = rt)
#   return(out)
# }

# ddm_logpdf_trial <- function(rt, choice, pars) {
#   fit <- WienR::dWDM(
#     t = rt,
#     response = choice,
#     a = pars["a"],
#     v = pars["v"],
#     w = pars["w"],
#     t0 = pars["t0"],
#     sv = pars["sv"],
#     sw = pars["sw"],
#     st0 = pars["st0"],
#   )
#   out <- fit$logvalue
#   return(out)
# }

# # RL-DDM helper
# compute_drift <- function(values, pars) {
#   value_diff <- values[2] - values[1]
#   drift <- pars["v_intercept"] + pars["v_scale"] * value_diff
#   return(unname(drift))
# }


# # Core model functions --------------------------------------------------------

# rlddm_simulate <- function(
#     task_environment,
#     pars,
#     initial_values = c(0, 0)
# ) {

#   pars <- fill_rlddm_pars(pars)

#   n_trials <- nrow(task_environment)
#   n_options <- ncol(task_environment)

#   values <- matrix(data = NA_real_, nrow = n_trials, ncol = n_options)
#   drifts <- numeric(length = n_trials)

#   choices <- integer(length = n_trials)
#   RTs <- numeric(length = n_trials)
#   outcomes <- numeric(length = n_trials)
#   prediction_errors <- numeric(length = n_trials)

#   values[1, ] <- initial_values

#   for (t in seq_len(n_trials)) {

#     # compute drift from learned values
#     drifts[t] <- compute_drift(values[t, ], pars)
#     pars_t <- c(pars, v = drifts[t])

#     # DDM-generated choice
#     sim <- ddm_sample_trial(pars_t)
#     choices[t] <- sim$choice
#     RTs[t] <- sim$rt

#     # outcome
#     outcomes[t] <- task_environment[t, choices[t]]

#     # learning
#     prediction_errors[t] <- outcomes[t] - values[t, choices[t]]
#     if (t < n_trials) {
#       values[t + 1, ] <- values[t, ]
#       values[t + 1, choices[t]] <- value_update(
#         value = values[t, choices[t]],
#         outcome = outcomes[t],
#         alpha = pars["alpha"]
#       )

#     }
#   }

#   data <- data.frame(
#     choices = choices,
#     RTs = RTs,
#     outcomes = outcomes
#   )
#   parameters <- as.list(pars)
#   parameters[["initial_values"]] <- initial_values

#   list(
#     data = data,
#     values = values,
#     drifts = drifts,
#     prediction_errors = prediction_errors,
#     parameters = parameters,
#     task_environment = task_environment
#   )
# }



# rlddm_run <- function(
#     data,
#     pars,
#     initial_values = c(0, 0)
# ) {

#   pars <- fill_rlddm_pars(pars)

#   choices <- data$choices
#   RTs <- data$RTs
#   outcomes <- data$outcomes

#   n_trials <- nrow(data)
#   n_options <- length(initial_values)

#   values <- matrix(data = NA_real_, nrow = n_trials, ncol = n_options)
#   drifts <- numeric(length = n_trials)
#   prediction_errors <- numeric(length = n_trials)
#   log_lik <- numeric(length = n_trials)

#   values[1, ] <- initial_values

#   for (t in seq_len(n_trials)) {

#     drifts[t] <- compute_drift(values[t, ], pars)
#     pars_t <- c(pars, v = drifts[t])

#     # log-likelihood of observed choice-RT pair
#     log_lik[t] <- ddm_logpdf_trial(
#       rt = RTs[t],
#       choice = choices[t],
#       pars = pars_t
#     )

#     # learning
#     prediction_errors[t] <- outcomes[t] - values[t, choices[t]]
#     if (t < n_trials) {
#       values[t + 1, ] <- values[t, ]
#       values[t + 1, choices[t]] <- value_update(
#         value = values[t, choices[t]],
#         outcome = outcomes[t],
#         alpha = pars["alpha"]
#       )

#     }
#   }

#   # summed log likelihood of observed choice-RT pairs
#   summed_log_lik <- sum(log_lik)

#   parameters <- as.list(pars)
#   parameters[["initial_values"]] <- initial_values

#   out <- list(
#     data = data,
#     values = values,
#     drifts = drifts,
#     prediction_errors = prediction_errors,
#     log_lik = log_lik,
#     summed_log_lik = summed_log_lik,
#     parameters = as.list(pars)
#   )
#   return(out)
# }



# rlddm_log_lik <- function(data, pars) {
#   fit <- rlddm_run(
#     data = data,
#     pars = pars
#   )
#   out <- fit$summed_log_lik
#   return(out)
# }

# rlddm_priors <- list(
#   alpha = distributional::dist_beta(shape1 = 5, shape2 = 5),
#   v_intercept = distributional::dist_normal(0, 0.5),
#   v_scale = distributional::dist_gamma(shape = 2, rate = 1),
#   a = distributional::dist_gamma(shape = 10, scale = 0.25),
#   w = distributional::dist_beta(shape1 = 20, shape2 = 20),
#   t0 = distributional::dist_gamma(shape = 20, scale = 0.015),
#   sv  = distributional::dist_gamma(shape = 2, scale = 0.25),
#   sw  = distributional::dist_beta(shape1 = 2, shape2 = 8),
#   st0 = distributional::dist_gamma(shape = 2, scale = 0.05)
# )


# rlddm_log_prior <- function(pars, priors = rlddm_priors) {
#   pars <- fill_rlddm_pars(pars)
#   out <- 0
#   for (i in seq_along(priors)) {
#     out <- out + density(priors[[i]], pars[i], log = TRUE)
#   }
#   return(out)
# }


# rlddm_log_posterior <- function(pars, data) {
#   log_prior <- rlddm_log_prior(pars)
#   log_lik <- rlddm_log_lik(data, pars)
#   log_post <- log_prior + log_lik
#   return(log_post)
# }


# # Plotting helpers ------------------------------------------------------------

# plot_values <- function(
#     fit, colours, show_legend = FALSE, option_names = NULL
# ) {
  
#   values_mat <- fit$values
#   n_trials <- nrow(values_mat)
  
#   matplot(
#     values_mat,
#     type = "l",
#     lty = 1,
#     lwd = 2,
#     col = colours,
#     xlab = "Trial",
#     ylab = "Estimated value",
#     main = "Learned values",
#     bty = "n"
#   )
#   axis(side = 1, at = pretty(seq_len(n_trials)))
  
#   if (show_legend) {
#     legend(
#       "bottomright",
#       legend = option_names,
#       col = colours,
#       lty = 1,
#       lwd = 2,
#       bty = "n",
#       cex = 0.9
#     )
#   }
  
#   invisible(NULL)
# }


# plot_drifts <- function(fit) {
  
#   n_trials <- length(fit$drifts)
#   plot(
#     x = seq_len(n_trials),
#     y = fit$drifts,
#     type = "l",
#     lty = 1,
#     lwd = 2,
#     xlab = "Trial",
#     ylab = "Drift rate",
#     main = "Drift scaled by value",
#     bty = "n"
#   )
#   axis(side = 1, at = pretty(seq_len(n_trials)))
  
#   abline(
#     h = fit$parameters$v_intercept,
#     lty = 2,
#     col = "grey"
#   )
  
#   invisible(NULL)
# }


# plot_pred_errors <- function(fit, colours) {
  
#   x <- seq_along(fit$prediction_errors)
#   n_trials <- length(fit$prediction_errors)
#   ylim <- range(fit$prediction_errors)
  
#   plot(
#     NA,
#     xlim = c(1, n_trials),
#     ylim = ylim,
#     xlab = "Trial",
#     ylab = "Prediction error",
#     main = "Prediction errors",
#     bty = "n"
#   )
#   axis(side = 1, at = pretty(seq_len(n_trials)))
  
#   for(i in seq_along(x)) {
#     segments(
#       x0 = x[i],
#       y0 = 0,
#       x1 = x[i],
#       y1 = fit$prediction_errors[i],
#       col = colours[fit$data$choices[i]],
#       lwd = 1
#     )
#   }
#   points(
#     x,
#     fit$prediction_errors,
#     pch = 16,
#     col = colours[fit$data$choices]
#   )
#   abline(
#     h = 0,
#     lty = 2,
#     col = "grey50"
#   )
  
#   invisible(NULL)
# }


# plot_outcomes <- function(
#     fit, colours, shapes,
#     task_environment = NULL, show_legend = FALSE, option_names = NULL
# ) {
  
#   ylim <- range(fit$data$outcomes)
#   n_trials <- nrow(fit$values)
#   n_options <- ncol(fit$values)
  
#   if (is.null(task_environment)) {
#     if ("task_environment" %in% names(fit)) {
#       task_environment <- fit$task_environment
#     }
#   }
  
#   if (!is.null(task_environment)) {
#     ylim <- range(ylim, unlist(task_environment))
#   }
  
#   plot(
#     NA,
#     xlim = c(1, n_trials),
#     ylim = ylim,
#     xlab = "Trial",
#     ylab = "Outcome",
#     main = "Observed outcomes",
#     bty = "n"
#   )
#   axis(side = 1, at = pretty(seq_len(n_trials)))
  
#   if(!is.null(task_environment)) {
#     for(i in seq_len(n_options)) {
#       lines(
#         task_environment[[i]],
#         col = "grey80",
#         lwd = 1
#       )
#     }
#   }
  
#   points(
#     x = seq_len(n_trials),
#     y = fit$data$outcomes,
#     pch = shapes[fit$data$choices],
#     col = colours[fit$data$choices],
#     cex = 1.2
#   )
  
#   if (show_legend) {
#     legend(
#       "bottomright",
#       legend = option_names,
#       pch = shapes,
#       col = colours,
#       cex = .9
#     )
#   }
  
#   invisible(NULL)
# }


# plot_ddm_schematic <- function(
#     pars,
#     colours,
#     n_traces = 10,
#     t_max = 5.0,
#     title = "Diffusion decision process",
#     seed = NULL
# ) {
  
#   if (!is.null(seed)) {
#     set.seed(seed)
#   }
#   if (!is.list(pars)) {
#     pars <- as.list(pars)
#   }
  
#   plot(
#     NA,
#     xlim = c(0, t_max),
#     ylim = c(-0.05 * pars$a, 1.05 * pars$a),
#     xlab = "Time",
#     ylab = "Evidence",
#     main = title,
#     axes = FALSE
#   )
#   axis(side = 1, at = pretty(seq(from = 0, to = t_max)))
#   y_ax_ticks <- pretty(seq(from = 0, to = pars$a))
#   axis(side = 2, at = y_ax_ticks, col.axis = "white")
#   axis(side = 2, at = y_ax_ticks[-c(1, length(y_ax_ticks))])
  
#   abline(h = c(0, pars$a), lty = 2)
#   abline(v = pars$t0, lty = 3)
  
#   text(par("usr")[1], pars$a, "Option 2", pos = 2, xpd = TRUE)
#   text(par("usr")[1], 0, "Option 1", pos = 2, xpd = TRUE)
  
#   # simulate schematic traces directly from the parameters
#   n_steps <- 500
#   dt <- t_max / n_steps
  
#   for (i in seq_len(n_traces)) {
    
#     # trial-specific parameters
#     v_i <- rnorm(n = 1, mean = pars$v, sd = pars$sv)
#     w_i <- ifelse(
#       test = pars$sw > 0,
#       yes = runif(n = 1, min = pars$w - pars$sw/2, max = pars$w + pars$sw/2),
#       no = pars$w
#     )
#     t0_i <- ifelse(
#       test = pars$st0 > 0,
#       yes = runif(n = 1, min = pars$t0, max = pars$t0 + pars$st0),
#       no = pars$t0
#     )
    
#     evidence <- rep(NA_real_, n_steps + 1)
#     time <- seq(0, t_max, length.out = n_steps + 1)
    
#     start_idx <- max(1, which(time >= t0_i)[1])
#     evidence[start_idx] <- w_i * pars$a
    
#     colour <- "grey70"
#     hit <- FALSE
    
#     for (j in (start_idx + 1):(n_steps + 1)) {
      
#       evidence[j] <- evidence[j-1] + v_i * dt +
#         rnorm(n = 1, mean = 0, sd = 1 * sqrt(dt))
      
#       if (evidence[j] >= pars$a) {
#         evidence[j] <- pars$a
#         evidence[(j+1):(n_steps+1)] <- NA_real_
#         colour <- colours[2]
#         hit <- TRUE
#         break
#       }
      
#       if (evidence[j] <= 0) {
#         evidence[j] <- 0
#         evidence[(j+1):(n_steps+1)] <- NA_real_
#         colour <- colours[1]
#         hit <- TRUE
#         break
#       }
      
#     }
    
#     lines(time, evidence, col = colour, lwd = 1)
    
#     if (hit) {
#       last <- max(which(!is.na(evidence)))
#       points(time[last], evidence[last], pch = 16, cex = 1, col = colour)
#     }
    
#   }
  
#   invisible(NULL)
# }



# plot_rt_hists <- function(data, colours) {
  
#   upper <- data$RTs[data$choices == 2L]
#   lower <- data$RTs[data$choices == 1L]
  
#   breaks <- pretty(data$RTs, n = 25)
  
#   hu <- hist(upper, breaks = breaks, plot = FALSE)
#   hl <- hist(lower, breaks = breaks, plot = FALSE)
#   ymax <- max(hu$counts, hl$counts)
  
#   plot(
#     NA,
#     xlim = range(hu$breaks),
#     ylim = c(-ymax, ymax),
#     xlab = "Response time",
#     ylab = "Frequency",
#     main = "Response time distributions",
#     yaxt = "n",
#     bty = "n"
#   )
#   axis(
#     side = 2,
#     at = pretty(c(-ymax, ymax)),
#     labels = abs(pretty(c(-ymax, ymax)))
#   )
#   abline(h = 0)
  
#   rect(
#     xleft = hu$breaks[-length(hu$breaks)],
#     xright = hu$breaks[-1],
#     ybottom = 0,
#     ytop = hu$counts,
#     col = colours[2],
#     border = "white"
#   )
#   rect(
#     xleft = hl$breaks[-length(hl$breaks)],
#     xright = hl$breaks[-1],
#     ybottom = -hl$counts,
#     ytop = 0,
#     col = colours[1],
#     border = "white"
#   )
  
#   legend(
#     "topright",
#     legend = c("Option 2", "Option 1"),
#     fill = rev(colours),
#     bty = "n"
#   )
  
#   invisible(NULL)
# }


# # User-facing Figure dispatcher -----------------------------------------------


# rlddm_plot <- function(
#     fit, n_traces = 10, t_max = 5.0, task_environment = NULL,
#     show_legends = FALSE, seed = NULL
# ) {

#   pars <- fit$parameters
#   data <- fit$data

#   colours <- c("#0072B2", "#D55E00")
#   shapes <- c(0, 1)
#   colours_light <- adjustcolor(colours, alpha.f = 0.5)
#   option_names <- paste("Option", 1:2)

#   old_par <- par(no.readonly = TRUE)
#   on.exit(par(old_par))
  
#   label <- c(
#     sprintf("alpha = %.2f", pars$alpha),
#     sprintf("v_0 = %.2f", pars$v_intercept),
#     sprintf("v_scale = %.2f", pars$v_scale),
#     sprintf("a = %.2f", pars$a),
#     sprintf("w = %.2f", pars$w),
#     sprintf("t0 = %.2f", pars$t0)
#   )
#   if (pars$sv > 0) {
#     label <- c(label, sprintf("sv = %.2f", pars$sv))
#   }
#   if (pars$sw > 0) {
#     label <- c(label, sprintf("sw = %.2f", pars$sw))
#   }
#   if (pars$st0 > 0) {
#     label <- c(label, sprintf("st0 = %.2f", pars$st0))
#   }
  
#   # PLOT 1: RL-style plot
#   par(
#     mfrow = c(2, 2),
#     mar = c(4, 4, 3, 1),
#     oma = c(0, 0, 2, 0)
#   )

#   # learned values 
#   plot_values(fit, colours, show_legends, option_names)
  
#   # choice drift rates scaled by values
#   plot_drifts(fit)
  
#   # prediction errors
#   plot_pred_errors(fit, colours)
  
#   # outcomes (and true rewards)
#   plot_outcomes(
#     fit, colours, shapes, task_environment, show_legends, option_names
#   )
  
#   mtext(
#     "RL-DDM",
#     side = 3,
#     outer = TRUE,
#     line = 0.5,
#     font = 2
#   )
#   mtext(
#     paste(label, collapse = "    "),
#     side = 3,
#     outer = TRUE,
#     line = -0.8,
#     cex = 1
#   )
  
#   # PLOT 2: DDM-style plots for several value regimes
#   par(
#     mfrow = c(3, 2),
#     mar = c(4, 4, 3, 1),
#     oma = c(0, 0, 2, 0)
#   )

#   # define 3 value "regimes"
#   value_diff <- fit$values[, 2] - fit$values[, 1]
#   cuts <- quantile(
#     value_diff,
#     probs = c(0.1, 0.4, 0.6, 0.9),
#     na.rm = TRUE,
#     names = FALSE
#   )
#   cuts <- unique(cuts)
#   if (length(cuts) < 4) {
#     cuts <- seq(
#       from = min(value_diff, na.rm = TRUE),
#       to = max(value_diff, na.rm = TRUE),
#       length.out = 4
#     )
#   }
#   groups <- cut(
#     value_diff,
#     breaks = cuts,
#     include.lowest = TRUE,
#     labels = c("low", "medium", "high")
#   )

#   # plot each regime
#   for (level in levels(groups)) {

#     idx <- which(groups == level)

#     # compute drift rate for representative values
#     values_rep <- apply(
#       X = fit$values[idx, , drop = FALSE],
#       MARGIN = 2,
#       FUN = median
#     )
#     drift_rep <- compute_drift(values_rep, unlist(pars))
#     pars_rep <- c(unlist(pars), v = drift_rep)

#     # left panel: diffusion process schematic
#     plot_ddm_schematic(
#       pars_rep,
#       colours = colours_light,
#       n_traces,
#       t_max,
#       title = sprintf(
#         "value difference = %.2f  (drift = %.2f)",
#         values_rep[2] - values_rep[1],
#         drift_rep
#       ),
#       seed
#     )

#     # right panel: RT distributions
#     plot_rt_hists(
#       data = fit$data[idx, , drop = FALSE],
#       colours = colours_light
#     )

#   }

#   mtext(
#     "RL-DDM",
#     side = 3,
#     outer = TRUE,
#     line = 0.5,
#     font = 2
#   )
#   mtext(
#     paste(label, collapse = "    "),
#     side = 3,
#     outer = TRUE,
#     line = -0.8,
#     cex = 1
#   )

#   invisible(NULL)
# }

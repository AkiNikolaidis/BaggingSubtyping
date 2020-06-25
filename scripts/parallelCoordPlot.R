########################################################
# Function for producing a prallel coordinate plot from
# tabular data with a group or unique row identifier
# ------------------------
# Ian Douglas - March 2020
# 0.0.1
# 

parallelCoordPlot <- function(data, 
                              unique.id.col, 
                              group.col, 
                              selection = everything(), 
                              plot.it = TRUE, 
                              plot_group_means = FALSE, 
                              scale = FALSE,
                              ...)
{
  require(ggplot2)
  require(rlang)
  require(dplyr)
  require(tidyr)
  dat <- dplyr::select(data, !!enquo(unique.id.col), !!enquo(group.col), all_of(selection)) %>%
    mutate_at(-1:-2, ~as.numeric(as.character(.)))
  if (scale) {
    dat[-1:-2] <- dat[-1:-2] %>% mutate_all(~((. - mean(.)) / sd(.)))
  }
  # Gather the columns whose values will be plotted into long format
  reshaped <- tidyr::gather(data = dat, key = key, value = value, 
                            -!!enquo(unique.id.col), -!!enquo(group.col)) %>%
    setNames(., c("id", "group", "dimension", "expression")) %>%
    mutate_at(vars(id, group), factor) 
  
  # The `group` aesthetic is mapped to the unit whose values
  # we want to connect along a single line (unique.id)
  # A larger grouping variable may be used to color points/lines for
  # units who belong to the same group (group.col)
  plt <- ggplot(data = NULL,
                aes(x = dimension, y = expression,
                    group = id,
                    color = group)) +
    geom_point(data = reshaped, 
               alpha = ifelse(plot_group_means,.3,.6), size = 1) +
    geom_line(data = reshaped, 
              alpha = .1, size = 1.5) +
    theme_linedraw() +
    theme(plot.background = element_rect(fill="beige"),
          panel.background = element_rect(fill="white"),
          panel.grid = element_line(color = "black"),
          axis.text.x = element_text(angle = 80, hjust=.9, vjust = .9)) +
    theme(...)
  if (plot_group_means) {
    plt<-plt + 
      geom_point(data = reshaped[-1] %>%
                   group_by(group, dimension) %>%
                   summarize(expression = mean(expression)),
                 aes(x = dimension, y = expression,
                     group = group), 
                 color = "red", alpha = 1, shape = 21, size = 3.5) +
      geom_line(data = reshaped[-1] %>%
                  group_by(group, dimension) %>%
                  summarize(expression = mean(expression)),
                aes(x = dimension, y = expression,
                    group = group,
                    color = group), 
                alpha = 1, size = 2.4)
  }
  plt
  #if (plot.it) {plt} else return(plt)
}


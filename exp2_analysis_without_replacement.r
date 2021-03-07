# Make R see the levels of "avg" (0=single trial, 1=avg3, 2=avg10) as

#df = read.csv('df_final_without_replacement.csv')
df = read.csv('df_final_reduced_unmatched.csv')

# a categorical variable and verify
df$avg = as.factor(df$avg)
is.factor(df$avg)
names(df)[5] = "trial_avg"

# This is how R will encode the categorical variable of average, i.e.:

# 1 2
# 0 0 0
# 1 1 0
# 2 0 1

# Where 0 (single trial) is the intercept
contrasts(df$trial_avg)

# Asterisks imply all main effects and interactions to be tested
# Can write out equivalent full model stating all terms with colon-separator
formula = 'outcome ~ classifier * window * trial_avg'

model = glm(formula = formula, data = df, family = binomial())
summary(model)

anova(model, test="Chisq")


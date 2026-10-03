from common import *
from evidently import Report
from evidently.presets import DataDriftPreset
train_df, test_df = X_train, X_val
report = Report([DataDriftPreset(method="psi")], include_tests="True")
my_eval = report.run(train_df, test_df)
my_eval.save_html("drift.html"); print("drift ok")

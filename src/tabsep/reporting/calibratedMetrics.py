from tabsep.modeling import CVResults
import numpy as np
import scipy.stats as st
import pandas as pd

if __name__ == "__main__":
    saveable_data = list()

    for model_name in ["LR", "Tabnet", "TST"]:
        for prediction_window in [3, 6, 12, 24]:
            collected_data = {"model": model_name, "window": prediction_window}

            cvr = CVResults.load(
                f"cache/{model_name}/sparse_labeled_{prediction_window}_cvresult.pkl"
            )

            print(f"=== {model_name} {prediction_window} ===")

            for metric in ["sensitivity", "specificity", "ppv", "npv"]:
                scores = np.array([getattr(res, metric) for res in cvr.results])
                intervals = st.t.interval(
                    confidence=0.95,
                    df=len(scores) - 1,
                    loc=scores.mean(),
                    scale=st.sem(scores),
                )
                print(f"{metric}: {scores.mean()} {intervals}")

                collected_data[metric] = (
                    f"{scores.mean():.3f} [{intervals[0]:.3f}, {intervals[1]:.3f}]"
                )

            saveable_data.append(collected_data)

    df = pd.DataFrame(data=saveable_data)
    df.to_csv("results/calibratedMetrics.csv", index=False)

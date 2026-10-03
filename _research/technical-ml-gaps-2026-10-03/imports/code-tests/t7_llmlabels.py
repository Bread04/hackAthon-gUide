from sklearn.metrics import cohen_kappa_score, accuracy_score
from collections import Counter
gold=[1,0,1,1,0,0,1,0,1,0]; gold_annot_a=gold; gold_annot_b=[1,0,1,0,0,0,1,0,1,1]
llm_runs=[[1,0,1,1,0,1,1,0,1,0],[1,0,1,1,0,0,1,0,0,0],[1,0,1,1,1,0,1,0,1,0]]
llm_vote = [Counter(v).most_common(1)[0][0] for v in zip(*llm_runs)]
print("LLM vs human  acc", accuracy_score(gold, llm_vote), "kappa", cohen_kappa_score(gold, llm_vote))
print("human vs human kappa (ceiling)", cohen_kappa_score(gold_annot_a, gold_annot_b))
print("run-to-run kappa", cohen_kappa_score(llm_runs[0], llm_runs[1]))
from faker import Faker
fake_person = Faker(); print(fake_person.name(), "|", fake_person.address().replace("\n",", "))

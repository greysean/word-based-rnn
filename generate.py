import model

latest, checkpoint = model.get_latest_model()
one_step = model.OneStep(latest)
print(one_step.generate_sentence())

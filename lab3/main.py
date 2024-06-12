import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

intelligence = ctrl.Antecedent(np.arange(70, 161, 1), 'intelligence')
talent = ctrl.Antecedent(np.arange(0, 11, 1), 'talent')
chance = ctrl.Consequent(np.arange(0, 101, 1), "chance")

intelligence.automf(3)
talent.automf(3)

chance['poor'] = fuzz.trimf(chance.universe, [0, 0, 50])
chance['average'] = fuzz.trimf(chance.universe, [0, 50, 100])
chance['good'] = fuzz.trimf(chance.universe, [50, 100, 100])

intelligence.view()
talent.view()
chance.view()

rule1 = ctrl.Rule(intelligence['poor'] | talent['poor'], chance['poor'])
rule2 = ctrl.Rule(intelligence['average'], chance['average'])
rule3 = ctrl.Rule(intelligence['good'] | talent['good'], chance['good'])

chance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
probabilities = ctrl.ControlSystemSimulation(chance_ctrl)

probabilities.input["intelligence"] = 78
probabilities.input["talent"] = 1

probabilities.compute()

print(probabilities.output['chance'])

chance.view(sim=probabilities)

import matplotlib.pyplot as plt
import numpy as np 

valeur_temps= [ 6.77, 12.4, 27.51,	43.75,	48.23,	54.28,	68.13,	68.13,]
pourcentage_massique_évaporée =[0.026994744, 	0.280668707,	0.49669967,	0.517171931,	0.510348226,	0.514048919,	0.51884655,	0.51228953]
pourcentage_massique_verre = np.array(pourcentage_massique_évaporée) *100
plt.figure()
#lt.plot(valeur_temps, pourcentage_massique_évaporée, 'r')
plt.plot(valeur_temps, pourcentage_massique_verre, 'bo')
plt.xlabel('temps(minutes)')
plt.ylabel('pourcentage massique verre (%)')
plt.show()
plt.close()
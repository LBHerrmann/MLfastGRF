from numpy import *
import matplotlib.pyplot as plt

numRefines0 = 7
numRefines1 = 4

numRefines0_str = str(numRefines0)
numRefines1_str = str(numRefines1)


sqrt_err = loadtxt("results/sqrt_err/sqrt_err_" + numRefines0_str + ".txt")
dimFE = sqrt_err[len(sqrt_err)-1]
sqrt_err = sqrt_err[:-1]
N = range(2,len(sqrt_err)+2)

p1 = polyfit(N[0:6], log(sqrt_err[0:6]),1)
number1 = str(float(p1[0]))
str1 = 'fit: ' + number1[0:6]
str2 = 'dim$(V_h)$=' + (str(dimFE))[0:6]



sqrt_err_2 = loadtxt("results/sqrt_err/sqrt_err_" + numRefines1_str + ".txt")
dimFE_2 = sqrt_err_2[len(sqrt_err_2)-1]
sqrt_err_2 = sqrt_err_2[:-1]


p2 = polyfit(N[0:6], log(sqrt_err_2[0:6]),1)
number2 = str(float(p2[0]))
str3 = 'fit: ' + number2[0:6]
str4 = 'dim$(V_h)$=' + (str(dimFE_2))[0:4]


plt.semilogy(N,sqrt_err,'+-',label=str2 )
plt.semilogy(N[0:6],exp(polyval(p1,N[0:6])),'--', label=str1)

plt.semilogy(N,sqrt_err_2,'x-',label=str4 )
plt.semilogy(N[0:6],exp(polyval(p2,N[0:6])),'--', label=str3)


plt.xlabel(r"$\widetilde{K}$", fontsize=16) 
plt.ylabel(r"relative error",fontsize=16)

plt.grid(True)

plt.legend(loc = "upper right")

plt.tick_params(labelsize=13)


ax = plt.gca()
#ax.invert_xaxis()

plt.tight_layout()

plt.savefig("plot_sqrt_err.pdf")

plt.show()

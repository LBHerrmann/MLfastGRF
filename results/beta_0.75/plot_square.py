from numpy import *
import matplotlib.pyplot as plt





dimFE = loadtxt("results/beta_0.75/square_dimFEspace.txt")
dimFE = dimFE[0:-1]


err_L2_1 = loadtxt("results/beta_0.75/square_L2L2_error_10_10.txt")
ref_val_1 = sqrt(err_L2_1[len(err_L2_1)-1])
err_L2_1 = sqrt(err_L2_1[0:-1])/ref_val_1
p1 = polyfit(log(dimFE[3:-1]), log(err_L2_1[3:-1]),1)
number1 = str(float(p1[0]))
str1 = 'fit: ' + number1[0:6]


plt.loglog(dimFE[:-1],err_L2_1[:-1],'d-',label=r"$\kappa_1^2=10,\kappa_2^2=10$")
plt.loglog(dimFE[3:-1],exp(polyval(p1,log(dimFE[3:-1]))),'--', label=str1)

err_L2_2 = loadtxt("results/beta_0.75/square_L2L2_error_20_200.txt")
ref_val_2 = sqrt(err_L2_2[len(err_L2_2)-1])
err_L2_2 = sqrt(err_L2_2[0:-1])/ref_val_2
p2 = polyfit(log(dimFE[3:-1]), log(err_L2_2[3:-1]),1)
number2 = str(float(p2[0]))
str2 = 'fit: ' + number2[0:6]


plt.loglog(dimFE[:-1],err_L2_2[:-1],'<-',label=r"$\kappa_1^2=20,\kappa_2^2=200$")
plt.loglog(dimFE[3:-1],exp(polyval(p2,log(dimFE[3:-1]))),'--', label=str2)


err_L2_3 = loadtxt("results/beta_0.75/square_L2L2_error_20_2000.txt")
ref_val_3 = sqrt(err_L2_3[len(err_L2_3)-1])
err_L2_3 = sqrt(err_L2_3[0:-1])/ref_val_3
p3 = polyfit(log(dimFE[3:-1]), log(err_L2_3[3:-1]),1)
number3 = str(float(p3[0]))
str3 = 'fit: ' + number3[0:6]


plt.loglog(dimFE[:-1],err_L2_3[:-1],'o-',label=r"$\kappa_1^2=20,\kappa_2^2=2000$")
plt.loglog(dimFE[3:-1],exp(polyval(p3,log(dimFE[3:-1]))),'--', label=str3)

err_L2_4 = loadtxt("results/beta_0.75/square_L2L2_error_2000_2000.txt")
ref_val_4 = sqrt(err_L2_4[len(err_L2_4)-1])
err_L2_4 = sqrt(err_L2_4[0:-1])/ref_val_4
p4 = polyfit(log(dimFE[3:-1]), log(err_L2_4[3:-1]),1)
number4 = str(float(p4[0]))
str4 = 'fit: ' + number4[0:6]


plt.loglog(dimFE[:-1],err_L2_4[:-1],'x-',label=r"$\kappa_1^2=2000=\kappa_2^2$")
plt.loglog(dimFE[3:-1],exp(polyval(p4,log(dimFE[3:-1]))),'--', label=str4)


plt.xlabel(r"dim$(V_h)$", fontsize=16) 
plt.ylabel(r"relative error",fontsize=16)

plt.grid(True)

plt.legend(loc = "lower left")

plt.tick_params(labelsize=13)


ax = plt.gca()
#ax.invert_xaxis()

plt.tight_layout()


plt.savefig("plot_square.pdf")

plt.show()

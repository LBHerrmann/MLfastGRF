from numpy import *
import matplotlib.pyplot as plt



dimFE = loadtxt("results/beta_1.50/square_dimFEspace.txt")
dimFE = dimFE[0:-1]
err_L2 = loadtxt("results/beta_1.50/square_L2L2_error_10_10.txt")
ref_val = sqrt(err_L2[len(err_L2)-1])
err_L2 = sqrt(err_L2[0:-1])/ref_val

p1 = polyfit(log(dimFE[3:]), log(err_L2[3:]),1)
number1 = str(float(p1[0]))
str1 = 'fit: ' + number1[0:6]

dimFE_polygon = loadtxt("results/beta_1.50/square_dimFEspace.txt")
dimFE_polygon = dimFE_polygon[0:-1]
err_L2_polygon = loadtxt("results/beta_1.50/polygon_L2L2_error_10_10.txt")
ref_val_polygon = sqrt(err_L2_polygon[len(err_L2_polygon)-1])
err_L2_polygon = sqrt(err_L2_polygon[0:-1])/ref_val_polygon

p2 = polyfit(log(dimFE_polygon[3:]), log(err_L2_polygon[3:]),1)
number2 = str(float(p2[0]))
str2 = 'fit: ' + number2[0:6]


plt.loglog(dimFE,err_L2,'d-',label=r"square")
plt.loglog(dimFE[3:],exp(polyval(p1,log(dimFE[3:]))),'--', label=str1)\


plt.loglog(dimFE_polygon,err_L2_polygon,'x-',label=r"polygon")
plt.loglog(dimFE_polygon[3:],exp(polyval(p2,log(dimFE_polygon[3:]))),'--', label=str2)


plt.xlabel(r"dim$(V_h)$", fontsize=16) 
plt.ylabel(r"relative error",fontsize=16)

plt.grid(True)

plt.tick_params(labelsize=13)

plt.grid(True)

plt.legend(loc = "upper right")


ax = plt.gca()
#ax.invert_xaxis()

plt.tight_layout()


plt.savefig("plot_square_polygon.pdf")

plt.show()

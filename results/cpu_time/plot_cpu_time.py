from numpy import *
import matplotlib.pyplot as plt

mesh = "square"



dofs = loadtxt("results/cpu_time/" + mesh + "_dimFEspace.txt")
dofs = dofs[0:-1]

time = loadtxt("results/cpu_time/" + mesh + "_cpu_time.txt")

time = [x/1000000.0 for x in time]


p1 = polyfit(log(dofs[3:]), log(time[3:]),1)

p2 = polyfit(log(dofs[3:]), log( ((time[0]/dofs[0])*dofs[3:] * log(dofs[3:])**2.0)  ),1)


number1 = str(float(p1[0]))
number2 = str(float(p2[0]))


str1 = 'fit: ' + number1[0:6]
str2 = 'fit: ' + number2[0:6]




plt.loglog(dofs,time,'d-',label=r"GRF sampling")
plt.loglog(dofs[3:],exp(polyval(p1,log(dofs[3:]))),'--', label=str1)

plt.loglog(dofs,((time[0]/dofs[0])*dofs * log(dofs)**2.0),'o-', label="$C\, N\, \log^{2.0}(N)$")
plt.loglog(dofs[3:],exp(polyval(p2,log(dofs[3:]))),'--', label=str2)

plt.xlabel(r"dim($V_h$)",fontsize=16) 
plt.ylabel(r"CPU time in seconds",fontsize=16)

plt.grid(True)

plt.legend(loc = "upper left")

plt.tick_params(labelsize=13)


plt.tight_layout()

plt.savefig("plot_cpu_time.pdf")

plt.show()

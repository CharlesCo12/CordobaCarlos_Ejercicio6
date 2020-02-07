import numpy as np
import matplotlib.pylab as plt
x = np.array([4.6, 6.0, 2.0, 5.8])
sigma = np.array([2.0, 1.5, 5.0, 1.0])
def L(mu):
    return (np.sum(-0.5*(x-mu)**2/(sigma**2)))
mu=np.linspace(0,10,100)
Lmu=[]
for i in mu:
    Lmu.append(L(i))
maximo=np.where(Lmu==np.max(Lmu))
Mu_max=mu[maximo[0][0]]

#Segunda Derivada
def second_derivate(x,h):
    a=L(x+h)-2*L(x)+L(x-h)
    return a/(h**2)
Delta_Mu=1/np.sqrt(-second_derivate(Mu_max,0.001))
proba=np.exp(Lmu)
normalizacion=np.trapz(proba,x=mu)
proba_new=proba/normalizacion
def aprox(h):
    A=1.0/(Delta_Mu*(np.sqrt(2*np.pi)))
    B=np.exp(-(h-Mu_max)**2/(2*(Delta_Mu**2)))
    return A*B

#Graficar
plt.plot(mu,aprox(mu),'--',label='Gaussian Aproximation')
plt.plot(mu,proba_new,label='Data')
plt.xlabel('$\mu$')
plt.ylabel(r'P($\mu|{x_{k}},{\sigma_k}$)')
plt.title('$\mu_0$ = {:.2f} {} {:.2f}'.format(Mu_max,'$\pm$',Delta_Mu))
plt.legend(loc=0.0)
plt.savefig('mean.png')
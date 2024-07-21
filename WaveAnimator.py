from matplotlib import pyplot as plt 
import matplotlib as mpl
import numpy as np 
from matplotlib.animation import FuncAnimation, FFMpegWriter

class WaveAnimator:
  def __init__(self, x, t, ylim=(-3,3)):

    self.fig = plt.figure()
    xLow = xGrid[0]
    xHigh = xGrid[len(xGrid)-1]

    self.axis = plt.axes(xlim=(xLow, xHigh), ylim=ylim)
    self.pLine, self.qLine = self.axis.plot([], [], 'b-', [], [], 'k-')

    self.uData = []

  def run(self, uData):

    self.uData = uData

    def animate(i):
      pData_i = self.uData[i][0]
      qData_i = self.uData[i][1]
      self.pLine.set_ydata(pData_i)
      self.qLine.set_ydata(qData_i)

      return self.pLine, self.qLine
    

    ani = FuncAnimation(self.fig, animate, frames=len(self.uData),
                          blit=True)
    
    plt.show()
    

    
if __name__=='__main__':
  xGrid = np.linspace(0.0, 1.0, 101)

  tMax = 10.0
  tGrid = np.linspace(0.0, tMax, 101)
  
  uData = []
  times = []
  for t in tGrid:
    times.append(t)
    p = np.cos(2*np.pi*(t-4.0*xGrid))
    q = np.sin(2*np.pi*(t-4.0*xGrid))

    uData.append((p,q))

  xLow = xGrid[0]
  xHigh = xGrid[len(xGrid)-1]

  fig = plt.figure()

  axis = plt.axes(xlim=(xLow, xHigh), ylim=(-1.5,1.5))
  #w = WaveAnimator(xGrid)

  pLine, = axis.plot(xGrid, uData[0][0], 'b-')
  qLine, = axis.plot(xGrid, uData[0][1], 'k-')

  def animate(i):
    axis.set_title('time={:.4f}'.format(times[i]))
    pData_i = uData[i][0]
    qData_i = uData[i][1]
    pLine.set_ydata(pData_i)
    qLine.set_ydata(qData_i)

    return pLine, qLine

  #mpl.use('Agg')

  ani = FuncAnimation(fig, animate, frames=len(uData),
                          blit=False)  
  
  writervideo = FFMpegWriter(fps=10) 
  
  plt.show()

  #ani.save('movie.mp4', writer=writervideo)
  #w.run(uData)







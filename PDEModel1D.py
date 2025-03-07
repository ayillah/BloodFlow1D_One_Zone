from abc import ABC, abstractmethod # ABC = "Abstract base class"

class PDEModel1D(ABC):
  def __init__(self, numVars): 
    self.numVars = numVars


  def numVars(self):
    return self.numVars


  @abstractmethod
  def speed(self):
    '''Return the speed at position x'''
    pass


  @abstractmethod
  def jacobian(self, x):
    '''Return the Jacobian at position x'''
    pass

  @abstractmethod
  def jacEigensystem(self, x):
    '''
    Return (ews, evs) at position x. The ews are returned
    as a 1D array [lam_1, lam_2, ..., lam_numVars]. The evs
    are returned as columns in a matrix V=[v_1,v_2,...,v_numVars]. 

    This is required for certain types of BCs.
    '''
    pass

  @abstractmethod
  def jacEigenvalues(self, x):
    '''
    Return the eigenvalues of the Jacobian at position x. 
    The ews are returned
    as a 1D array [lam_1, lam_2, ..., lam_numVars]. 
    '''
    pass


  @abstractmethod
  def applyLeftBC(self, x, t, dx, dt, u_prev):
    '''
    Do whatever is needed to obtain u_cur at the left boundary point.
    
    Arguments are:
      *) x -- location of left boundary point
      *) t -- current time (i.e., time at end of step)
      *) dx -- grid spacing dx = x_1 - x_0
      *) dt -- timestep size
      *) u_prev -- solution at previous timestep 
    '''
    pass

  
  @abstractmethod
  def applyRightBC(self, x, t, dx, dt, u_prev):
    '''
    Do whatever is needed to obtain u_cur at the right boundary point.
    
    Arguments are:
      *) x -- location of right boundary point
      *) t -- current time (i.e., time at end of step)
      *) dx -- grid spacing dx = x_{nx} - x_{nx-1}
      *) dt -- timestep size
      *) u_prev -- solution at previous timestep 
    '''
    pass

  
  

  

  

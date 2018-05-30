import sys, os, time
from stdlib import math
from cctbx.array_family import flex
import iotbx.phil
from iotbx import pdb
from sastbx.data_reduction import saxs_read_write
from mmtbx.monomer_library import server, pdb_interpretation
from sastbx import intensity
from sastbx.intensity import sas_library
from sastbx.intensity import znk_model
from interface import get_input
import numpy as np

global f
master_params =  iotbx.phil.parse("""\
sas_I{
  method = *she debye zernike
  .type=choice
  .help="Type of theoretical model: debye or she (default)"

  structure = None
  .type=path
  .help="Input structure"

  experimental_data = None
  .type=path
  .help="Experimental data to be compared to model"

  data_reduct=True
  .type=bool
  .help="Experimental data interpolation to speed intensity calculation"

  pdblist = None
  .type=path
  .help="The file that contains PDB filenames for the same protein"

  q_start = 0
  .type = float
  .help = "Start of Q array"

  q_stop = 0.2
  .type = float
  .help = "End of Q aray"

  n_step = 21
  .type=int
  .help="Bin size"

  znk_nmax = 20
  .type=int
  .help="maximum order of zernike polynomials"

  output = "output.iq"
  .type=path
  .help=str

  internals
  .help="Some core parameters. Most likely no need to change this"
  {
    rho=0.334
    .type=float
    .help="solvent electron density"
    delta=3.0
    .type=float
    .help="hydration layer thickness, default=3.0 angstrom"
    drho=0.03
    .type=float
    .help="hydration layer electron density difference"
    max_L=15
    .type=int
    .help="To be added"
    max_i=17
    .type=int
    .help="To be added"
    f_step=0.8
    .type=float
    .help="Integration step when integration Bessel function"
    integration_q_step=0.01
    .type=float
    .help="Step size in q-dependent integration lookup table"
    solvent_radius_scale=0.91
    .type=float
    .help="Determines how much solvent is displaced: Vsolvent = Vvdw*(solvent_radius_scale)^3"
    protein_radius_scale=1.2
    .type=float
    .help="To de added"
    use_adp=False
    .type=bool
    .help="Determines if atomic B values (adp's) are used"
    implicit_hydrogens=True
    .type=bool
    .help="When True, vdw radii are changed and atomic form factors will be adapted to simulate the presence of hydrogen atoms."
    solvent_scale=False
    .help="Determines if a solvent form factor will be refined when experimental data is present"
    .type=bool
  }


}

""")

def write_json(filename, x,y1, y2=None):
  out=open(filename, 'w')

  head = '{"elements":['
  ele1 = '{"type": "line", "dot-style":{"type": "dot", "dot-size": 3, "colour":"#DFC329"}, "width":3, "colour": "DFC329", "text":"model", "font-size":10,'
  print>>out, head, ele1, '"values":',
  y1 = flex.log(flex.abs(y1)+1e-16 )
  y_min=flex.min(y1)-1.0
  y_max=flex.max(y1)+1.0
  print>>out, '[' + ', '.join('%5.6f' % v for v in y1) + ']',
#  for xx, yy in zip( x[:-2], y1[:-2]):
    #print>>out,'{"x":%f, "y":%f},'%(xx,yy),
#  print>>out,'{"x":%f, "y":%f}'%(x[-1],y1[-1]),
  print>>out, '}',  # end of y1
  if(y2 is not None):
    ele2 = ',{"type": "line", "dot-style":{"type": "dot", "dot-size": 1, "colour":"#111111"}, "width":1, "colour": "111111", "text":"expt", "font-size":10,'
    print>>out, ele2, '"values":',
    y2 = flex.log(flex.abs(y2)+1e-16)
    y_min=min(flex.min(y2)-1.0, y_min)
    y_max=max(flex.max(y2)+1.0, y_max)
    print>>out, '[' + ', '.join('%5.6f' % v for v in y2) + ']',
    print>>out, '}', # end of y2
  print>>out, ']',  # end of elements
  ### now setup axis ###
  print>>out,',"y_axis":{"min":%f, "max":%f}'%(y_min, y_max),
  steps = int(0.05/(x[1]-x[0]))
  x_labels = '["'
  for xx in x:
    x_labels = x_labels + str(xx) + '","'
  x_labels=x_labels[0:-2]+']'
  print>>out,',"x_axis":{"min":%d, "max":%d, "steps":%d, "labels":{"labels":%s,"steps":%d}}'%(0,x.size(),steps,x_labels, steps),
  print>>out,'}' ##end of the file
  out.close()

def linear_fit(x,y,s): # Standard least square fitting
  var = s*s
  sum_x2 = flex.sum( x*x/var )
  #sum_y2 = flex.sum( y*y/var )
  sum_xy = flex.sum( x*y/var )
  sum_x  = flex.sum( x / var )
  sum_y  = flex.sum( y / var )
  N = x.size()
  sum_inv_var = flex.sum(1.0/var)
  det = sum_inv_var * sum_x2 - sum_x * sum_x
  scale = (sum_inv_var * sum_xy - sum_x*sum_y ) / det
  offset = (sum_x2*sum_y - sum_x * sum_xy) /det

  return scale, offset

def linear_fit2(x,y,s): # now constant subtraction, original crysol approach
  var = s*s   # x Ical; y Iexp; s sigma
  sum_x2 = flex.sum( x*x/var )
  sum_xy = flex.sum( x*y/var )
  N = x.size()
  scale = sum_xy/sum_x2
  offset = 0
  return scale, offset

# copied from zernike_model/search_pdb
def reduce_raw_data(raw_data, qmax, bandwidth, level=0.05, q_background=None, outfile=''): 
    log2 = sys.stdout
    
    with open(outfile,"a") as log:
      print >>log, " ====  Data reduction ==== "
    
      print >>log, "  Preprocessing of data increases efficiency of shape retrieval procedure.\n"
   
      print >>log,"   -  Interpolation stepsize                           :  %4.3e"%bandwidth
      print >>log,"   -  Uniform density criteria:  level is set to       :  %4.3e"%level
      print >>log,"                                 maximum q to consider :  %4.3e"%qmax

    print >>log2, " ====  Data reduction ==== "
    
    print >>log2, "  Preprocessing of data increases efficiency of shape retrieval procedure.\n"
 
    print >>log2,"   -  Interpolation stepsize                           :  %4.3e"%bandwidth
    print >>log2,"   -  Uniform density criteria:  level is set to       :  %4.3e"%level
    print >>log2,"                                 maximum q to consider :  %4.3e"%qmax

    qmin_indx = flex.max_index( raw_data.i )
    qmin = raw_data.q[qmin_indx]
    if qmax > raw_data.q[-1]:
      qmax = raw_data.q[-1]
    with open(outfile,"a") as log:
      print >>log, "      Resulting q range to use in  search:   q start   :  %4.3e"%qmin
      print >>log, "                                             q stop    :  %4.3e"%qmax
    
    print >>log2, "      Resulting q range to use in  search:   q start   :  %4.3e"%qmin
    print >>log2, "                                             q stop    :  %4.3e"%qmax
    raw_q = raw_data.q[qmin_indx:]
    raw_i = raw_data.i[qmin_indx:]
    raw_s = raw_data.s[qmin_indx:]
    ### Take care of the background (set zero at very high q) ###
    if( q_background is not None):
      cutoff = flex.bool( raw_q > q_background )
      q_bk_indx = flex.last_index( cutoff, False )
      if( q_bk_indx < raw_q.size() ):
        bkgrd = flex.mean( raw_i[q_bk_indx:] )
        with open(f,"a") as log:
          print >>log,  "Background correction: I=I-background, where background=", bkgrd
        print >>log2,  "Background correction: I=I-background, where background=", bkgrd
        raw_i = flex.abs( raw_i -bkgrd )

    q = flex.double( range(int( (qmax-qmin)/bandwidth )+1) )*bandwidth + qmin
    raw_data.i = flex.linear_interpolation( raw_q, raw_i, q )
    raw_data.s = flex.linear_interpolation( raw_q, raw_s, q )
    raw_data.q = q

    return raw_data



class solvent_parameter_optimisation(object):
  def __init__(self, she_object, observed_data):
    # we'll only optimize the scale factor and form factor of excluded solvent
    self.rm = 1.62
    self.rho=0.334
    self.drho=0.03
    self.obs = observed_data
    self.she_object = she_object
    self.rm_fluct_scale = -(4.0*math.pi/3.0)**1.5*math.pi*flex.pow(self.obs.q,2.0)*self.rm**2.0
    ### setup the scan range ###
    self.default_a = 1.0
    self.a_range=flex.double(range(-10,11))/50.0+self.default_a
    self.drho_range = (flex.double(range(-10,21))/10.0+1.0)*self.drho

    self.scan()


  def scan(self):
    self.best_score, self.best_i_calc, self.best_scale, self.best_offset=self.target(self.drho, self.default_a)
    self.best_drho = self.drho
    self.best_a = self.default_a
    for drho in self.drho_range:
      for a in self.a_range:
        score, i_calc, s, off = self.target(drho, a)
        if(score < self.best_score ):
          self.best_score = score
          self.best_drho = drho
          self.best_a = a
          self.best_i_calc = i_calc.deep_copy()
          self.best_scale = s
          self.best_offset = off
    return

  def compute_scale_array(self, a):
    scales = a**3.0 * flex.exp(self.rm_fluct_scale*(a**2.0 - 1))
    return scales


  def get_scaled_data(self):
    return self.best_i_calc

  def get_scales(self):
    return self.best_scale, self.best_offset, self.best_drho, self.best_a*self.rm

  def target(self, drho, a):
    self.she_object.update_solvent_params(self.rho,drho)
    this_scale = self.compute_scale_array( a )
    i_calc = self.she_object.Iscale(this_scale)

    s, off = linear_fit( i_calc, self.obs.i, self.obs.s )
    i_calc = s*i_calc + off
    result = flex.sum(flex.pow2( (i_calc-self.obs.i)/self.obs.s ) )
    return result, i_calc, s, off





def write_she_data( q, i,a,b,c, file_name ):
  f = open(file_name,'w')
  print >>f, '#', 'q\t', 'I_total\t\t', 'I_A\t\t', 'I_C\t\t', 'I_B'
  for qq, i0,ia,ic,ib in zip(q,i,a,c,b):
    print >> f, qq, '\t',i0,'\t',ia,'\t',ic,'\t',ib
  f.close()
  return

def write_debye_data( q, i, file_name ):
  f = open(file_name,'w')
  print >>f, '#', 'q\tI_total'
  for qq, i0 in zip(q,i):
    print >> f, qq, '\t',i0
  f.close()
  return


#def run(args, log=sys.stdout):
def run(args):
  global f
  f = os.path.join(os.path.split(sys.path[0])[0],"she.txt")
  with open(f,"w") as tempf:
    tempf.truncate()
  
  #check if we have experimental data
  t1=time.time()
  exp_data = None
  q_values = None
  var = None


  with open(f,"a") as tempf:
    params = get_input( args, master_params, "sas_I", banner, print_help,tempf)
  
  
  if (params is None):
    exit()

  if params.sas_I.experimental_data is not None:
    exp_data = saxs_read_write.read_standard_ascii_qis(params.sas_I.experimental_data)
    #exp_data.s = flex.sqrt( exp_data.i )
    if params.sas_I.data_reduct:
      qmax = exp_data.q[-1]
      bandwidth = 0.5/(params.sas_I.n_step-1.0)
      exp_data=reduce_raw_data( exp_data, qmax, bandwidth,outfile=f )
    q_values = exp_data.q
    var = flex.pow(exp_data.s,2.0)

  if q_values is None:
    q_values = params.sas_I.q_start +  \
              (params.sas_I.q_stop-params.sas_I.q_start
              )*flex.double( range(params.sas_I.n_step) )/(
                params.sas_I.n_step-1)
  # read in pdb file
  pdbi = pdb.hierarchy.input(file_name=params.sas_I.structure)
  #atoms = pdbi.hierarchy.atoms()
  atoms = pdbi.hierarchy.models()[0].atoms()
  # predefine some arrays we will need
  dummy_atom_types = flex.std_string()
  radius= flex.double()
  b_values = flex.double()
  occs = flex.double()
  xyz = flex.vec3_double()
  # keep track of the atom types we have encountered
  dummy_at_collection = []
  for atom in atoms:
   #if(not atom.hetero):   #### temporarily added
    b_values.append( atom.b )
    occs.append( atom.occ )
    xyz.append( atom.xyz )

  # Hydrogen controls whether H is treated explicitly or implicitly
  Hydrogen = not params.sas_I.internals.implicit_hydrogens

### Using Zernike Expansion to Calculate Intensity ###
  '''
  if(params.sas_I.method == 'zernike'):
    znk_nmax=params.sas_I.znk_nmax
    absolute_Io = znk_model.calc_abs_Io( atoms, Hydrogen)
    if( absolute_Io == 0.0): ## in case pdb hierarchy parse did not work out correctly
      absolute_Io = sas_library.calc_abs_Io_from_pdb( params.sas_I.structure, Hydrogen )
    if(Hydrogen):
      density = znk_model.get_density( atoms ) ## Get number of electrons as density
    else:
      density = znk_model.get_density( atoms ) + 1  ## add one H-atom to each heavy atom as a correction
    znk_engine = znk_model.xyz2znk(xyz,absolute_Io,znk_nmax, density=density)
    calc_i, calc_i_vac, calc_i_sol, calc_i_layer=znk_engine.calc_intensity(q_values)
    if(params.sas_I.experimental_data is not None):
      if params.sas_I.internals.solvent_scale:
        znk_engine.optimize_solvent(exp_data)
        calc_i = znk_engine.best_i_calc
      else:  #quick scaling
        scale, offset = linear_fit( calc_i, exp_data.i, exp_data.s )
        calc_i = calc_i*scale + offset

      CHI2 = flex.mean(flex.pow((calc_i-exp_data.i)/exp_data.s,2.0))
      CHI=math.sqrt(CHI2)
      with open(f,"a") as log:
        print >>log, "fitting to experimental curve, chi = %5.4e"%CHI

      print  "fitting to experimental curve, chi = %5.4e"%CHI
     
      write_debye_data(q_values, calc_i, params.sas_I.output+".fit")
      write_json(params.sas_I.output+"data.json", q_values, calc_i, y2=exp_data.i)
    else: ## scaled to the absolute I(0)
      write_she_data(q_values, calc_i, calc_i_vac, calc_i_layer, calc_i_sol, params.sas_I.output)
      write_json(params.sas_I.output+"data.json", q_values, calc_i)

    with open(f,"a") as log:
      print >>log,  znk_engine.summary()
      print >>log, "Done! total time used: %5.4e (seconds)"%(time.time()-t1)   

    print   znk_engine.summary()
    print  "Done! total time used: %5.4e (seconds)"%(time.time()-t1) 
    return
### End of Zernike Model ###

  '''

  dummy_ats= sas_library.read_dummy_type(file_name=params.sas_I.structure)
  for at in dummy_ats:
    if at not in dummy_at_collection:
      dummy_at_collection.append( at )


  radius_dict={}
  ener_lib=server.ener_lib()
  for dummy in dummy_at_collection:
    if(Hydrogen):
      radius_dict[dummy]=ener_lib.lib_atom[dummy].vdw_radius
    else:
      if ener_lib.lib_atom[dummy].vdwh_radius is not None:
        radius_dict[dummy]=ener_lib.lib_atom[dummy].vdwh_radius
      else:
        radius_dict[dummy]=ener_lib.lib_atom[dummy].vdw_radius

    if(radius_dict[dummy] is None):
      with open(f,"a") as log:
        print >> log, "****************** WARNING WARNING  *******************"
        print >> log, "Did not find atom type: ", dummy, "default value 1.58 A was used"
        print >> log, "*******************************************************"

      print  "****************** WARNING WARNING  *******************"
      print  "Did not find atom type: ", dummy, "default value 1.58 A was used"
      print  "*******************************************************"
      radius_dict[dummy]=1.58

  for at in dummy_ats:
    dummy_atom_types.append( at)
    radius.append(radius_dict[at])

  Scaling_factors=sas_library.load_scaling_factor()


  #------------------
  #
  B_factor_on=params.sas_I.internals.use_adp
  max_i = params.sas_I.internals.max_i
  max_L = params.sas_I.internals.max_L
  f_step= params.sas_I.internals.f_step
  q_step= params.sas_I.internals.integration_q_step
  solvent_radius_scale=params.sas_I.internals.solvent_radius_scale
  protein_radius_scale=params.sas_I.internals.protein_radius_scale
  rho=params.sas_I.internals.rho
  drho=params.sas_I.internals.drho
  delta=params.sas_I.internals.delta
  #------------------
  scat_lib_dummy =  sas_library.build_scattering_library( dummy_at_collection,
                                              q_values,
                                              radius_dict,
                                              solvent_radius_scale,
                                              Hydrogen,
                                              Scaling_factors)

  new_indx =flex.int()
  new_coord = flex.vec3_double()

  model=intensity.model(xyz,
                        radius*protein_radius_scale,
                        b_values,
                        occs,
                        dummy_ats,
                        scat_lib_dummy,
                        B_factor_on)
  t2=time.time()
  


  if(params.sas_I.method == 'she'):
    max_z_eps=0.02
    max_z=model.get_max_radius()*(q_values[-1]+max_z_eps) + max_z_eps
    engine = intensity.she_engine( model, scat_lib_dummy,max_i,max_L,f_step, q_step,max_z, delta,rho,drho )
    engine.update_solvent_params(rho,drho)
    i = engine.I()
    a = engine.get_IA()
    b = engine.get_IB()
    c = engine.get_IC()

    attri = engine.Area_Volume()
    with open(f,"a") as log:
      print >> log, "Inner surface Area of the Envelop is (A^2.0): ", attri[0];
      print >> log, "Inner Volume of the Envelop is       (A^3.0): ", attri[1];
      print >> log, "Volume of the Envelop shell is       (A^3.0): ", attri[2];
    
    return np.array(i)
    
    '''
    if params.sas_I.output is not None:
       write_she_data( q_values, i,a,b,c, params.sas_I.output )
       write_json(params.sas_I.output+"data.json", q_values, i)

    
    if params.sas_I.pdblist is not None:
      pdblist=params.sas_I.pdblist
      if(os.path.isfile(pdblist)):
        list= open(pdblist,'r')
        for line in list:
          filename=line.split('\n')[0]
          pdbi = pdb.hierarchy.input(file_name=filename)
          t21 = time.time()
          atoms = pdbi.hierarchy.atoms()
          new_coord.clear()
          new_indx.clear()
          i=0
          for atom in atoms:
            new_coord.append( atom.xyz )
            new_indx.append(i)
            i=i+1

          engine.update_coord(new_coord,new_indx)
          i = engine.I()
          a = engine.get_IA()
          b = engine.get_IB()
          c = engine.get_IC()
          attri = engine.Area_Volume()
          with open(f,"a") as log:
            print >> log, "Inner surface Area of the Envelop is (A^2.0): ", attri[0]
            print >> log, "Inner Volume of the Envelop is       (A^3.0): ", attri[1]
            print >> log, "Volume of the Envelop shell is       (A^3.0): ", attri[2]

          print  "Inner surface Area of the Envelop is (A^2.0): ", attri[0]
          print  "Inner Volume of the Envelop is       (A^3.0): ", attri[1]
          print  "Volume of the Envelop shell is       (A^3.0): ", attri[2]
          write_she_data( q_values, i,a,b,c, filename+'.int' )
          with open(f,"a") as log:
            print >> log, '\nfininshed pdb ', filename, 'at: ',time.ctime(t21),'\n'
          print  '\nfininshed pdb ', filename, 'at: ',time.ctime(t21),'\n'

  #  attri = engine.Area_Volume2()
  #  print "Inner surface Area of the Envelop is (A^2.0): ", attri[0];

  elif(params.sas_I.method == 'debye'):
    engine = intensity.debye_engine (model, scat_lib_dummy)
    i = engine.I()
    if params.sas_I.output is not None:
       write_debye_data(q_values, i, params.sas_I.output)
       write_json(params.sas_I.output+"data.json", q_values, i)

  if(params.sas_I.experimental_data is not None):
    if params.sas_I.internals.solvent_scale:
      # more thorough scaling
      solvent_optim = solvent_parameter_optimisation(she_object=engine,
                                                     observed_data=exp_data )

      scale, offset, drho, a = solvent_optim.get_scales()
      i = solvent_optim.get_scaled_data()
    else:
      #quick scaling
      scale, offset = linear_fit( i, exp_data.i, exp_data.s )
      i = scale*i+offset

    with open(f,"a") as log:
      print >>log,  "Scaled calculated data against experimental data"
      print >>log, "Scale factor : %5.4e"%scale
      print >>log,"Offset       : %5.4e"%offset
  
    print   "Scaled calculated data against experimental data"
    print  "Scale factor : %5.4e"%scale
    print "Offset       : %5.4e"%offset

    if  params.sas_I.internals.solvent_scale:
      with open(f,"a") as log:
        print >> log, "  Solvent average R ra   : ", a
        print >> log, "  Solvation Contrast drho: ", drho
      
      print  "  Solvent average R ra   : ", a
      print  "  Solvation Contrast drho: ", drho
    print
    write_debye_data(q_values, i, params.sas_I.output+".fit")
    write_json(params.sas_I.output+"data.json", q_values, i, y2=exp_data.i)
    CHI2 = flex.mean(flex.pow((i-exp_data.i)/exp_data.s,2.0))
    CHI=math.sqrt(CHI2)
    with open(f,"a") as log:
      print >>log, "fitting to experimental curve, chi = %5.4e"%CHI
    print  "fitting to experimental curve, chi = %5.4e"%CHI
  
  t3=time.time()
  
  with open(f,"a") as log:
    print >> log, "Done! total time used: %5.4e (seconds)"%(t3-t1)
    print >>log, 'start running at:                ',time.ctime(t1)
    print >>log, 'finished PDB file processing at: ',time.ctime(t2)
    print >>log, 'got all desired I(q) at :        ',time.ctime(t3)

  print  "Done! total time used: %5.4e (seconds)"%(t3-t1)
  print  'start running at:                ',time.ctime(t1)
  print  'finished PDB file processing at: ',time.ctime(t2)
  print  'got all desired I(q) at :        ',time.ctime(t3)
  with open(f,"a") as log:
    log.write("__END__")
  '''
banner = """
============================================================================
                        Debye/SHE/Zernike Model
   Model data from a Debye or Spherical Harmonics/Zernike Expansion engine
============================================================================

"""

def print_help(log):
  print "\nUsuage: \n"
  print "  sastbx.she model=mymodel structure=mystructure.pdb experimental_data=myexperimentaldata.qis pdblist=pdblist.txt q_start=q_start q_stop=q_stop n_step=n_step output=outputfile\n"
  print "  Required arguments:\n"
  print "    mystructure             the PDB file to be evaluated (must be provided)"
  print
  print "  Optional arguments:\n"
  print "    mymodel                 the model type to be used, it should be either debye or she (default is she)"
  print "    myexperimentaldata.qis  the sas profile, columns are q, I(q), and std"
  print "    q_start, q_stop         defines the range of q"
  print "    outputfile              the file used to store computed sas profile, default is output.iq (could be overwritten)\n"



if __name__ == "__main__":
  run( sys.argv[1:] )
